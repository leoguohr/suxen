import math
import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F
from types import SimpleNamespace


EPS=1e-8
GROUPS=('tp','tn','fp','fn')
PAIR_CHUNK=13000000


def _require_positive_finite(name: str, value: float) -> None:
    """Reject scoring multipliers that could erase or reverse interval signs."""

    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and positive")


def _split_spacetime(embedding: Tensor) -> tuple[Tensor, Tensor]:
    """把 ``[..., D]`` 等分为空间坐标和时间坐标。"""

    if embedding.shape[-1] % 2:
        raise ValueError("spacetime embedding dimension must be even")
    return embedding.chunk(2, dim=-1)


def _squared_parallelogram_area(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """用 Gram 行列式计算 ``||(a-c) wedge (b-c)||^2``。"""

    first = a - c
    second = b - c
    return (
        first.square().sum(dim=-1) * second.square().sum(dim=-1)
        - (first * second).sum(dim=-1).square()
    ).clamp_min(0.0)


def first_order_interval(left: Tensor, right: Tensor) -> Tensor:
    """计算 ``||s_u-s_v||^2 - ||t_u-t_v||^2``；大于零表示 edge。

    decoder 不直接输出 edge 概率，而是为每个顶点输出学习到的 ``(space,time)``
    坐标。顶点对由这个类 Minkowski interval 的正负号分类，因此零是自然阈值。
    """

    # 若 spacetime_dim=32，则 space/time 各 16 维；输出 shape 与输入前缀相同。
    left_space, left_time = _split_spacetime(left)
    right_space, right_time = _split_spacetime(right)
    spatial = (left_space - right_space).square().sum(dim=-1)
    temporal = (left_time - right_time).square().sum(dim=-1)
    return spatial - temporal


def second_order_interval(
    first: Tensor,
    second: Tensor,
    third: Tensor,
    *,
    area_factor: float = 1.0,
) -> Tensor:
    """计算 ``A_space^2 - A_time^2``；大于零表示 face。

    三角形面积平方是平行四边形 Gram 行列式的四分之一。历史 raw 定义使用
    ``area_factor=1``；设为 ``0.25`` 可显式恢复三角形面积平方且不改变正负号。
    """

    _require_positive_finite("area_factor", area_factor)
    # 三个输入通常都是 [N,32]，每一行对应一个候选 triplet。
    first_space, first_time = _split_spacetime(first)
    second_space, second_time = _split_spacetime(second)
    third_space, third_time = _split_spacetime(third)
    spatial_area = _squared_parallelogram_area(first_space, second_space, third_space)
    temporal_area = _squared_parallelogram_area(first_time, second_time, third_time)
    return area_factor * (spatial_area - temporal_area)


def face_interval_logits(
    first: Tensor,
    second: Tensor,
    third: Tensor,
    *,
    logit_scale: float = 1.0,
    area_factor: float = 1.0,
) -> Tensor:
    """Scale second-order intervals into BCE logits without changing their signs."""

    _require_positive_finite("logit_scale", logit_scale)
    return logit_scale * second_order_interval(
        first,
        second,
        third,
        area_factor=area_factor,
    )


def _all_pair_chunks(
    vertex_count: int,
    device: torch.device,
    chunk_size: int,
    *,
    materialize_limit: int = 20_000_000,
):
    """把每个 ``(i,j), i<j`` 恰好产生一次，避免 Python 逐行构造。

    常见 mesh 直接用 CUDA ``triu_indices`` 一次生成索引，再按 chunk 切片；
    极大 mesh 改用 flat index + ``searchsorted`` 分块生成，避免索引本身过大。
    两条路径的 pair 顺序和监督集合完全相同。
    """

    if vertex_count < 2:
        raise ValueError("edge supervision needs at least two vertices")
    if chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    total_pairs = vertex_count * (vertex_count - 1) // 2
    if total_pairs <= materialize_limit:
        pairs = torch.triu_indices(
            vertex_count,
            vertex_count,
            offset=1,
            device=device,
            dtype=torch.long,
        ).T
        for start in range(0, total_pairs, chunk_size):
            yield pairs[start : start + chunk_size]
        return

    row_lengths = torch.arange(
        vertex_count - 1, 0, -1, device=device, dtype=torch.long
    )
    row_ends = torch.cumsum(row_lengths, dim=0)
    for start in range(0, total_pairs, chunk_size):
        flat = torch.arange(
            start,
            min(total_pairs, start + chunk_size),
            device=device,
            dtype=torch.long,
        )
        left = torch.searchsorted(row_ends, flat, right=True)
        previous_end = torch.where(
            left > 0, row_ends[left - 1], torch.zeros_like(left)
        )
        right = left + 1 + flat - previous_end
        yield torch.stack((left, right), dim=1)


def soft4_sums(logits, labels):
    s = logits.float()
    y = labels.to(s)
    p = s.sigmoid()  # Membership must not contribute to the gradient.
    weights = torch.stack([y*p, (1-y)*(1-p), (1-y)*p, y*(1-p)])
    bce = F.binary_cross_entropy_with_logits(s, y, reduction='none')
    return (weights*bce).sum(dim=1, dtype=torch.float32), weights.sum(dim=1, dtype=torch.float32)


teacher=SimpleNamespace(_all_pair_chunks=_all_pair_chunks)
probe=SimpleNamespace(first_order_interval=first_order_interval)


def soft4_loss(z, keys, chunk, scale):
    numerator = torch.zeros(4, device=z.device, dtype=torch.float32)
    mass = torch.zeros_like(numerator)
    for pair in teacher._all_pair_chunks(len(z), z.device, chunk):
        k = pair[:, 0]*len(z)+pair[:, 1]
        at = torch.searchsorted(keys, k)
        y = (at < len(keys)) & (keys[at.clamp_max(len(keys)-1)] == k)
        logits = probe.first_order_interval(z[pair[:, 0]], z[pair[:, 1]])*scale
        ns, ms = soft4_sums(logits, y)
        numerator = numerator+ns
        mass = mass+ms
    means = numerator/(mass+EPS)
    stats = {g: dict(mass=float(mass[j]), numerator=float(numerator[j].detach()), mean=float(means[j].detach()))
             for j,g in enumerate(GROUPS)}
    return means.mean(), stats


def weights(logits,y):
 p=logits.float().sigmoid();y=y.float()
 return torch.stack([y*p,(1-y)*(1-p),(1-y)*p,y*(1-p)])


def soft(logits,y,w=None):
 if w is None:w=weights(logits,y)
 b=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y.float(),reduction='none')
 return (w*b).sum(1),(w.sum(1))


def edge_logits(e,pairs,scales):return first_order_interval(e[pairs[:,0]],e[pairs[:,1]])*scales['edge_logit_scale']


def face_logits(e,tris,scales):return face_interval_logits(*(e[tris[:,j]] for j in range(3)),logit_scale=scales['face_logit_scale'],area_factor=scales['face_interval_factor'])


def objective(rows,data,scales,base=None):
 terms=[];detail=[];saved=[]
 for i,d in enumerate(data):
  e=rows[2][i];n=len(e);pairs=torch.triu_indices(n,n,1,device='cuda').T
  edges=np.asarray(d['edges']);edges=edges.T if edges.shape[0]==2 else edges
  keys=np.sort(edges,axis=1);keys=np.unique(keys[:,0]*n+keys[:,1]);ids=(pairs[:,0]*n+pairs[:,1]).cpu().numpy()
  y=torch.as_tensor(np.isin(ids,keys),device='cuda',dtype=torch.float32)
  sums=[];masses=[];edge_saved=[]
  for k in range(0,len(pairs),65536):
   l=edge_logits(e,pairs[k:k+65536],scales);edge_saved.append(l.detach().cpu().numpy())
   w=None if base is None else weights(torch.as_tensor(base[i]['edge_train_logits'][k:k+65536],device='cuda'),y[k:k+65536])
   ns,ms=soft(l,y[k:k+65536],w);sums.append(ns);masses.append(ms)
  le=(torch.stack(sums).sum(0)/(torch.stack(masses).sum(0)+1e-8)).mean()
  if base is None:
   canonical=torch.as_tensor(keys,device='cuda',dtype=torch.long)
   le,_=soft4_loss(e,canonical,PAIR_CHUNK,scales['edge_logit_scale'])
  tr=torch.as_tensor(np.concatenate([d['positive'],d['mixed']]),device='cuda',dtype=torch.long)
  fy=torch.cat([torch.ones(len(d['positive']),device='cuda'),torch.zeros(len(d['mixed']),device='cuda')])
  fl=face_logits(rows[3][i],tr,scales)
  fw=None if base is None else weights(torch.as_tensor(base[i]['face_train_logits'],device='cuda'),fy)
  ns,ms=soft(fl,fy,fw);lf=(ns/(ms+1e-8)).mean();terms.append(le+lf)
  detail.append(dict(edge=float(le.detach()),face=float(lf.detach())))
  saved.append(dict(edge_train_logits=np.concatenate(edge_saved),face_train_logits=fl.detach().cpu().numpy()))
 return torch.stack(terms).mean(),detail,saved


def loss_mu804(edge, face, pool, scales):
    value, parts, saved = objective(((), (), (edge,), (face,)), [pool], scales)
    return value/4, parts, saved
