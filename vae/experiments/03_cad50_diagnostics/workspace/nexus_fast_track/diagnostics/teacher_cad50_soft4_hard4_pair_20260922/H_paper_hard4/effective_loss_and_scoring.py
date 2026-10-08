import math
import torch
from torch import Tensor
from torch.nn import functional as F
from types import SimpleNamespace
EPS=1e-8
GROUPS=("tp","tn","fp","fn")
LOSS_MODE="soft4"
CALL_COUNTS={"edge":0,"face":0}

def set_mode(mode):
    global LOSS_MODE
    assert mode in ["soft4", "paper_hard4"]
    LOSS_MODE=mode

def _split_spacetime(embedding: Tensor) -> tuple[Tensor, Tensor]:
    """把 ``[..., D]`` 等分为空间坐标和时间坐标。"""

    if embedding.shape[-1] % 2:
        raise ValueError("spacetime embedding dimension must be even")
    return embedding.chunk(2, dim=-1)


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
    p = s.sigmoid()
    weights = torch.stack([y*p, (1-y)*(1-p), (1-y)*p, y*(1-p)])
    bce = F.binary_cross_entropy_with_logits(s, y, reduction='none')
    return (weights*bce).sum(dim=1, dtype=torch.float32), weights.sum(dim=1, dtype=torch.float32)


teacher=SimpleNamespace(_all_pair_chunks=_all_pair_chunks)
probe=SimpleNamespace(first_order_interval=first_order_interval)

def edge_reconstruction_loss(z, keys, chunk, scale):
    CALL_COUNTS["edge"] += 1
    numerator = torch.zeros(4, device=z.device, dtype=torch.float32)
    mass = torch.zeros_like(numerator)
    for pair in teacher._all_pair_chunks(len(z), z.device, chunk):
        k = pair[:, 0]*len(z)+pair[:, 1]
        at = torch.searchsorted(keys, k)
        y = (at < len(keys)) & (keys[at.clamp_max(len(keys)-1)] == k)
        logits = probe.first_order_interval(z[pair[:, 0]], z[pair[:, 1]])*scale
        ns, ms = task_sums(logits, y)
        numerator = numerator+ns
        mass = mass+ms
    means = group_means(numerator, mass)
    stats = {g: dict(mass=float(mass[j]), numerator=float(numerator[j].detach()), mean=float(means[j].detach()))
             for j,g in enumerate(GROUPS)}
    return means.mean(), stats


def hard4_sums(logits, labels):
    s=logits.float()
    positive=logits.detach()>0
    gt=labels==1
    masks=torch.stack([gt&positive, (~gt)&(~positive), (~gt)&positive, gt&(~positive)])
    bce=F.binary_cross_entropy_with_logits(s, labels.to(s), reduction="none")
    return (masks.to(s)*bce).sum(dim=1,dtype=torch.float32), masks.sum(dim=1,dtype=torch.float32)


def task_sums(logits, labels):
    return hard4_sums(logits,labels) if LOSS_MODE=="paper_hard4" else soft4_sums(logits,labels)


def group_means(numerator, mass):
    # Empty hard groups contribute 0; this is this experiment's explicit convention.
    return numerator/mass.clamp_min(1) if LOSS_MODE=="paper_hard4" else numerator/(mass+EPS)


def face_reconstruction_loss(logits, labels, chunk):
    CALL_COUNTS["face"] += 1
    if LOSS_MODE=="soft4":
        ns,ms=soft4_sums(logits,labels)
        return (ns/(ms+EPS)).mean()
    numerator=torch.zeros(4,device=logits.device,dtype=torch.float32)
    count=torch.zeros_like(numerator)
    for start in range(0,len(logits),chunk):
        ns,cs=hard4_sums(logits[start:start+chunk],labels[start:start+chunk])
        numerator=numerator+ns
        count=count+cs
    return (numerator/count.clamp_min(1)).sum()/4
