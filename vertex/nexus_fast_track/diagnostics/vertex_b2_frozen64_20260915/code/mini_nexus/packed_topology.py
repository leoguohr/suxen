"""把不同大小的 mesh 组织成可并行计算的 Topology-AE batch。

先理解这里的边界：这个文件只改变“怎样把多个对象交给 GPU”，不改变任何
训练数学定义。每个对象自己的顶点、面、正边、正面、负面以及最终 loss 权重
都保持不变。相近大小的 mesh 会放进同一个 padded batch；过大的 mesh 会自然
成为单对象 batch，以免 padding 浪费太多算力和显存。

本文件的数据流是::

    Nexus2KSample 列表
      -> 按二次复杂度估算计算量
      -> 在 DDP ranks 之间分配对象
      -> 每个 rank 内按尺寸组成若干小组
      -> collate 成 vertices[B,Vmax,3] + vertex_mask[B,Vmax]
      -> faces/edge/face/incidence 继续以 tuple 保存

最后一点很重要：拓扑张量不被拼成一个新图，也不重排 vertex identity。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .data_2k import Nexus2KSample


@dataclass(frozen=True)
class PackedTopologyBatch:
    """Topology AE 当前阶段真正需要的最小 batch。

    ``vertices`` 的 shape 是 ``[B,Vmax,3]``；不足 ``Vmax`` 的尾部补零。
    ``vertex_mask`` 的 shape 是 ``[B,Vmax]``，True 才表示真实顶点。
    其余字段是长度 B 的 tuple：第 b 项始终只属于第 b 个 UID，因此 collate
    不会偷偷改变 Stage-3/Stage-4 已对齐的 vertex identity。
    """

    uids: tuple[str, ...]
    vertices: Tensor
    vertex_mask: Tensor
    faces: tuple[Tensor, ...]
    edge_index: tuple[Tensor, ...]
    face_set: tuple[Tensor, ...]
    incidence_index: tuple[Tensor, ...]

    def to(self, device: torch.device | str) -> "PackedTopologyBatch":
        # dataclass 是 frozen 的，不能原地修改；返回字段相同的新对象。
        # UID 是普通字符串留在 CPU，参与计算的 Tensor 才搬到 GPU。
        return PackedTopologyBatch(
            uids=self.uids,
            vertices=self.vertices.to(device),
            vertex_mask=self.vertex_mask.to(device),
            faces=tuple(value.to(device) for value in self.faces),
            edge_index=tuple(value.to(device) for value in self.edge_index),
            face_set=tuple(value.to(device) for value in self.face_set),
            incidence_index=tuple(value.to(device) for value in self.incidence_index),
        )


def collate_packed_topology(samples: list[Nexus2KSample]) -> PackedTopologyBatch:
    """只整理不可变的 mesh/topology 字段，不生成或修复任何标签。"""

    if not samples:
        raise ValueError("cannot collate an empty topology batch")
    maximum_vertices = max(len(sample.vertices) for sample in samples)
    # 只有 vertices 需要规则的 batch 维，所以在第 2 维补到本组最大 V。
    vertices = torch.zeros((len(samples), maximum_vertices, 3), dtype=torch.float32)
    vertex_mask = torch.zeros((len(samples), maximum_vertices), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        count = len(sample.vertices)
        vertices[batch_index, :count] = sample.vertices
        vertex_mask[batch_index, :count] = True
    return PackedTopologyBatch(
        uids=tuple(sample.uid for sample in samples),
        vertices=vertices,
        vertex_mask=vertex_mask,
        faces=tuple(sample.faces for sample in samples),
        edge_index=tuple(sample.edge_index for sample in samples),
        face_set=tuple(sample.face_set for sample in samples),
        incidence_index=tuple(sample.incidence_index for sample in samples),
    )


def topology_attention_cost(sample: Nexus2KSample) -> int:
    """估算一个对象的主要二次复杂度，供调度使用而不参与 loss。

    Encoder attention 看 ``V+F`` 个图节点，约为 ``(V+F)^2``；Decoder 和
    全顶点对 edge 监督主要随 ``V^2`` 增长。这里只需要相对大小，不追求
    精确的 FLOPs，所以把两项直接相加。
    """

    vertex_count = len(sample.vertices)
    node_count = vertex_count + len(sample.faces)
    return node_count * node_count + vertex_count * vertex_count


def plan_padded_batches(
    samples: list[Nexus2KSample],
    *,
    max_meshes: int,
    max_padding_ratio: float,
) -> list[list[Nexus2KSample]]:
    """把相近大小的 mesh 分组，并限制 padding 造成的二次计算浪费。

    对一个候选分组，``padding_ratio = padded_cost / true_cost``。超过上限时
    结束当前组、另开一组。排序只影响执行顺序；UID 及其全部标签仍绑定在
    同一个 ``Nexus2KSample`` 中。
    """

    if max_meshes <= 0:
        raise ValueError("max_meshes must be positive")
    if max_padding_ratio < 1.0:
        raise ValueError("max_padding_ratio must be at least 1")
    ordered = sorted(
        samples,
        key=lambda sample: (len(sample.vertices) + len(sample.faces), sample.uid),
    )
    batches: list[list[Nexus2KSample]] = []
    current: list[Nexus2KSample] = []
    for sample in ordered:
        # 先试着把 sample 加入当前组，再判断新组是否仍满足两项约束。
        proposal = current + [sample]
        vertex_max = max(len(value.vertices) for value in proposal)
        node_max = max(len(value.vertices) + len(value.faces) for value in proposal)
        padded_cost = len(proposal) * (node_max * node_max + vertex_max * vertex_max)
        true_cost = sum(topology_attention_cost(value) for value in proposal)
        padding_ratio = padded_cost / true_cost
        if current and (len(proposal) > max_meshes or padding_ratio > max_padding_ratio):
            batches.append(current)
            current = [sample]
        else:
            current = proposal
    if current:
        batches.append(current)
    return batches


def balance_samples_across_ranks(
    samples: list[Nexus2KSample], world_size: int
) -> list[list[Nexus2KSample]]:
    """用贪心法让各 DDP rank 获得接近的二次计算量。

    从最贵的对象开始，每次交给当前累计 cost 最小的 rank。它比简单地按
    mesh 数量均分更适合变长数据；但它仍只改变调度，不改变每个 mesh 在
    全局均值中的权重。
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if len(samples) < world_size:
        raise ValueError("each DDP rank needs at least one mesh")
    assignments: list[list[Nexus2KSample]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for sample in sorted(samples, key=topology_attention_cost, reverse=True):
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        assignments[rank].append(sample)
        loads[rank] += topology_attention_cost(sample)
    return assignments
