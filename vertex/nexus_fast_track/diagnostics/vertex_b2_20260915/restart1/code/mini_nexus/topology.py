"""Nexus Topology Autoencoder 与 Spacetime Interval 解码。

按下面的数据流从上到下阅读本文件::

    mesh (vertices, faces)
      -> vertex nodes + face nodes
      -> interleaved local GraphSAGE / global Transformer encoder
      -> final LayerNorm -> mean/log-variance heads
      -> one Gaussian latent per vertex
      -> attention-only decoder
      -> final LayerNorm -> separate linear heads
      -> edge and face Spacetime embeddings
      -> first/second-order intervals
      -> edge/face decisions

论文与独立实现的边界
--------------------
论文明确写了 vertex/face 图节点、交替的 GraphSAGE 与 Transformer、每顶点
64 维 latent、纯 attention decoder、一/二阶 Spacetime Interval、edge 全顶点对
监督，以及采样负 face triplet。但论文没有公开所有工程细节。本独立复现把
``encoder_layers=24`` 解释成 12 组 GraphSAGE+Transformer；Encoder Transformer
保留标准 4x FFN；Decoder attention block 不含 FFN。head 数、Encoder MLP ratio、
Decoder Q/K/V 宽度、Spacetime 维数、数值截断和空 loss 组处理均是本项目公开的
实现选择，不能说成作者官方代码细节。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Spacetime indicators and balanced binary loss.
def _center_packed_embedding(embedding: Tensor, vertex_mask: Tensor) -> Tensor:
    """Remove each mesh's translation, excluding padding from the mean."""

    counts = vertex_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    valid = vertex_mask.unsqueeze(-1)
    means = (embedding * valid).sum(dim=1, keepdim=True) / counts
    return (embedding - means).masked_fill(~valid, 0.0)


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


def _require_positive_finite(name: str, value: float) -> None:
    """Reject scoring multipliers that could erase or reverse interval signs."""

    if value <= 0 or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and positive")


def edge_interval_logits(
    left: Tensor,
    right: Tensor,
    *,
    logit_scale: float = 1.0,
) -> Tensor:
    """Scale first-order intervals into BCE logits without changing their signs."""

    _require_positive_finite("logit_scale", logit_scale)
    return logit_scale * first_order_interval(left, right)


def _squared_parallelogram_area(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """用 Gram 行列式计算 ``||(a-c) wedge (b-c)||^2``。"""

    first = a - c
    second = b - c
    return (
        first.square().sum(dim=-1) * second.square().sum(dim=-1)
        - (first * second).sum(dim=-1).square()
    ).clamp_min(0.0)


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


def _classification_masks(logits: Tensor, labels: Tensor) -> dict[str, Tensor]:
    """固定当前预测正负号，据此划分 TP/TN/FP/FN 四组。"""

    truth = labels.bool()
    predicted = logits.detach() > 0.0
    return {
        "tp": predicted & truth,
        "tn": ~predicted & ~truth,
        "fp": predicted & ~truth,
        "fn": ~predicted & truth,
    }


def paper_balanced_binary_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    counts_on_device: bool = False,
) -> tuple[Tensor, dict[str, int | Tensor]]:
    """按照 Nexus Eq. (7)，先对 TP/TN/FP/FN 各组求 BCE 均值。

    普通 BCE 会被数量巨大的 TN 主导。Nexus 先在四组内部平均，再让四组等权。
    分组时 detach，是因为分组只用于平衡权重，不是需要求导的预测操作。

    论文没有规定某组为空时怎么办；本实现只平均当前非空的组。
    """

    if logits.shape != labels.shape or not len(logits):
        raise ValueError("logits and labels must be non-empty tensors of equal shape")
    losses = F.binary_cross_entropy_with_logits(
        logits, labels.to(logits.dtype), reduction="none"
    )
    masks = _classification_masks(logits, labels)
    group_losses = [losses[mask].mean() for mask in masks.values() if mask.any()]
    if not group_losses:
        raise ValueError("at least one TP/TN/FP/FN group must be non-empty")
    counts = {name: mask.sum() for name, mask in masks.items()}
    if not counts_on_device:
        counts = {name: int(value.item()) for name, value in counts.items()}
    return torch.stack(group_losses).mean(), counts


def vae_kl_loss(mu: Tensor, log_variance: Tensor) -> Tensor:
    """Mean KL divergence from the diagonal posterior to ``N(0, I)``."""

    return -0.5 * (1.0 + log_variance - mu.square() - log_variance.exp()).mean()


def mesh_edges(faces: Tensor) -> Tensor:
    """Return sorted, unique undirected edges from triangle indices."""

    pairs = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [0, 2]]], dim=0)
    pairs = torch.sort(pairs, dim=-1).values
    return torch.unique(pairs, dim=0)


def all_vertex_pairs(vertex_count: int, device: torch.device) -> Tensor:
    """Materialize all unordered pairs; use chunked code for realistic meshes."""

    return torch.combinations(torch.arange(vertex_count, device=device), r=2)


def build_vertex_face_incidence(
    faces: Tensor, vertex_count: int
) -> tuple[Tensor, Tensor]:
    """建立 vertex-face incidence graph 的两个方向。

    vertex 节点编号为 ``0..V-1``，face 节点编号为 ``V..V+F-1``。对一个
    ``[i,j,k]`` 面建立六条有向消息：i/j/k -> face 以及 face -> i/j/k。
    此处不修 mesh、不猜 connectivity；``faces`` 就是 Stage-4 拓扑真值。
    """

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    face_nodes = (
        torch.arange(len(faces), device=faces.device, dtype=torch.long) + vertex_count
    )
    vertices = faces.long().reshape(-1)
    repeated_face_nodes = face_nodes[:, None].expand(-1, 3).reshape(-1)
    source = torch.cat([vertices, repeated_face_nodes], dim=0)
    target = torch.cat([repeated_face_nodes, vertices], dim=0)
    return source, target


def sample_face_triplets(
    faces: Tensor,
    vertex_count: int,
    *,
    negative_ratio: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Return every positive face and random non-face triplets.

    Enumerating V choose 3 is only acceptable for tiny meshes. Nexus instead
    supervises all positives plus sampled negatives, which this function does.
    """

    if negative_ratio < 0:
        raise ValueError("negative_ratio must be non-negative")
    device = faces.device
    positive = torch.unique(torch.sort(faces, dim=-1).values, dim=0)
    positive_set = {tuple(face.tolist()) for face in positive.cpu()}
    negative_count = int(round(len(positive) * negative_ratio))
    if negative_count == 0:
        labels = torch.ones(len(positive), device=device, dtype=torch.float32)
        return positive, labels

    maximum_negative = vertex_count * (vertex_count - 1) * (vertex_count - 2) // 6 - len(positive)
    negative_count = min(negative_count, maximum_negative)
    if negative_count == 0:
        labels = torch.ones(len(positive), device=device, dtype=torch.float32)
        return positive, labels
    negative_set: set[tuple[int, int, int]] = set()
    while len(negative_set) < negative_count:
        draw_count = max(4 * (negative_count - len(negative_set)), 32)
        draw = torch.randint(
            vertex_count, (draw_count, 3), generator=generator, device="cpu"
        )
        draw = torch.sort(draw, dim=-1).values
        for triplet in draw.tolist():
            key = tuple(triplet)
            if key[0] == key[1] or key[1] == key[2] or key in positive_set:
                continue
            negative_set.add(key)
            if len(negative_set) == negative_count:
                break
    negative = torch.tensor(sorted(negative_set), dtype=torch.long, device=device)
    triplets = torch.cat([positive, negative], dim=0)
    labels = torch.cat(
        [
            torch.ones(len(positive), device=device),
            torch.zeros(len(negative), device=device),
        ]
    )
    return triplets, labels


# Topology VAE encoder and decoder.
class MeanSAGEConv(nn.Module):
    """不依赖图学习库的 GraphSAGE mean aggregation。

    输出为 ``W_self*x_i + W_neighbor*mean(x_j for j -> i)``，负责 encoder 中
    局部、知道拓扑连接关系的信息传递。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, nodes: Tensor, source: Tensor, target: Tensor) -> Tensor:
        # nodes 是 [V+F,C]。index_add_ 按 target 把所有 source feature 相加，
        # 再除以入度得到 mean neighbor feature，不需要 torch-geometric。
        neighbor_sum = torch.zeros_like(nodes)
        neighbor_sum.index_add_(0, target, nodes[source])
        counts = torch.zeros(
            (len(nodes), 1), dtype=nodes.dtype, device=nodes.device
        )
        counts.index_add_(
            0,
            target,
            torch.ones((len(target), 1), dtype=nodes.dtype, device=nodes.device),
        )
        neighbor_mean = neighbor_sum / counts.clamp_min(1.0)
        return self.self_projection(nodes) + self.neighbor_projection(neighbor_mean)


class GraphTransformerBlock(nn.Module):
    """先做一层局部 GraphSAGE，再做一层全局 Transformer。

    GraphSAGE 只沿 vertex-face incidence edge 通信；全局 self-attention 再让
    远处节点和不同组件交换信息。二者交替是论文 topology encoder 的核心。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.graph = MeanSAGEConv(hidden_dim)
        self.graph_activation = nn.SiLU()
        self.transformer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, nodes: Tensor, source: Tensor, target: Tensor) -> Tensor:
        # Pre-normalize, aggregate local neighbors, and add a residual update.
        graph_update = self.graph(self.graph_norm(nodes), source, target)
        nodes = nodes + self.graph_activation(graph_update)
        # The leading batch dimension is temporary: this function handles one mesh.
        return self.transformer(nodes.unsqueeze(0)).squeeze(0)

    def forward_packed(
        self,
        nodes: Tensor,
        node_mask: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Process padded, independent graphs without cross-mesh attention."""

        batch_size, maximum_nodes, hidden_dim = nodes.shape
        flat_nodes = nodes.reshape(batch_size * maximum_nodes, hidden_dim)
        graph_update = self.graph(self.graph_norm(flat_nodes), source, target)
        nodes = nodes + self.graph_activation(graph_update.reshape_as(nodes))
        nodes = nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        nodes = self.transformer(nodes, src_key_padding_mask=~node_mask)
        return nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)


class AttentionBlock(nn.Module):
    """用于纯 attention decoder 的 pre-norm self-attention 残差块。

    论文只公开 decoder 是 pure attention、hidden=1024，但未公开 Q/K/V 投影
    维度、head 数或内部 bottleneck。这里采用 PyTorch 标准 full-width MHA：
    Q/K/V 和输出都保持 hidden_dim；不加入 FFN。两者都是明确的独立实现选择。
    使用论文公开宽度时，总参数量也与论文约 100M 的量级一致。
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

    def forward(self, nodes: Tensor) -> Tensor:
        normalized = self.norm(nodes).unsqueeze(0)
        update, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        return nodes + update.squeeze(0)

    def forward_packed(self, nodes: Tensor, node_mask: Tensor) -> Tensor:
        """Apply full attention inside each padded mesh and nowhere else."""

        normalized = self.norm(nodes)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~node_mask,
            need_weights=False,
        )
        nodes = nodes + update
        return nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)


class TopologyAutoencoder(nn.Module):
    """带混合图 encoder 的“每顶点 latent”变分自编码器。

    ``hidden_dim`` 控制 encoder 宽度；``encoder_layers`` 统计单独层数，必须为
    偶数，因为每两层组成一组 GraphSAGE+Transformer；``decoder_layers`` 统计
    attention block；decoder 可以比 encoder 更宽。论文公开的 24/512 encoder
    与 16/1024 decoder 由同一个类实例化。
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        spacetime_dim: int = 16,
        num_heads: int = 4,
        num_layers: int | None = None,
        *,
        encoder_layers: int | None = None,
        decoder_layers: int | None = None,
        decoder_hidden_dim: int | None = None,
        encoder_dropout: float = 0.1,
    ):
        super().__init__()
        if spacetime_dim % 2:
            raise ValueError("spacetime_dim must be even")
        if encoder_layers is None:
            encoder_layers = 24 if num_layers is None else 2 * num_layers
        if decoder_layers is None:
            decoder_layers = 16 if num_layers is None else num_layers
        if decoder_hidden_dim is None:
            decoder_hidden_dim = 1024 if num_layers is None else hidden_dim
        if encoder_layers <= 0 or encoder_layers % 2:
            raise ValueError("encoder_layers must be a positive even number")
        if decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        if hidden_dim % num_heads or decoder_hidden_dim % num_heads:
            raise ValueError("encoder and decoder dimensions must divide num_heads")
        self.vertex_input = nn.Linear(3, hidden_dim)
        # A face node starts from its geometric centroid, not from a learned face ID.
        self.face_input = nn.Linear(3, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            GraphTransformerBlock(hidden_dim, num_heads, dropout=encoder_dropout)
            for _ in range(encoder_layers // 2)
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        self.encoder_output_norm = nn.LayerNorm(hidden_dim)

        # Nexus Eq. (6), PDF p.5 L033-L040 写作 Z=Dec(H_V)，没有再次注入 V。
        # 因而 paper-aligned 路径只把每顶点 latent 投影到 decoder hidden width。
        self.latent_input = nn.Linear(latent_dim, decoder_hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            AttentionBlock(decoder_hidden_dim, num_heads)
            for _ in range(decoder_layers)
        )
        self.edge_embedding = nn.Linear(decoder_hidden_dim, spacetime_dim)
        self.face_embedding = nn.Linear(decoder_hidden_dim, spacetime_dim)
        self.decoder_output_norm = nn.LayerNorm(decoder_hidden_dim)

    def encode(
        self,
        vertices: Tensor,
        faces: Tensor,
        incidence_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """把 ``vertices[V,3], faces[F,3]`` 编码为 ``mu/logvar[V,L]``。"""

        # Create V vertex-node features directly from normalized coordinates.
        vertex_features = self.vertex_input(vertices)
        # Create F face-node features from the mean of each face's three vertices.
        centroids = vertices[faces].mean(dim=1)
        face_features = self.face_input(centroids)
        # The encoder sequence has V+F nodes; vertex identity remains 0..V-1.
        nodes = torch.cat([vertex_features, face_features], dim=0)
        if incidence_index is None:
            source, target = build_vertex_face_incidence(faces, len(vertices))
        else:
            if incidence_index.ndim != 2 or incidence_index.shape[0] != 2:
                raise ValueError("incidence_index must have shape [2,I]")
            source, target = incidence_index[0], incidence_index[1]
        for block in self.encoder_blocks:
            nodes = block(nodes, source, target)
        # Face nodes help encode topology but only vertex nodes receive latents.
        vertex_hidden = self.encoder_output_norm(nodes[: len(vertices)])
        return self.mu(vertex_hidden), self.log_variance(vertex_hidden).clamp(-10.0, 10.0)

    def sample(self, mu: Tensor, log_variance: Tensor) -> Tensor:
        """训练时使用 VAE 重参数化技巧采样 latent。"""

        if self.training:
            return mu + torch.exp(0.5 * log_variance) * torch.randn_like(mu)
        return mu

    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """按论文 Eq. (6) 把每顶点 latent 解码成 edge/face 两套 embedding。"""

        hidden = self.latent_input(latent)
        for block in self.decoder_blocks:
            hidden = block(hidden)
        hidden = self.decoder_output_norm(hidden)
        # Only center the output coordinates; spacetime intervals ignore translation.
        edge_embedding = self.edge_embedding(hidden)
        face_embedding = self.face_embedding(hidden)
        edge_embedding = edge_embedding - edge_embedding.mean(dim=0, keepdim=True)
        face_embedding = face_embedding - face_embedding.mean(dim=0, keepdim=True)
        return edge_embedding, face_embedding

    def forward(
        self,
        vertices: Tensor,
        faces: Tensor,
        incidence_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, log_variance = self.encode(vertices, faces, incidence_index)
        latent = self.sample(mu, log_variance)
        edge_embedding, face_embedding = self.decode(latent)
        return latent, mu, log_variance, edge_embedding, face_embedding

    def encode_packed(
        self,
        vertices: Tensor,
        vertex_mask: Tensor,
        faces: tuple[Tensor, ...],
        incidence_index: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        """Encode several variable-length meshes in one padded Transformer call."""

        batch_size, maximum_vertices, _ = vertices.shape
        vertex_counts = vertex_mask.sum(dim=1).tolist()
        node_counts = [count + len(face) for count, face in zip(vertex_counts, faces)]
        maximum_nodes = max(node_counts)
        nodes = vertices.new_zeros((batch_size, maximum_nodes, self.mu.in_features))
        node_mask = torch.zeros(
            (batch_size, maximum_nodes), dtype=torch.bool, device=vertices.device
        )
        source_rows = []
        target_rows = []
        for batch_index, (vertex_count, face, incidence) in enumerate(
            zip(vertex_counts, faces, incidence_index)
        ):
            vertex_count = int(vertex_count)
            vertex_features = self.vertex_input(vertices[batch_index, :vertex_count])
            centroids = vertices[batch_index, :vertex_count][face].mean(dim=1)
            face_features = self.face_input(centroids)
            node_count = vertex_count + len(face)
            nodes[batch_index, :node_count] = torch.cat(
                [vertex_features, face_features], dim=0
            )
            node_mask[batch_index, :node_count] = True
            offset = batch_index * maximum_nodes
            source_rows.append(incidence[0] + offset)
            target_rows.append(incidence[1] + offset)
        source = torch.cat(source_rows)
        target = torch.cat(target_rows)
        for block in self.encoder_blocks:
            nodes = block.forward_packed(nodes, node_mask, source, target)

        vertex_hidden = self.encoder_output_norm(nodes[:, :maximum_vertices])
        mu = self.mu(vertex_hidden).masked_fill(~vertex_mask.unsqueeze(-1), 0.0)
        log_variance = self.log_variance(vertex_hidden).clamp(-10.0, 10.0)
        log_variance = log_variance.masked_fill(~vertex_mask.unsqueeze(-1), 0.0)
        return mu, log_variance

    def sample_packed(
        self,
        mu: Tensor,
        log_variance: Tensor,
        vertex_mask: Tensor,
        sample_seeds: tuple[int, ...] | None,
    ) -> Tensor:
        """Sample only valid vertices, optionally with UID-stable seeds."""

        if not self.training:
            return mu
        latent = torch.zeros_like(mu)
        for batch_index, mask in enumerate(vertex_mask):
            if sample_seeds is None:
                noise = torch.randn_like(mu[batch_index, mask])
            else:
                generator = torch.Generator(device=mu.device)
                generator.manual_seed(sample_seeds[batch_index])
                noise = torch.randn(
                    mu[batch_index, mask].shape,
                    dtype=mu.dtype,
                    device=mu.device,
                    generator=generator,
                )
            latent[batch_index, mask] = (
                mu[batch_index, mask]
                + torch.exp(0.5 * log_variance[batch_index, mask]) * noise
            )
        return latent

    def decode_packed(
        self, latent: Tensor, vertex_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Decode padded per-vertex latents with mesh-isolated attention."""

        hidden = self.latent_input(latent)
        hidden = hidden.masked_fill(~vertex_mask.unsqueeze(-1), 0.0)
        for block in self.decoder_blocks:
            hidden = block.forward_packed(hidden, vertex_mask)
        hidden = self.decoder_output_norm(hidden)
        edge_embedding = self.edge_embedding(hidden)
        face_embedding = self.face_embedding(hidden)
        return (
            _center_packed_embedding(edge_embedding, vertex_mask),
            _center_packed_embedding(face_embedding, vertex_mask),
        )

    def forward_packed(
        self,
        vertices: Tensor,
        vertex_mask: Tensor,
        faces: tuple[Tensor, ...],
        incidence_index: tuple[Tensor, ...],
        *,
        sample_seeds: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Packed equivalent of :meth:`forward` for independent meshes."""

        mu, log_variance = self.encode_packed(
            vertices, vertex_mask, faces, incidence_index
        )
        latent = self.sample_packed(mu, log_variance, vertex_mask, sample_seeds)
        edge_embedding, face_embedding = self.decode_packed(latent, vertex_mask)
        return latent, mu, log_variance, edge_embedding, face_embedding


def topology_autoencoder_loss(
    vertices: Tensor,
    faces: Tensor,
    mu: Tensor,
    log_variance: Tensor,
    edge_embedding: Tensor,
    face_embedding: Tensor,
    *,
    kl_weight: float = 1e-4,
    negative_face_ratio: float = 1.0,
    edge_logit_scale: float = 1.0,
    face_logit_scale: float = 1.0,
    face_interval_factor: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Reference loss for small meshes.

    It materializes all V choose 2 edge pairs and is therefore for tests/small
    objects.  Formal training uses the mathematically identical chunked version
    in ``training_2k.py`` to avoid an O(V^2) pair tensor in GPU memory.
    """

    device = vertices.device
    pairs = all_vertex_pairs(len(vertices), device)
    true_edges = {tuple(edge.tolist()) for edge in mesh_edges(faces).cpu()}
    edge_labels = torch.tensor(
        [tuple(pair.tolist()) in true_edges for pair in pairs.cpu()],
        device=device,
        dtype=vertices.dtype,
    )
    edge_logits = edge_interval_logits(
        edge_embedding[pairs[:, 0]],
        edge_embedding[pairs[:, 1]],
        logit_scale=edge_logit_scale,
    )
    edge_loss, _ = paper_balanced_binary_loss(edge_logits, edge_labels)

    triplets, face_labels = sample_face_triplets(
        faces,
        len(vertices),
        negative_ratio=negative_face_ratio,
        generator=generator,
    )
    face_labels = face_labels.to(vertices.dtype)
    face_logits = face_interval_logits(
        face_embedding[triplets[:, 0]],
        face_embedding[triplets[:, 1]],
        face_embedding[triplets[:, 2]],
        logit_scale=face_logit_scale,
        area_factor=face_interval_factor,
    )
    face_loss, _ = paper_balanced_binary_loss(face_logits, face_labels)
    kl_loss = vae_kl_loss(mu, log_variance)
    total = edge_loss + face_loss + kl_weight * kl_loss
    return total, {"edge": edge_loss, "face": face_loss, "kl": kl_loss}


# Deterministic topology recovery and orientation post-processing.
@torch.no_grad()
def enumerate_edge_triangles(edges: Tensor, vertex_count: int) -> Tensor:
    """Return every sorted 3-cycle in an undirected edge graph."""

    neighbors = [set() for _ in range(vertex_count)]
    for left, right in edges.cpu().tolist():
        neighbors[left].add(right)
        neighbors[right].add(left)

    triangles = []
    for left in range(vertex_count):
        for middle in (value for value in neighbors[left] if value > left):
            triangles.extend(
                (left, middle, right)
                for right in neighbors[left].intersection(neighbors[middle])
                if right > middle
            )
    if not triangles:
        return torch.empty((0, 3), dtype=torch.long, device=edges.device)
    return torch.tensor(triangles, dtype=torch.long, device=edges.device)


@torch.no_grad()
def orient_faces_consistently(vertices: Tensor, faces: Tensor) -> Tensor:
    """Give recovered faces a deterministic, locally consistent winding.

    Nexus states that a separate orientation correction is required, but does
    not publish its algorithm.  This independent post-process propagates
    opposite shared-edge directions across manifold regions.  A closed region
    is then flipped as a whole when its signed volume is negative.  Boundary
    regions receive consistent local winding but have no intrinsic outside;
    non-manifold edges are deliberately not used for propagation.
    """

    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("faces must have shape [F,3]")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices must have shape [V,3]")
    if not len(faces):
        return faces.clone()

    face_rows = faces.cpu().tolist()
    edge_uses: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for face_index, (a, b, c) in enumerate(face_rows):
        for start, end in ((a, b), (b, c), (c, a)):
            edge = (min(start, end), max(start, end))
            edge_uses.setdefault(edge, []).append((face_index, start < end))

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in face_rows]
    for uses in edge_uses.values():
        if len(uses) != 2:
            continue
        (left, left_forward), (right, right_forward) = uses
        must_differ = left_forward == right_forward
        adjacency[left].append((right, must_differ))
        adjacency[right].append((left, must_differ))

    flips: list[bool | None] = [None] * len(face_rows)
    components: list[list[int]] = []
    for root in range(len(face_rows)):
        if flips[root] is not None:
            continue
        flips[root] = False
        component = []
        stack = [root]
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor, must_differ in adjacency[current]:
                expected = bool(flips[current]) ^ must_differ
                if flips[neighbor] is None:
                    flips[neighbor] = expected
                    stack.append(neighbor)
        components.append(component)

    oriented = faces.clone()
    for face_index, flip in enumerate(flips):
        if flip:
            oriented[face_index] = oriented[face_index, [0, 2, 1]]

    # Signed volume has an unambiguous sign only for closed manifold regions.
    vertices_cpu = vertices.detach().cpu()
    oriented_cpu = oriented.cpu()
    for component in components:
        component_set = set(component)
        closed = all(
            len([face for face, _ in uses if face in component_set]) == 2
            for uses in edge_uses.values()
            if any(face in component_set for face, _ in uses)
        )
        if not closed:
            continue
        triangles = vertices_cpu[oriented_cpu[component]]
        signed_volume = torch.sum(
            triangles[:, 0]
            * torch.cross(triangles[:, 1], triangles[:, 2], dim=1)
        )
        if signed_volume < 0:
            oriented[component] = oriented[component][:, [0, 2, 1]]
            oriented_cpu[component] = oriented[component].cpu()
    return oriented


@torch.no_grad()
def recover_topology(
    edge_embedding: Tensor,
    face_embedding: Tensor,
    *,
    edge_threshold: float = 0.0,
    face_threshold: float = 0.0,
    edge_logit_scale: float = 1.0,
    face_logit_scale: float = 1.0,
    face_interval_factor: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """按照论文要求的顺序恢复 topology。

    一阶 interval 先产生 edge graph。只有图中的 3-cycle 才可能成为 face，
    因此二阶测试只检查这些 cycle，而不是全部 ``V choose 3``。返回的 face
    尚无方向；orientation correction 属于后续确定性处理。非零 threshold 的单位是
    乘过对应 scale/factor 后的 scaled-logit；论文的零阈值不受正乘数影响。
    """

    # Preserve eager validation even when no predicted edge triangle reaches the
    # face scorer and the function returns early.
    _require_positive_finite("face_logit_scale", face_logit_scale)
    _require_positive_finite("face_interval_factor", face_interval_factor)
    vertex_count = len(edge_embedding)
    pairs = all_vertex_pairs(vertex_count, edge_embedding.device)
    edge_score = edge_interval_logits(
        edge_embedding[pairs[:, 0]],
        edge_embedding[pairs[:, 1]],
        logit_scale=edge_logit_scale,
    )
    edges = pairs[edge_score > edge_threshold]
    candidates = enumerate_edge_triangles(edges, vertex_count)
    if not len(candidates):
        return edges, torch.empty((0, 3), dtype=torch.long, device=edge_embedding.device)

    face_score = face_interval_logits(
        face_embedding[candidates[:, 0]],
        face_embedding[candidates[:, 1]],
        face_embedding[candidates[:, 2]],
        logit_scale=face_logit_scale,
        area_factor=face_interval_factor,
    )
    return edges, candidates[face_score > face_threshold]
