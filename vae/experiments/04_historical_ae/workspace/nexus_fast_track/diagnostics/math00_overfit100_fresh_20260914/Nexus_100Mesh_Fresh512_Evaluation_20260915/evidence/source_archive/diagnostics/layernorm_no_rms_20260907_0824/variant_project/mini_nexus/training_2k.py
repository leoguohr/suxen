"""在冻结的 Nexus2K manifest 上进行单卡 Topology AE 训练。

本文件连接三个不能混淆的部分：

1. ``TopologyAutoencoder`` 预测每顶点 edge/face embedding；
2. Stage-4 ``edge_index`` 与 ``face_set`` 提供不可修改的 ground truth；
3. edge loss 检查所有无序顶点对；face loss 使用全部正 face 加采样负 triplet。

全顶点对 edge 规则和 TP/TN/FP/FN 平衡来自论文。chunk 大小、face 负样本文件、
随机种子以及逐对象执行，都是在此明确记录的独立工程选择。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .negative_candidates import (
    SampledTopologyNegatives,
    TopologyNegativeCandidateStore,
)
from .topology import (
    TopologyAutoencoder,
    edge_interval_logits,
    face_interval_logits,
    first_order_interval,
    paper_balanced_binary_loss,
    sample_face_triplets,
    second_order_interval,
    vae_kl_loss,
)

TopologyEmbeddingRows = tuple[
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]


# Checkpoint compatibility and exact all-pairs edge supervision.
def topology_architecture_from_saved_args(
    saved_args: dict[str, object]
) -> dict[str, int]:
    """Recover architecture arguments needed to load a checkpoint strictly.

    New checkpoints store explicit encoder/decoder fields.  The final branch is
    compatibility code for historical mini-Nexus checkpoints; it is not used to
    reinterpret a new paper-aligned run.
    """

    if "topology_encoder_layers" in saved_args:
        encoder_layers = int(saved_args["topology_encoder_layers"])
        decoder_layers = int(saved_args["topology_decoder_layers"])
        decoder_hidden_dim = int(saved_args["topology_decoder_hidden_dim"])
    elif "encoder_layers" in saved_args:
        encoder_layers = int(saved_args["encoder_layers"])
        decoder_layers = int(saved_args["decoder_layers"])
        decoder_hidden_dim = int(saved_args["decoder_hidden_dim"])
    else:
        legacy_layers = max(1, int(saved_args["num_layers"]) // 2)
        encoder_layers = 2 * legacy_layers
        decoder_layers = legacy_layers
        decoder_hidden_dim = int(saved_args["hidden_dim"])
    return {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "decoder_hidden_dim": decoder_hidden_dim,
    }


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


def _canonical_positive_edge_keys(
    positive_edges: Tensor, vertex_count: int, device: torch.device
) -> Tensor:
    """规范化真 edge，并把 ``(i,j)`` 编成整数 ``i*V+j``。

    整数 key 让每个 chunk 在 GPU 上用 ``searchsorted`` 判断真边，无需为几百万
    顶点对构造 Python set。
    """

    if positive_edges.ndim != 2:
        raise ValueError("positive_edges must have shape [2,E] or [E,2]")
    positive = positive_edges.T if positive_edges.shape[0] == 2 else positive_edges
    if positive.shape[1] != 2:
        raise ValueError("positive_edges must have shape [2,E] or [E,2]")
    positive = torch.unique(
        torch.sort(positive.long().to(device), dim=-1).values, dim=0
    )
    if len(positive) and (positive.min() < 0 or positive.max() >= vertex_count):
        raise ValueError("positive edge index is outside [0,V)")
    return torch.sort(positive[:, 0] * vertex_count + positive[:, 1]).values


def paper_edge_loss_all_pairs(
    edge_embedding: Tensor,
    positive_edges: Tensor,
    *,
    pair_chunk_size: int = 262_144,
    positive_keys: Tensor | None = None,
    counts_on_device: bool = False,
    logit_scale: float = 1.0,
) -> tuple[Tensor, dict[str, int | Tensor]]:
    """分 chunk 监督所有无序顶点对，并使用 Nexus Eq. (7)。

    本函数故意不使用采样负 edge sidecar。所有真 edge 为正，其余 ``V choose 2``
    顶点对全部为负。每个 chunk 的 BCE 按全局 TP/TN/FP/FN 累加，最后才求四组
    均值，因此改变 chunk 大小不会改变 loss。
    """

    # 输入 edge_embedding 的 shape 是 [V,D]；这里不接收抽样 pair，
    # 而是从 V 推导完整的 V(V-1)/2 个无序候选。
    vertex_count = len(edge_embedding)
    if positive_keys is None:
        positive_keys = _canonical_positive_edge_keys(
            positive_edges, vertex_count, edge_embedding.device
        )
    # 四组必须跨所有 chunk 全局累计。若先对每个 chunk 求均值再平均，
    # chunk 中正负比例不同会偷偷改变论文目标。
    group_names = ("tp", "tn", "fp", "fn")
    group_sums = torch.zeros(
        len(group_names), dtype=edge_embedding.dtype, device=edge_embedding.device
    )
    group_counts = torch.zeros(
        len(group_names), dtype=torch.long, device=edge_embedding.device
    )
    pair_count = 0
    positive_count = torch.zeros((), dtype=torch.long, device=edge_embedding.device)
    for pairs in _all_pair_chunks(
        vertex_count, edge_embedding.device, pair_chunk_size
    ):
        # 通过是否属于排序后的真 edge key，给当前 chunk 生成精确标签。
        keys = pairs[:, 0] * vertex_count + pairs[:, 1]
        locations = torch.searchsorted(positive_keys, keys)
        labels = torch.zeros(len(pairs), dtype=torch.bool, device=pairs.device)
        valid = locations < len(positive_keys)
        labels[valid] = positive_keys[locations[valid]] == keys[valid]
        # 一阶 interval 本身作为 BCE logit；大于零即预测为 edge。
        logits = edge_interval_logits(
            edge_embedding[pairs[:, 0]],
            edge_embedding[pairs[:, 1]],
            logit_scale=logit_scale,
        )
        losses = F.binary_cross_entropy_with_logits(
            logits, labels.to(logits.dtype), reduction="none"
        )
        truth = labels
        predicted = logits.detach() > 0.0
        # TP=0, TN=1, FP=2, FN=3。每个 pair 只属于一个组；scatter_add
        # 比同时展开四份 [4,chunk] mask 更省显存流量。
        group_ids = torch.where(
            truth,
            torch.where(predicted, 0, 3),
            torch.where(predicted, 2, 1),
        )
        # sum/count 留在 GPU 上跨 chunk 累加，避免每个 chunk 反复 .item()
        # 造成 CPU/GPU 同步。最终公式仍是四组各求均值后再等权平均。
        chunk_sums = torch.zeros_like(group_sums).scatter_add(
            0, group_ids, losses
        )
        group_sums = group_sums + chunk_sums
        group_counts = group_counts + torch.bincount(group_ids, minlength=4)
        pair_count += len(pairs)
        positive_count = positive_count + labels.sum()
    # 论文未公开空组处理；这里只平均当前实际存在的组。
    non_empty = group_counts > 0
    loss = (group_sums[non_empty] / group_counts[non_empty]).mean()
    if counts_on_device:
        pair_value = group_counts.sum()
        return loss, {
            **dict(zip(group_names, group_counts.unbind())),
            "pairs": pair_value,
            "positive": positive_count,
            "negative": pair_value - positive_count,
        }
    count_values = [int(value) for value in group_counts.detach().cpu().tolist()]
    positive_value = int(positive_count.detach().cpu())
    return loss, {
        **dict(zip(group_names, count_values)),
        "pairs": pair_count,
        "positive": positive_value,
        "negative": pair_count - positive_value,
    }


@torch.no_grad()
def raw_interval_second_moments(
    edge_embedding: Tensor,
    face_embedding: Tensor,
    positive_faces: Tensor,
    negative_faces: Tensor,
    *,
    pair_chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Measure unscaled interval second moments for one normalized mesh.

    The face statistic deliberately excludes both ``face_logit_scale`` and the
    optional triangle-area factor.  A calibrated scale therefore cannot cancel
    a later ``1.0`` versus ``0.25`` single-variable face experiment.
    """

    edge_square_sum = torch.zeros(
        (), dtype=torch.float64, device=edge_embedding.device
    )
    edge_count = 0
    for pairs in _all_pair_chunks(
        len(edge_embedding), edge_embedding.device, pair_chunk_size
    ):
        raw_edge_logits = first_order_interval(
            edge_embedding[pairs[:, 0]], edge_embedding[pairs[:, 1]]
        )
        edge_square_sum += raw_edge_logits.double().square().sum()
        edge_count += len(raw_edge_logits)

    positive_faces = positive_faces.long().to(face_embedding.device)
    negative_faces = negative_faces.long().to(face_embedding.device)
    face_square_sum = torch.zeros(
        (), dtype=torch.float64, device=face_embedding.device
    )
    face_count = 0
    for triplets in (positive_faces, negative_faces):
        if not len(triplets):
            continue
        raw_face_logits = second_order_interval(
            face_embedding[triplets[:, 0]],
            face_embedding[triplets[:, 1]],
            face_embedding[triplets[:, 2]],
            area_factor=1.0,
        )
        face_square_sum += raw_face_logits.double().square().sum()
        face_count += len(raw_face_logits)
    if edge_count == 0 or face_count == 0:
        raise ValueError("logit calibration requires non-empty edge and face candidates")
    return edge_square_sum / edge_count, face_square_sum / face_count


# Legacy online-negative loss retained for pilot checkpoint compatibility.
def sampled_topology_autoencoder_loss(
    vertices: Tensor,
    positive_edges: Tensor,
    positive_faces: Tensor,
    mu: Tensor,
    log_variance: Tensor,
    edge_embedding: Tensor,
    face_embedding: Tensor,
    *,
    face_negative_ratio: float = 1.0,
    kl_weight: float = 1e-4,
    edge_logit_scale: float = 1.0,
    face_logit_scale: float = 1.0,
    face_interval_factor: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Training loss when no versioned face-negative sidecar is supplied.

    Despite the historical function name, the edge term uses every pair. Only
    face negatives are randomly sampled.
    """

    edge_loss, edge_counts = paper_edge_loss_all_pairs(
        edge_embedding, positive_edges, logit_scale=edge_logit_scale
    )

    triplets, face_labels = sample_face_triplets(
        positive_faces,
        len(vertices),
        negative_ratio=face_negative_ratio,
        generator=generator,
    )
    face_logits = face_interval_logits(
        face_embedding[triplets[:, 0]],
        face_embedding[triplets[:, 1]],
        face_embedding[triplets[:, 2]],
        logit_scale=face_logit_scale,
        area_factor=face_interval_factor,
    )
    face_loss, face_counts = paper_balanced_binary_loss(
        face_logits, face_labels.to(vertices.dtype)
    )
    kl_loss = vae_kl_loss(mu, log_variance)
    total = edge_loss + face_loss + kl_weight * kl_loss
    return total, {
        "edge": edge_loss,
        "face": face_loss,
        "kl": kl_loss,
        **{f"edge_{name}": edge_loss.new_tensor(value) for name, value in edge_counts.items()},
        **{f"face_{name}": face_loss.new_tensor(value) for name, value in face_counts.items()},
    }


def topology_autoencoder_loss_with_face_negatives(
    positive_edges: Tensor,
    positive_faces: Tensor,
    negative_faces: Tensor,
    mu: Tensor,
    log_variance: Tensor,
    edge_embedding: Tensor,
    face_embedding: Tensor,
    *,
    kl_weight: float = 1e-4,
    pair_chunk_size: int = 262_144,
    positive_edge_keys: Tensor | None = None,
    edge_logit_scale: float = 1.0,
    face_logit_scale: float = 1.0,
    face_interval_factor: float = 1.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """使用全部 edge pair 和给定 face 负样本计算训练 loss。

    Edge 使用全部 ``V choose 2``，因此这里不接收也不采样 negative edge。
    Face 使用全部正例与原样提供的负 triplet，保持既定 overfit 实验不变。
    """

    device = edge_embedding.device
    positive_edges = positive_edges.T if positive_edges.shape[0] == 2 else positive_edges
    positive_edges = positive_edges.long().to(device)
    positive_faces = positive_faces.long().to(device)
    negative_faces = negative_faces.long().to(device)

    # Edge：直接监督全部 pair；不存在额外的 negative-edge 输入。
    edge_loss, edge_counts = paper_edge_loss_all_pairs(
        edge_embedding,
        positive_edges,
        pair_chunk_size=pair_chunk_size,
        positive_keys=positive_edge_keys,
        counts_on_device=True,
        logit_scale=edge_logit_scale,
    )

    # Face：所有 GT face 都是正例；sidecar 中固定 triplet 是负例。
    face_positive_logits = face_interval_logits(
        face_embedding[positive_faces[:, 0]],
        face_embedding[positive_faces[:, 1]],
        face_embedding[positive_faces[:, 2]],
        logit_scale=face_logit_scale,
        area_factor=face_interval_factor,
    )
    face_negative_logits = face_interval_logits(
        face_embedding[negative_faces[:, 0]],
        face_embedding[negative_faces[:, 1]],
        face_embedding[negative_faces[:, 2]],
        logit_scale=face_logit_scale,
        area_factor=face_interval_factor,
    )
    face_logits = torch.cat([face_positive_logits, face_negative_logits], dim=0)
    face_labels = torch.cat(
        [
            torch.ones_like(face_positive_logits),
            torch.zeros_like(face_negative_logits),
        ],
        dim=0,
    )
    face_loss, face_counts = paper_balanced_binary_loss(
        face_logits, face_labels, counts_on_device=True
    )
    # VAE KL：让每顶点 posterior N(mu,sigma^2) 不要偏离 N(0,I) 太远。
    # 系数 1e-4 很小，当前训练主要由 edge+face 重建目标驱动。
    kl_loss = vae_kl_loss(mu, log_variance)
    total = edge_loss + face_loss + kl_weight * kl_loss
    return total, {
        "edge": edge_loss,
        "face": face_loss,
        "kl": kl_loss,
        **{f"edge_{name}": value.to(edge_loss) for name, value in edge_counts.items()},
        **{f"face_{name}": value.to(face_loss) for name, value in face_counts.items()},
    }


# Current variable-length, per-mesh training wrapper.
class Nexus2KTopologyAESystem(nn.Module):
    """封装 Topology AE 与逐对象变长 loss。

    batch 中 vertex 是 padding tensor+mask，但每个 mesh 的 face/topology 长度
    不同。网络可以逐对象或 packed 执行，重建 loss 始终逐 mesh 后等权平均。
    模型容量由构造函数显式给出；paper-aligned overfit 使用论文公开的
    24/512 与 16/1024 配置。
    """

    attention_backend = "pytorch_multihead_attention"
    attention_sequence_isolation = "padding_mask"
    network_compute_precision = "fp32"
    non_flash_compute_precision = "fp32"
    flash_attention_kernel_io_precision = None
    reconstruction_compute_precision = "fp32"

    def __init__(
        self,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        spacetime_dim: int = 32,
        num_heads: int = 4,
        num_layers: int | None = None,
        encoder_layers: int | None = None,
        decoder_layers: int | None = None,
        decoder_hidden_dim: int | None = None,
        encoder_dropout: float = 0.1,
        edge_negative_ratio: float = 1.0,
        face_negative_ratio: float = 1.0,
        pair_chunk_size: int = 262_144,
        negative_candidate_store: TopologyNegativeCandidateStore | None = None,
        fixed_overfit_face_negatives: bool = False,
        normalize_spacetime_embeddings: bool = False,
        embedding_normalization_eps: float = 1e-6,
        edge_logit_scale: float = 1.0,
        face_logit_scale: float = 1.0,
        face_interval_factor: float = 1.0,
    ):
        super().__init__()
        # Retained only so historical configs/scripts still instantiate. Edge
        # supervision is always all-pairs, so this value must not enter the loss.
        del edge_negative_ratio
        self.face_negative_ratio = face_negative_ratio
        self.pair_chunk_size = pair_chunk_size
        self.negative_candidate_store = negative_candidate_store
        self.fixed_overfit_face_negatives = fixed_overfit_face_negatives
        self.set_fixed_logit_scales(edge_logit_scale, face_logit_scale)
        if face_interval_factor <= 0 or not math.isfinite(face_interval_factor):
            raise ValueError("face_interval_factor must be finite and positive")
        self.face_interval_factor = float(face_interval_factor)
        self.last_loss_components: dict[str, float] = {}
        self._positive_edge_key_cache: dict[tuple[str, int, str], Tensor] = {}
        self._fixed_overfit_negative_cache: dict[str, SampledTopologyNegatives] = {}
        self.autoencoder = TopologyAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            spacetime_dim=spacetime_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            encoder_dropout=encoder_dropout,
            normalize_spacetime_embeddings=normalize_spacetime_embeddings,
            embedding_normalization_eps=embedding_normalization_eps,
        )

    @classmethod
    def from_saved_args(
        cls, saved_args: dict[str, object]
    ) -> "Nexus2KTopologyAESystem":
        """Rebuild one evaluation system from checkpoint metadata."""

        return cls(
            hidden_dim=int(saved_args["hidden_dim"]),
            latent_dim=int(saved_args["latent_dim"]),
            spacetime_dim=int(saved_args["spacetime_dim"]),
            num_heads=int(saved_args["num_heads"]),
            **topology_architecture_from_saved_args(saved_args),
            encoder_dropout=float(saved_args.get("encoder_dropout", 0.1)),
            edge_negative_ratio=float(saved_args.get("edge_negative_ratio", 1.0)),
            face_negative_ratio=float(saved_args.get("face_negative_ratio", 1.0)),
            normalize_spacetime_embeddings=bool(
                saved_args.get("normalize_spacetime_embeddings", False)
            ),
            embedding_normalization_eps=float(
                saved_args.get("embedding_normalization_eps", 1e-6)
            ),
            edge_logit_scale=float(saved_args.get("edge_logit_scale", 1.0)),
            face_logit_scale=float(saved_args.get("face_logit_scale", 1.0)),
            face_interval_factor=float(saved_args.get("face_interval_factor", 1.0)),
        )

    def scoring_contract(self) -> dict[str, object]:
        """Return the JSON-safe scoring state required to interpret outputs."""

        normalized = bool(self.autoencoder.normalize_spacetime_embeddings)
        return {
            "embedding_normalization": (
                "per_mesh_center_global_unit_rms"
                if normalized
                else "per_mesh_center_only"
            ),
            "normalize_spacetime_embeddings": normalized,
            "embedding_normalization_eps": float(
                self.autoencoder.embedding_normalization_eps
            ),
            "edge_logit_scale": self.edge_logit_scale,
            "face_logit_scale": self.face_logit_scale,
            "face_interval_factor": self.face_interval_factor,
            "zero_threshold_sign_preserved": True,
        }

    def runtime_contract(self) -> dict[str, object]:
        """Return the execution backend needed for faithful reconstruction."""

        return {
            "training_system": type(self).__name__,
            "attention_backend": self.attention_backend,
            "attention_sequence_isolation": self.attention_sequence_isolation,
            "network_compute_precision": self.network_compute_precision,
            "non_flash_compute_precision": self.non_flash_compute_precision,
            "flash_attention_kernel_io_precision": (
                self.flash_attention_kernel_io_precision
            ),
            "reconstruction_compute_precision": self.reconstruction_compute_precision,
        }

    def get_extra_state(self) -> dict[str, object]:
        """Persist fixed scoring controls alongside model tensors."""

        return {"format_version": 1, **self.scoring_contract()}

    def set_extra_state(self, state: object) -> None:
        """Restore and validate scoring controls from ``state_dict``."""

        if not isinstance(state, dict) or state.get("format_version") != 1:
            raise ValueError("unsupported Topology AE scoring state")
        self.set_fixed_logit_scales(
            float(state["edge_logit_scale"]),
            float(state["face_logit_scale"]),
        )
        face_interval_factor = float(state["face_interval_factor"])
        if face_interval_factor <= 0 or not math.isfinite(face_interval_factor):
            raise ValueError("face_interval_factor must be finite and positive")
        normalization_eps = float(state["embedding_normalization_eps"])
        if normalization_eps <= 0 or not math.isfinite(normalization_eps):
            raise ValueError("embedding_normalization_eps must be finite and positive")
        self.face_interval_factor = face_interval_factor
        self.autoencoder.normalize_spacetime_embeddings = bool(
            state["normalize_spacetime_embeddings"]
        )
        self.autoencoder.embedding_normalization_eps = normalization_eps
        expected_mode = (
            "per_mesh_center_global_unit_rms"
            if self.autoencoder.normalize_spacetime_embeddings
            else "per_mesh_center_only"
        )
        if state.get("embedding_normalization", expected_mode) != expected_mode:
            raise ValueError("inconsistent embedding normalization scoring state")

    def _load_from_state_dict(
        self,
        state_dict: dict[str, object],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Make pre-scoring-state checkpoints strict-load as their old defaults."""

        extra_state_key = prefix + "_extra_state"
        if extra_state_key not in state_dict:
            state_dict[extra_state_key] = self.get_extra_state()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def set_fixed_logit_scales(
        self, edge_logit_scale: float, face_logit_scale: float
    ) -> None:
        """Set frozen positive branch scales used after interval calculation."""

        for name, value in (
            ("edge_logit_scale", edge_logit_scale),
            ("face_logit_scale", face_logit_scale),
        ):
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, float(value))

    def forward(
        self,
        batch: object,
        *,
        negative_seed: int,
        fixed_negatives: bool = False,
        sample_seed_offset: int = 0,
        packed: bool = False,
        sample_seeds: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return the mean object loss and expose averaged components for logs.

        ``sample_seed_offset`` is the global index of this micro-batch's first
        mesh.  It keeps face-negative sampling identical when the same effective
        batch is split into different micro-batch sizes.
        """

        if packed:
            return self._forward_packed(
                batch,
                negative_seed=negative_seed,
                fixed_negatives=fixed_negatives,
                sample_seed_offset=sample_seed_offset,
                sample_seeds=sample_seeds,
            )

        # batch 内对象 V/F 不同，因此不能直接 stack 成统一的 [B,N,C] attention。
        # 这里 enumerate(zip(...)) 每次只取 batch 中的一个 mesh：
        # sample_index 是 Python int（0,1,...,B-1），只用来访问同一个对象的
        # uid、vertices、faces、edge/face 标签；它不是数据集下标，也不是 batch。
        # 因而 B=20 时会循环20次，分别构图、编码、解码和算 loss。最后再对
        # “对象”取平均，而不是把大 mesh 的候选数量当成额外权重。
        losses = []
        component_rows: list[dict[str, Tensor]] = []
        incidences = getattr(batch, "incidence_index", (None,) * len(batch.faces))
        for sample_index, (
            vertices_padded,
            mask,
            faces,
            positive_edges,
            positive_faces,
            incidence_index,
        ) in enumerate(
            zip(
                batch.vertices,
                batch.vertex_mask,
                batch.faces,
                batch.edge_index,
                batch.face_set,
                incidences,
            )
        ):
            # 此时 vertices_padded 对应单个对象，shape 是 [V_max,3]，不再有
            # batch 维；mask 选出这个对象实际存在的 V 个顶点。
            # 去掉 batch padding；Stage-4 faces 引用的就是这些精确 vertex ID。
            vertices = vertices_padded[mask]
            # autoencoder 此次只收到一个 mesh 的 vertices[V,3]、faces[F,3]。
            # encoder 看到 GT mesh topology；decoder 输出每顶点的两套坐标编码。
            _, mu, log_variance, edge_embedding, face_embedding = self.autoencoder(
                vertices, faces, incidence_index
            )
            if self.negative_candidate_store is None:
                # 旧分支：使用 seed 可复现地在线采样 face 负样本。
                generator = torch.Generator(device="cpu")
                generator.manual_seed(negative_seed + sample_index)
                loss, components = sampled_topology_autoencoder_loss(
                    vertices,
                    positive_edges,
                    positive_faces,
                    mu,
                    log_variance,
                    edge_embedding,
                    face_embedding,
                    face_negative_ratio=self.face_negative_ratio,
                    edge_logit_scale=self.edge_logit_scale,
                    face_logit_scale=self.face_logit_scale,
                    face_interval_factor=self.face_interval_factor,
                    generator=generator,
                )
            else:
                loss, components = self._loss_with_face_negatives(
                    uid=batch.uids[sample_index],
                    vertex_count=len(vertices),
                    positive_edges=positive_edges,
                    positive_faces=positive_faces,
                    mu=mu,
                    log_variance=log_variance,
                    edge_embedding=edge_embedding,
                    face_embedding=face_embedding,
                    negative_seed=negative_seed,
                    fixed_negatives=fixed_negatives,
                    sample_seed_offset=sample_seed_offset + sample_index,
                )
            component_rows.append(components)
            losses.append(loss)
        self._record_components(component_rows)
        # losses 中每个标量属于一个 mesh。stack 后得到 [B]，mean 才形成这个
        # Nexus2KBatch 的平均 loss。虽然统计意义上的 batch 是 B，但昂贵的
        # Transformer/全顶点对计算在上面是逐 mesh 串行发生的。
        return torch.stack(losses).mean()

    def _select_face_negatives(
        self,
        *,
        uid: str,
        vertex_count: int,
        positive_edges: Tensor,
        positive_faces: Tensor,
        negative_seed: int,
        fixed_negatives: bool,
        sample_seed_offset: int,
    ) -> SampledTopologyNegatives:
        """Select one mesh's face negatives, reusing fixed-overfit samples."""

        if self.negative_candidate_store is None:
            raise RuntimeError("face-negative loss requires a candidate store")
        if self.fixed_overfit_face_negatives:
            sampled = self._fixed_overfit_negative_cache.get(uid)
            if sampled is None:
                sampled = self.negative_candidate_store.sample_fixed_overfit_faces(
                    uid,
                    positive_edges=positive_edges,
                    positive_faces=positive_faces,
                    vertex_count=vertex_count,
                    seed=negative_seed,
                )
                self._fixed_overfit_negative_cache[uid] = sampled
            return sampled
        return self.negative_candidate_store.sample(
            uid,
            positive_edge_count=positive_edges.shape[-1],
            positive_face_count=len(positive_faces),
            seed=negative_seed + sample_seed_offset,
            fixed=fixed_negatives,
            include_edges=False,
        )

    def topology_embedding_rows(
        self,
        batch: object,
        *,
        sample_seeds: tuple[int, ...] | None = None,
    ) -> TopologyEmbeddingRows:
        """Run this system's packed backend and return unpadded rows per mesh."""

        _, mu, log_variance, edge_embeddings, face_embeddings = (
            self.autoencoder.forward_packed(
                batch.vertices,
                batch.vertex_mask,
                batch.faces,
                batch.incidence_index,
                sample_seeds=sample_seeds,
            )
        )
        masks = tuple(batch.vertex_mask)
        return tuple(
            tuple(values[index, mask] for index, mask in enumerate(masks))
            for values in (mu, log_variance, edge_embeddings, face_embeddings)
        )

    @torch.no_grad()
    def _calibration_moment_sums_from_rows(
        self,
        batch: object,
        edge_rows: tuple[Tensor, ...],
        face_rows: tuple[Tensor, ...],
        *,
        negative_seed: int,
        fixed_negatives: bool,
    ) -> Tensor:
        """Return equal-mesh raw interval moment sums without computing a loss."""

        totals = torch.zeros(3, dtype=torch.float64, device=edge_rows[0].device)
        for sample_index, (edge_embedding, face_embedding) in enumerate(
            zip(edge_rows, face_rows)
        ):
            positive_edges = batch.edge_index[sample_index]
            positive_faces = batch.face_set[sample_index]
            sampled = self._select_face_negatives(
                uid=batch.uids[sample_index],
                vertex_count=len(edge_embedding),
                positive_edges=positive_edges,
                positive_faces=positive_faces,
                negative_seed=negative_seed,
                fixed_negatives=fixed_negatives,
                sample_seed_offset=sample_index,
            )
            edge_moment, face_moment = raw_interval_second_moments(
                edge_embedding,
                face_embedding,
                positive_faces,
                sampled.faces,
                pair_chunk_size=self.pair_chunk_size,
            )
            totals[0] += edge_moment
            totals[1] += face_moment
            totals[2] += 1
        return totals

    @torch.no_grad()
    def calibration_interval_moment_sums(
        self,
        batch: object,
        *,
        negative_seed: int,
        fixed_negatives: bool,
        sample_seeds: tuple[int, ...],
    ) -> Tensor:
        """Run the standard packed backend and collect calibration statistics."""

        _, _, edge_rows, face_rows = self.topology_embedding_rows(
            batch,
            sample_seeds=sample_seeds,
        )
        return self._calibration_moment_sums_from_rows(
            batch,
            edge_rows,
            face_rows,
            negative_seed=negative_seed,
            fixed_negatives=fixed_negatives,
        )

    def _loss_with_face_negatives(
        self,
        *,
        uid: str,
        vertex_count: int,
        positive_edges: Tensor,
        positive_faces: Tensor,
        mu: Tensor,
        log_variance: Tensor,
        edge_embedding: Tensor,
        face_embedding: Tensor,
        negative_seed: int,
        fixed_negatives: bool,
        sample_seed_offset: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Select one mesh's face negatives, then compute its reconstruction loss."""

        sampled = self._select_face_negatives(
            uid=uid,
            vertex_count=vertex_count,
            positive_edges=positive_edges,
            positive_faces=positive_faces,
            negative_seed=negative_seed,
            fixed_negatives=fixed_negatives,
            sample_seed_offset=sample_seed_offset,
        )
        cache_key = (uid, vertex_count, str(edge_embedding.device))
        positive_edge_keys = self._positive_edge_key_cache.get(cache_key)
        if positive_edge_keys is None:
            positive_edge_keys = _canonical_positive_edge_keys(
                positive_edges, vertex_count, edge_embedding.device
            )
            self._positive_edge_key_cache[cache_key] = positive_edge_keys
        loss, components = topology_autoencoder_loss_with_face_negatives(
            positive_edges,
            positive_faces,
            sampled.faces,
            mu,
            log_variance,
            edge_embedding,
            face_embedding,
            pair_chunk_size=self.pair_chunk_size,
            positive_edge_keys=positive_edge_keys,
            edge_logit_scale=self.edge_logit_scale,
            face_logit_scale=self.face_logit_scale,
            face_interval_factor=self.face_interval_factor,
        )
        return loss, {
            **components,
            "negative_faces": loss.new_tensor(float(len(sampled.faces))),
            **{
                f"negative_face_{source}": loss.new_tensor(float(count))
                for source, count in sampled.face_source_counts.items()
            },
        }

    def _record_components(self, rows: list[dict[str, Tensor]]) -> None:
        """Synchronize each averaged diagnostic once, not once per mesh."""

        averaged = {
            name: float(
                torch.stack([row[name].detach().to(rows[0][name]) for row in rows])
                .mean()
                .cpu()
            )
            for name in rows[0]
        }
        self.last_loss_components = averaged

    def _forward_packed(
        self,
        batch: object,
        *,
        negative_seed: int,
        fixed_negatives: bool,
        sample_seed_offset: int,
        sample_seeds: tuple[int, ...] | None,
    ) -> Tensor:
        """Run encoder/decoder in parallel, then preserve per-mesh reconstruction."""

        mu_rows, logvar_rows, edge_rows, face_rows = self.topology_embedding_rows(
            batch,
            sample_seeds=sample_seeds,
        )
        losses = []
        component_rows: list[dict[str, Tensor]] = []
        for sample_index, (positive_edges, positive_faces) in enumerate(
            zip(batch.edge_index, batch.face_set)
        ):
            vertex_count = len(mu_rows[sample_index])
            loss, components = self._loss_with_face_negatives(
                uid=batch.uids[sample_index],
                vertex_count=vertex_count,
                positive_edges=positive_edges,
                positive_faces=positive_faces,
                mu=mu_rows[sample_index],
                log_variance=logvar_rows[sample_index],
                edge_embedding=edge_rows[sample_index],
                face_embedding=face_rows[sample_index],
                negative_seed=negative_seed,
                fixed_negatives=fixed_negatives,
                sample_seed_offset=sample_seed_offset + sample_index,
            )
            losses.append(loss)
            component_rows.append(components)
        self._record_components(component_rows)
        return torch.stack(losses).mean()
