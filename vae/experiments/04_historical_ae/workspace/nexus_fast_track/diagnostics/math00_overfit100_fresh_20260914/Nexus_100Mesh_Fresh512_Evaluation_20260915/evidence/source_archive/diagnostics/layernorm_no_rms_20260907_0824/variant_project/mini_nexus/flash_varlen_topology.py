"""Topology AE 的 FP32 + BF16 FlashAttention-varlen 执行后端。

它不是第二套模型，也没有新增可训练参数。这里直接复用
``mini_nexus.topology`` 中同一组 GraphSAGE、Transformer、Decoder 和 head
权重，只把 padded attention 改写为 FlashAttention 的变长序列调用。

精度边界是刻意固定的：输入、参数、GraphSAGE、LayerNorm、QKV/output projection、
FFN、VAE、decoder residual、Spacetime、BCE/KL 和总 loss 都使用 FP32。只有送入
外部 FlashAttention-varlen 内核的 QKV 及其直接输出使用 BF16；内核输出会立刻
转回 FP32，再进入 output projection。这样满足 FlashAttention 的 dtype 限制，
同时把低精度范围限制在无法使用 FP32 的内核边界内。

多个 mesh 的 token 虽然物理拼接在一起，但 ``cu_seqlens`` 会告诉 FlashAttention
每段的起止位置，因此不同 UID 之间不会发生 attention。这一模块改变的是执行
内核和内存布局，不是模型结构或训练目标。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .training_2k import Nexus2KTopologyAESystem, TopologyEmbeddingRows
from .topology import normalize_embedding

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
except ImportError as error:  # pragma: no cover - depends on the CUDA image
    flash_attn_varlen_qkvpacked_func = None
    _FLASH_IMPORT_ERROR = error
else:
    _FLASH_IMPORT_ERROR = None


def _cu_seqlens(lengths: list[int], device: torch.device) -> Tensor:
    """把每段长度变成 FlashAttention 要求的 int32 累积边界。

    例如 lengths=[3,5] 会得到 [0,3,8]，表示 tokens[0:3] 与 tokens[3:8]
    是两个互不通信的序列。
    """

    values = torch.tensor([0, *lengths], dtype=torch.int32, device=device)
    return values.cumsum(dim=0, dtype=torch.int32)


def _flash_self_attention(
    attention: nn.MultiheadAttention,
    tokens: Tensor,
    cu_seqlens: Tensor,
    maximum_length: int,
) -> Tensor:
    """用原 ``MultiheadAttention`` 参数执行一次变长 self-attention。

    PyTorch MHA 把 Q/K/V 的权重合并存于 ``in_proj_weight``。这里先做完全相同
    的线性投影，再 reshape 为 FlashAttention 要求的
    ``[total_tokens,3,num_heads,head_dim]``，最后继续使用原输出投影。
    """

    if flash_attn_varlen_qkvpacked_func is None:
        raise RuntimeError("flash-attn is unavailable") from _FLASH_IMPORT_ERROR
    if attention.in_proj_weight is None:
        raise ValueError("separate q/k/v projection weights are not supported")
    hidden_dim = attention.embed_dim
    head_count = attention.num_heads
    head_dim = hidden_dim // head_count
    if tokens.dtype != torch.float32:
        raise TypeError(f"FlashAttention input tokens must be FP32: {tokens.dtype}")
    projection_parameters = {
        "in_proj_weight": attention.in_proj_weight,
        "in_proj_bias": attention.in_proj_bias,
        "out_proj_weight": attention.out_proj.weight,
        "out_proj_bias": attention.out_proj.bias,
    }
    non_fp32_parameters = [
        f"{name}:{value.dtype}"
        for name, value in projection_parameters.items()
        if value is not None and value.dtype != torch.float32
    ]
    if non_fp32_parameters:
        raise TypeError(
            "FlashAttention projection parameters must be FP32: "
            + ", ".join(non_fp32_parameters)
        )

    # 这里使用基线 MHA 自己的参数。即使上层误开 autocast，QKV projection
    # 仍强制使用 FP32；只有紧邻 external FlashAttention 的 I/O 转成 BF16。
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        qkv_fp32 = F.linear(
            tokens, attention.in_proj_weight, attention.in_proj_bias
        )
    if qkv_fp32.dtype != torch.float32:
        raise TypeError(f"QKV projection must be FP32: {qkv_fp32.dtype}")
    qkv_bf16 = (
        qkv_fp32.reshape(len(tokens), 3, head_count, head_dim)
        .to(dtype=torch.bfloat16)
        .contiguous()
    )
    dropout = float(attention.dropout) if attention.training else 0.0
    # cu_seqlens 提供序列边界；causal=False 表示每个 mesh 内做双向全 attention。
    attended_bf16 = flash_attn_varlen_qkvpacked_func(
        qkv_bf16,
        cu_seqlens,
        maximum_length,
        dropout_p=dropout,
        softmax_scale=None,
        causal=False,
    )
    if attended_bf16.dtype != torch.bfloat16:
        raise TypeError(
            f"FlashAttention kernel output must be BF16: {attended_bf16.dtype}"
        )
    with torch.autocast(device_type=tokens.device.type, enabled=False):
        attended_fp32 = attended_bf16.reshape(len(tokens), hidden_dim).to(
            dtype=torch.float32
        )
        output = F.linear(
            attended_fp32,
            attention.out_proj.weight,
            attention.out_proj.bias,
        )
    if output.dtype != torch.float32:
        raise TypeError(f"attention output projection must be FP32: {output.dtype}")
    return output


def _encoder_transformer_varlen(
    layer: nn.TransformerEncoderLayer,
    tokens: Tensor,
    cu_seqlens: Tensor,
    maximum_length: int,
) -> Tensor:
    """逐项复现基线 pre-norm Transformer，只替换 attention 内核。

    顺序仍是 ``LN -> attention -> residual -> LN -> FFN -> residual``；FFN、
    activation 与 dropout 都直接取自原 ``TransformerEncoderLayer``。
    """

    if not layer.norm_first:
        raise ValueError("the experiment expects the baseline pre-norm encoder")
    attention_update = _flash_self_attention(
        layer.self_attn,
        layer.norm1(tokens),
        cu_seqlens,
        maximum_length,
    )
    tokens = tokens + layer.dropout1(attention_update)
    hidden = layer.linear1(layer.norm2(tokens))
    hidden = layer.activation(hidden)
    hidden = layer.dropout(hidden)
    return tokens + layer.dropout2(layer.linear2(hidden))


def _split_rows(tokens: Tensor, lengths: list[int]) -> tuple[Tensor, ...]:
    """按原始 mesh 长度把物理拼接的 token 恢复为逐对象 tuple。"""

    return tuple(tokens.split(lengths, dim=0))


def _flash_varlen_autoencoder_forward(
    autoencoder: nn.Module,
    vertices: Tensor,
    vertex_mask: Tensor,
    faces: tuple[Tensor, ...],
    incidence_index: tuple[Tensor, ...],
    sample_seeds: tuple[int, ...] | None,
) -> tuple[
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]:
    """完成一次 packed 前向，并返回逐 mesh 的四组结果。

    输入 ``vertices`` 为 ``[B,Vmax,3]``；返回的每个 tuple 长度为 B，而第 b
    项长度为该 UID 的真实 V。这样后续 loss 仍然逐 mesh 计算并等权平均。
    """

    if vertices.dtype != torch.float32:
        raise TypeError(f"Topology AE vertices must be FP32: {vertices.dtype}")
    # mask 恢复每个对象的真实顶点数；face 数来自未 padding 的 tuple。
    vertex_counts = [int(value) for value in vertex_mask.sum(dim=1).tolist()]
    node_lengths = [count + len(face) for count, face in zip(vertex_counts, faces)]
    node_rows = []
    source_rows = []
    target_rows = []
    node_offset = 0
    # 先把各 mesh 的 vertex nodes 与 face-centroid nodes 物理拼成一条长序列。
    # incidence 加 node_offset 后成为这条长序列中的索引，但不会跨 mesh 连边。
    for batch_index, (vertex_count, face, incidence) in enumerate(
        zip(vertex_counts, faces, incidence_index)
    ):
        sample_vertices = vertices[batch_index, :vertex_count]
        vertex_features = autoencoder.vertex_input(sample_vertices)
        centroids = sample_vertices[face].mean(dim=1)
        face_features = autoencoder.face_input(centroids)
        node_rows.append(torch.cat((vertex_features, face_features), dim=0))
        source_rows.append(incidence[0] + node_offset)
        target_rows.append(incidence[1] + node_offset)
        node_offset += vertex_count + len(face)

    nodes = torch.cat(node_rows, dim=0)
    source = torch.cat(source_rows)
    target = torch.cat(target_rows)
    node_cu = _cu_seqlens(node_lengths, nodes.device)
    maximum_nodes = max(node_lengths)
    # 每个 composite block 仍执行一层局部 MeanSAGEConv 和一层全局 Transformer。
    for block in autoencoder.encoder_blocks:
        graph_update = block.graph(block.graph_norm(nodes), source, target)
        nodes = nodes + block.graph_activation(graph_update)
        nodes = _encoder_transformer_varlen(
            block.transformer, nodes, node_cu, maximum_nodes
        )

    # Encoder 序列包含 vertex+face nodes；VAE latent 只取每段开头的 vertex nodes。
    vertex_indices = []
    node_offset = 0
    for vertex_count, node_count in zip(vertex_counts, node_lengths):
        vertex_indices.append(
            torch.arange(
                node_offset,
                node_offset + vertex_count,
                dtype=torch.long,
                device=nodes.device,
            )
        )
        node_offset += node_count
    vertex_hidden = autoencoder.encoder_output_norm(nodes[torch.cat(vertex_indices)])
    mu = autoencoder.mu(vertex_hidden)
    log_variance = autoencoder.log_variance(vertex_hidden).clamp(-10.0, 10.0)

    # 训练时执行 z = mu + sigma * epsilon；评估时直接用 posterior mean。
    # UID-stable seed 保证改变 rank 分配或 pack 顺序不会改变某 UID 的 epsilon。
    if autoencoder.training:
        mu_rows = _split_rows(mu, vertex_counts)
        logvar_rows = _split_rows(log_variance, vertex_counts)
        latent_rows = []
        for sample_index, (sample_mu, sample_logvar) in enumerate(
            zip(mu_rows, logvar_rows)
        ):
            if sample_seeds is None:
                noise = torch.randn_like(sample_mu)
            else:
                generator = torch.Generator(device=sample_mu.device)
                generator.manual_seed(sample_seeds[sample_index])
                noise = torch.randn(
                    sample_mu.shape,
                    dtype=sample_mu.dtype,
                    device=sample_mu.device,
                    generator=generator,
                )
            latent_rows.append(
                sample_mu + torch.exp(0.5 * sample_logvar) * noise
            )
        latent = torch.cat(latent_rows, dim=0)
    else:
        latent = mu

    # Decoder 只处理 V 个 vertex latent；edge 与 face 使用分开的输出 head。
    hidden = autoencoder.latent_input(latent)
    vertex_cu = _cu_seqlens(vertex_counts, hidden.device)
    maximum_vertices = max(vertex_counts)
    for block in autoencoder.decoder_blocks:
        update = _flash_self_attention(
            block.attention,
            block.norm(hidden),
            vertex_cu,
            maximum_vertices,
        )
        hidden = hidden + update

    hidden = autoencoder.decoder_output_norm(hidden)

    # Center spacetime embeddings in FP32. Translation does not change an exact
    # interval; the explicit cast also guards the reconstruction boundary.
    edge_rows = tuple(
        row.float()
        for row in _split_rows(autoencoder.edge_embedding(hidden), vertex_counts)
    )
    face_rows = tuple(
        row.float()
        for row in _split_rows(autoencoder.face_embedding(hidden), vertex_counts)
    )
    if autoencoder.normalize_spacetime_embeddings:
        edge_rows = tuple(
            normalize_embedding(row, autoencoder.embedding_normalization_eps)
            for row in edge_rows
        )
        face_rows = tuple(
            normalize_embedding(row, autoencoder.embedding_normalization_eps)
            for row in face_rows
        )
    else:
        edge_rows = tuple(row - row.mean(dim=0, keepdim=True) for row in edge_rows)
        face_rows = tuple(row - row.mean(dim=0, keepdim=True) for row in face_rows)
    outputs = (
        _split_rows(mu, vertex_counts),
        _split_rows(log_variance, vertex_counts),
        edge_rows,
        face_rows,
    )
    if any(value.dtype != torch.float32 for rows in outputs for value in rows):
        raise TypeError("all non-Flash Topology AE outputs must be FP32")
    return outputs


class FlashVarlenNexus2KTopologyAESystem(Nexus2KTopologyAESystem):
    """把基类 packed 前向替换为 Flash 版本，loss 仍调用同一基类实现。"""

    attention_backend = "flash_attn_varlen_qkvpacked"
    attention_sequence_isolation = "cu_seqlens"
    network_compute_precision = "fp32_except_flash_attention_bf16"
    non_flash_compute_precision = "fp32"
    flash_attention_kernel_io_precision = "bf16"
    reconstruction_compute_precision = "fp32"

    def topology_embedding_rows(
        self,
        batch: object,
        *,
        sample_seeds: tuple[int, ...] | None = None,
    ) -> TopologyEmbeddingRows:
        """Run the faithful Flash-varlen backend and return one row per mesh."""

        return _flash_varlen_autoencoder_forward(
            self.autoencoder,
            batch.vertices,
            batch.vertex_mask,
            batch.faces,
            batch.incidence_index,
            sample_seeds,
        )
