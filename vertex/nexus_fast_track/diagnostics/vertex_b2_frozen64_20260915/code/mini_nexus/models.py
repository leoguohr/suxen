"""VecSet condition encoding and a conditional flow Transformer with 3D RoPE."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 64):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: Tensor) -> Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / max(half - 1, 1)
        )
        angles = time[:, None] * frequencies[None, :]
        encoded = torch.cat([angles.cos(), angles.sin()], dim=-1)
        if encoded.shape[-1] < self.frequency_dim:
            encoded = torch.cat([encoded, torch.zeros_like(encoded[:, :1])], dim=-1)
        return self.mlp(encoded)


def farthest_point_sample(
    positions: Tensor, num_samples: int, mask: Tensor | None = None
) -> Tensor:
    """Deterministic batched FPS without requiring torch-cluster."""

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,N,3]")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    batch_size, point_count, _ = positions.shape
    if mask is None:
        mask = torch.ones(
            (batch_size, point_count), dtype=torch.bool, device=positions.device
        )
    if mask.shape != (batch_size, point_count):
        raise ValueError("condition mask must have shape [B,N]")
    if (mask.sum(dim=1) < num_samples).any():
        raise ValueError("each condition needs at least num_samples valid points")

    result = torch.empty(
        (batch_size, num_samples), dtype=torch.long, device=positions.device
    )
    for batch_index in range(batch_size):
        valid_indices = torch.where(mask[batch_index])[0]
        valid_positions = positions[batch_index, valid_indices]
        centroid = valid_positions.mean(dim=0, keepdim=True)
        chosen_local = (valid_positions - centroid).square().sum(dim=-1).argmax()
        minimum_distance = torch.full(
            (len(valid_positions),), torch.inf, device=positions.device, dtype=positions.dtype
        )
        for sample_index in range(num_samples):
            result[batch_index, sample_index] = valid_indices[chosen_local]
            chosen_position = valid_positions[chosen_local]
            distance = (valid_positions - chosen_position).square().sum(dim=-1)
            minimum_distance = torch.minimum(minimum_distance, distance)
            chosen_local = minimum_distance.argmax()
    return result


class PointFourierEmbedding(nn.Module):
    """VecSet-style Fourier coordinate embedding plus raw point attributes."""

    def __init__(self, input_dim: int, output_dim: int, fourier_dim: int = 48):
        super().__init__()
        if input_dim < 3:
            raise ValueError("point input must begin with xyz")
        if fourier_dim % 6:
            raise ValueError("fourier_dim must be divisible by 6")
        frequency_count = fourier_dim // 6
        frequencies = math.pi * torch.pow(2.0, torch.arange(frequency_count).float())
        self.register_buffer("frequencies", frequencies)
        self.projection = nn.Linear(fourier_dim + input_dim, output_dim)

    def forward(self, points_and_features: Tensor) -> Tensor:
        xyz = points_and_features[..., :3]
        angles = xyz.unsqueeze(-1) * self.frequencies
        fourier = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        return self.projection(torch.cat([fourier, points_and_features], dim=-1))


class VecSetConditionEncoder(nn.Module):
    """FPS anchors + query-to-point cross-attention VecSet encoder.

    This follows the public 3DShape2VecSet encoder structure. Normals remain
    point attributes when the condition has six channels.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 96,
        num_tokens: int = 16,
        num_heads: int = 4,
        fourier_dim: int = 48,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.point_embedding = PointFourierEmbedding(input_dim, hidden_dim, fourier_dim)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(
        self, points_and_normals: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        if points_and_normals.ndim != 3:
            raise ValueError("condition must have shape [B,P,C]")
        if mask is not None and mask.shape != points_and_normals.shape[:2]:
            raise ValueError("condition mask must have shape [B,P]")
        indices = farthest_point_sample(
            points_and_normals[..., :3], self.num_tokens, mask
        )
        point_features = self.point_embedding(points_and_normals)
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, point_features.shape[-1])
        queries = torch.gather(point_features, 1, gather_indices)
        normalized_context = self.context_norm(point_features)
        update, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_context,
            normalized_context,
            key_padding_mask=None if mask is None else ~mask,
            need_weights=False,
        )
        vectors = queries + update
        return vectors + self.ffn(self.ffn_norm(vectors))


# Backward-compatible name used by the first teaching scripts.
PointConditionEncoder = VecSetConditionEncoder


def _rotate_pairs(values: Tensor, angles: Tensor) -> Tensor:
    even = values[..., 0::2]
    odd = values[..., 1::2]
    cosine = angles.cos().unsqueeze(1)
    sine = angles.sin().unsqueeze(1)
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)


def apply_3d_rope(query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Apply independent rotary frequencies to x, y and z channel groups."""

    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must share shape [B,H,N,D]")
    if positions.shape != (query.shape[0], query.shape[2], 3):
        raise ValueError("3D RoPE positions must have shape [B,N,3]")
    head_dim = query.shape[-1]
    axis_dim = (head_dim // 6) * 2
    if axis_dim == 0:
        return query, key
    inverse_frequency = torch.pow(
        10_000.0,
        -torch.arange(0, axis_dim, 2, device=query.device, dtype=query.dtype)
        / axis_dim,
    )
    query_parts: list[Tensor] = []
    key_parts: list[Tensor] = []
    for axis in range(3):
        start = axis * axis_dim
        stop = start + axis_dim
        angles = positions[..., axis].to(query.dtype).unsqueeze(-1) * inverse_frequency
        query_parts.append(_rotate_pairs(query[..., start:stop], angles))
        key_parts.append(_rotate_pairs(key[..., start:stop], angles))
    rotary_dim = 3 * axis_dim
    query_parts.append(query[..., rotary_dim:])
    key_parts.append(key[..., rotary_dim:])
    return torch.cat(query_parts, dim=-1), torch.cat(key_parts, dim=-1)


class RotarySelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: Tensor, positions: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        batch_size, token_count, hidden_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_3d_rope(query, key, positions)
        attention_mask = None
        if padding_mask is not None:
            attention_mask = ~padding_mask[:, None, None, :]
        result = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        ).transpose(1, 2).reshape(
            batch_size, token_count, hidden_dim
        )
        return self.output(result)


class ConditionalBlock(nn.Module):
    """3D-RoPE self-attention, condition cross-attention and FFN."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = RotarySelfAttention(hidden_dim, num_heads)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(
        self,
        x: Tensor,
        time_embedding: Tensor,
        condition_tokens: Tensor,
        positions: Tensor,
        padding_mask: Tensor | None,
    ) -> Tensor:
        time_bias = self.time_modulation(time_embedding).unsqueeze(1)
        hidden = self.self_norm(x + time_bias)
        x = x + self.self_attention(hidden, positions, padding_mask)
        hidden = self.cross_norm(x + time_bias)
        update, _ = self.cross_attention(
            hidden, condition_tokens, condition_tokens, need_weights=False
        )
        x = x + update
        return x + self.ffn(self.ffn_norm(x + time_bias))


class ConditionalFlowTransformer(nn.Module):
    """DiT velocity model with 3D RoPE and optional learnable octree depth."""

    def __init__(
        self,
        data_dim: int,
        metadata_dim: int,
        hidden_dim: int = 96,
        condition_dim: int = 96,
        num_layers: int = 4,
        num_heads: int = 4,
        max_depth: int = 9,
        use_depth_embedding: bool = True,
    ):
        super().__init__()
        self.metadata_dim = metadata_dim
        self.max_depth = max_depth
        self.data_projection = nn.Linear(data_dim, hidden_dim)
        self.metadata_projection = nn.Linear(metadata_dim, hidden_dim)
        self.condition_projection = (
            nn.Identity()
            if condition_dim == hidden_dim
            else nn.Linear(condition_dim, hidden_dim)
        )
        self.time_embedding = TimestepEmbedding(hidden_dim)
        self.depth_embedding = (
            nn.Embedding(max_depth + 1, hidden_dim) if use_depth_embedding else None
        )
        self.blocks = nn.ModuleList(
            [ConditionalBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, data_dim)

    def forward(
        self,
        noisy_data: Tensor,
        time: Tensor,
        metadata: Tensor,
        condition_tokens: Tensor,
        mask: Tensor | None = None,
        *,
        positions: Tensor | None = None,
        depths: Tensor | None = None,
    ) -> Tensor:
        if noisy_data.shape[:-1] != metadata.shape[:-1]:
            raise ValueError("data and metadata token dimensions must match")
        if metadata.shape[-1] != self.metadata_dim:
            raise ValueError("metadata feature dimension does not match model")
        padding_mask = None if mask is None else ~mask
        if positions is None:
            if metadata.shape[-1] < 3:
                raise ValueError("3D RoPE needs positions or three metadata coordinates")
            positions = metadata[..., :3]
        if positions.shape != (*noisy_data.shape[:2], 3):
            raise ValueError("positions must have shape [B,N,3]")

        x = self.data_projection(noisy_data) + self.metadata_projection(metadata)
        if self.depth_embedding is not None:
            if depths is None:
                if self.metadata_dim >= 4:
                    depths = (metadata[..., -1] * self.max_depth).round().long()
                else:
                    depths = torch.zeros(
                        noisy_data.shape[:2], device=noisy_data.device, dtype=torch.long
                    )
            if depths.shape != noisy_data.shape[:2]:
                raise ValueError("depths must have shape [B,N]")
            x = x + self.depth_embedding(depths.clamp(0, self.max_depth))

        condition_tokens = self.condition_projection(condition_tokens)
        time_embedding = self.time_embedding(time)
        for block in self.blocks:
            x = block(x, time_embedding, condition_tokens, positions, padding_mask)
        prediction = self.output(self.output_norm(x))
        if mask is not None:
            prediction = prediction * mask.unsqueeze(-1).to(prediction.dtype)
        return prediction
