"""Point-conditioned vertex flow models for the Nexus reproduction.

VecSet's one aggregation block plus seven self-attention blocks, the precise
DiT size, and the absolute-coordinate path are reproduction choices. Masks
throughout this module use True for valid tokens.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


def _valid_mask(mask: Tensor | None, shape: tuple[int, int], device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B,N]")
    return mask


def _fourier_xyz(positions: Tensor) -> Tensor:
    frequencies = math.pi * 2.0 ** torch.arange(
        8, device=positions.device, dtype=torch.float32
    )
    xyz = positions.float()
    angles = xyz.unsqueeze(-1) * frequencies
    return torch.cat([xyz, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


@torch.no_grad()
def farthest_point_sample(
    positions: Tensor, num_samples: int, mask: Tensor | None = None
) -> Tensor:
    """Deterministic batched FPS with unique indices, even for duplicate XYZ."""
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [B,P,3]")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    valid = _valid_mask(mask, positions.shape[:2], positions.device)
    if (valid.sum(dim=1) < num_samples).any():
        raise ValueError("each point cloud needs at least num_samples valid points")
    xyz = positions.float().masked_fill(~valid.unsqueeze(-1), 0)
    centroid = xyz.sum(dim=1) / valid.sum(dim=1, keepdim=True)
    distances = (xyz - centroid.unsqueeze(1)).square().sum(dim=-1)
    chosen = distances.masked_fill(~valid, -torch.inf).argmax(dim=1)
    minimum = torch.full_like(distances, torch.inf).masked_fill(~valid, -torch.inf)
    indices = torch.empty(
        (positions.shape[0], num_samples), dtype=torch.long, device=positions.device
    )
    batch_indices = torch.arange(positions.shape[0], device=positions.device)
    for sample_index in range(num_samples):
        indices[:, sample_index] = chosen
        center = xyz[batch_indices, chosen].unsqueeze(1)
        minimum = torch.minimum(minimum, (xyz - center).square().sum(dim=-1))
        minimum.scatter_(1, chosen.unsqueeze(1), -torch.inf)
        chosen = minimum.argmax(dim=1)
    return indices


def parent_centers(parent_codes: Tensor, depths: Tensor) -> Tensor:
    """Return parent centers in [-1,1]; depths are target child depths, not parents.

    parent_codes is [B,N,3]. depths may be [B] or [B,N]. The caller validates
    depth ranges and clears padding before invoking this coordinate transform.
    """
    if depths.ndim == 1:
        depths = depths[:, None]
    resolution = 2.0 ** (depths.float() - 1)
    return -1.0 + 2.0 * (parent_codes.float() + 0.5) / resolution.unsqueeze(-1)


def apply_vertex_rope(
    query: Tensor, key: Tensor, positions: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply axial XYZ RoPE to Q/K [B,H,N,D], retaining leftover channels.

    positions is [B,N,3] in reference-grid units, not normalized coordinates.
    With head width 128, each axis rotates 21 pairs and two channels remain
    unchanged. Angles and rotations are evaluated in float32.
    """
    if query.ndim != 4 or query.shape != key.shape:
        raise ValueError("query and key must share shape [B,H,N,D]")
    if positions.shape != (query.shape[0], query.shape[2], 3):
        raise ValueError("RoPE positions must have shape [B,N,3]")
    pairs = query.shape[-1] // 6
    if pairs == 0:
        return query, key
    frequencies = 10_000.0 ** (
        -torch.arange(pairs, device=query.device, dtype=torch.float32) / pairs
    )
    angles = positions.float().unsqueeze(-1) * frequencies
    cosine = angles.cos().unsqueeze(1)
    sine = angles.sin().unsqueeze(1)
    rotary_dim = 6 * pairs

    def rotate(values: Tensor) -> Tensor:
        paired = values[..., :rotary_dim].float().reshape(
            *values.shape[:-1], 3, pairs, 2
        )
        even, odd = paired.unbind(dim=-1)
        rotated = torch.stack(
            [even * cosine - odd * sine, even * sine + odd * cosine], dim=-1
        ).flatten(-3).to(values.dtype)
        return torch.cat([rotated, values[..., rotary_dim:]], dim=-1)

    return rotate(query), rotate(key)


class _LayerNorm(nn.LayerNorm):
    def __init__(self, hidden_dim: int, affine: bool = True):
        super().__init__(hidden_dim, eps=1e-6, elementwise_affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.float(), self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(), self.eps,
        ).to(x.dtype)


class _HeadRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
        return (values * self.weight.float()[None, :, None, :]).to(x.dtype)


class _Attention(nn.Module):
    def __init__(
        self, hidden_dim: int, num_heads: int,
        context_dim: int | None = None, qk_norm: bool = False,
    ):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.context_dim = context_dim
        if context_dim is None:
            self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        else:
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key_value = nn.Linear(context_dim, 2 * hidden_dim)
        self.query_norm = _HeadRMSNorm(num_heads, self.head_dim) if qk_norm else nn.Identity()
        self.key_norm = _HeadRMSNorm(num_heads, self.head_dim) if qk_norm else nn.Identity()
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, x: Tensor, context: Tensor | None = None,
        key_mask: Tensor | None = None, positions: Tensor | None = None,
    ) -> Tensor:
        batch_size, token_count, hidden_dim = x.shape
        if self.context_dim is None:
            q, k, v = self.qkv(x).reshape(
                batch_size, token_count, 3, self.num_heads, self.head_dim
            ).unbind(dim=2)
        else:
            q = self.query(x).reshape(batch_size, token_count, self.num_heads, self.head_dim)
            k, v = self.key_value(context).reshape(
                batch_size, context.shape[1], 2, self.num_heads, self.head_dim
            ).unbind(dim=2)
        q = self.query_norm(q.transpose(1, 2))
        k = self.key_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)
        if positions is not None:
            q, k = apply_vertex_rope(q, k, positions)
        attention_mask = None if key_mask is None else key_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0
        )
        return self.output(attended.transpose(1, 2).reshape(batch_size, token_count, hidden_dim))


def _ffn(hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(approximate="tanh"),
        nn.Linear(4 * hidden_dim, hidden_dim),
    )


class _ConditionCrossBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.query_norm = _LayerNorm(hidden_dim)
        self.context_norm = _LayerNorm(hidden_dim)
        self.attention = _Attention(hidden_dim, num_heads, context_dim=hidden_dim)
        self.ffn_norm = _LayerNorm(hidden_dim)
        self.ffn = _ffn(hidden_dim)

    def forward(self, queries: Tensor, context: Tensor, mask: Tensor) -> Tensor:
        x = queries + self.attention(
            self.query_norm(queries), self.context_norm(context), key_mask=mask
        )
        return x + self.ffn(self.ffn_norm(x))


class _ConditionSelfBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention_norm = _LayerNorm(hidden_dim)
        self.attention = _Attention(hidden_dim, num_heads)
        self.ffn_norm = _LayerNorm(hidden_dim)
        self.ffn = _ffn(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class VertexConditionEncoder(nn.Module):
    """Eight blocks by default: FPS-query cross-attention, then seven self blocks."""
    def __init__(
        self, input_dim: int = 6, hidden_dim: int = 2048,
        num_tokens: int = 1024, num_heads: int = 16, num_layers: int = 8,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if input_dim < 3 or num_layers < 1 or num_tokens < 1:
            raise ValueError("input_dim >= 3, num_layers >= 1 and num_tokens >= 1 are required")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.point_embedding = nn.Linear(input_dim + 48, hidden_dim)
        self.blocks = nn.ModuleList([
            _ConditionCrossBlock(hidden_dim, num_heads),
            *[_ConditionSelfBlock(hidden_dim, num_heads) for _ in range(num_layers - 1)],
        ])
        self.output_norm = _LayerNorm(hidden_dim)

    def forward(
        self, points_and_normals: Tensor, mask: Tensor | None = None,
        *, fps_indices: Tensor | None = None,
    ) -> Tensor:
        # Precomputed FPS indices are valid only for the same point order and mask.
        if points_and_normals.ndim != 3 or points_and_normals.shape[-1] != self.input_dim:
            raise ValueError("point cloud must have shape [B,P,input_dim]")
        valid = _valid_mask(mask, points_and_normals.shape[:2], points_and_normals.device)
        points = points_and_normals.masked_fill(~valid.unsqueeze(-1), 0)
        indices = fps_indices
        if indices is None:
            indices = farthest_point_sample(points[..., :3], self.num_tokens, valid)
        elif indices.shape != (points.shape[0], self.num_tokens) or indices.dtype != torch.long:
            raise ValueError("FPS indices must be long with shape [B,num_tokens]")
        features = torch.cat([_fourier_xyz(points[..., :3]), points[..., 3:].float()], dim=-1)
        embedded = self.point_embedding(features.to(self.point_embedding.weight.dtype))
        queries = embedded.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            x = checkpoint(self.blocks[0], queries, embedded, valid, use_reentrant=False)
            for block in self.blocks[1:]:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.blocks[0](queries, embedded, valid)
            for block in self.blocks[1:]:
                x = block(x)
        return self.output_norm(x)


class _VertexBlock(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, num_heads: int):
        super().__init__()
        self.self_norm = _LayerNorm(hidden_dim, affine=False)
        self.cross_norm = _LayerNorm(hidden_dim)
        self.ffn_norm = _LayerNorm(hidden_dim, affine=False)
        self.self_attention = _Attention(hidden_dim, num_heads, qk_norm=True)
        self.cross_attention = _Attention(hidden_dim, num_heads, condition_dim, qk_norm=True)
        self.ffn = _ffn(hidden_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim))

    def forward(
        self, x: Tensor, time_embedding: Tensor, condition: Tensor,
        positions: Tensor, mask: Tensor, condition_mask: Tensor,
    ) -> Tensor:
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = (
            self.modulation(time_embedding).unsqueeze(1).chunk(6, dim=-1)
        )
        h = self.self_norm(x) * (1 + scale_sa) + shift_sa
        x = x + gate_sa * self.self_attention(h, key_mask=mask, positions=positions)
        x = x + self.cross_attention(self.cross_norm(x), condition, key_mask=condition_mask)
        h = self.ffn_norm(x) * (1 + scale_ff) + shift_ff
        x = x + gate_ff * self.ffn(h)
        return x.masked_fill(~mask.unsqueeze(-1), 0)


class VertexDiT(nn.Module):
    """Shared-across-depth vertex velocity model with independent DiT blocks.

    The 256-dimensional sinusoidal time encoding uses 1000*t, an explicit
    engineering choice for a flow time in [0,1]. Parent XYZ Fourier features
    retain absolute location; self-RoPE uses the same centers in D-bit grid units.
    Only self/FFN branches use AdaLN-Zero; ungated cross-attention means an
    entire block is not initially the identity. The final velocity head is zero.
    """
    def __init__(
        self, hidden_dim: int = 1536, condition_dim: int = 2048,
        num_layers: int = 36, num_heads: int = 12, max_depth: int = 9,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if num_layers < 1 or max_depth < 1 or condition_dim < 1:
            raise ValueError("num_layers, max_depth and condition_dim must be positive")
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.use_checkpoint = use_checkpoint
        self.data_embedding = nn.Linear(8, hidden_dim)
        self.position_embedding = nn.Linear(51, hidden_dim)
        self.depth_embedding = nn.Embedding(max_depth, hidden_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            _VertexBlock(hidden_dim, condition_dim, num_heads) for _ in range(num_layers)
        ])
        self.output_norm = _LayerNorm(hidden_dim, affine=False)
        self.output = nn.Linear(hidden_dim, 8)
        self.apply(self._initialize_linear)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _initialize_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, noisy: Tensor, time: Tensor, parent_codes: Tensor, depths: Tensor,
        condition_tokens: Tensor, mask: Tensor | None = None,
        condition_mask: Tensor | None = None,
    ) -> Tensor:
        if noisy.ndim != 3 or noisy.shape[-1] != 8:
            raise ValueError("noisy occupancy must have shape [B,N,8]")
        batch_size, token_count = noisy.shape[:2]
        if parent_codes.shape != (batch_size, token_count, 3) or parent_codes.dtype != torch.long:
            raise ValueError("parent_codes must be long with shape [B,N,3]")
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [B]")
        if depths.shape not in [(batch_size,), (batch_size, token_count)] or depths.dtype != torch.long:
            raise ValueError("depths must be long with shape [B] or [B,N]")
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch_size or condition_tokens.shape[-1] != self.condition_dim:
            raise ValueError("condition_tokens must have shape [B,M,condition_dim]")
        valid = _valid_mask(mask, noisy.shape[:2], noisy.device)
        condition_valid = _valid_mask(condition_mask, condition_tokens.shape[:2], condition_tokens.device)
        if not condition_valid.any(dim=1).all():
            raise ValueError("each object needs at least one valid condition token")
        if depths.ndim == 1:
            depths = depths[:, None].expand(-1, token_count)
        if (((depths < 1) | (depths > self.max_depth)) & valid).any():
            raise ValueError("valid target depths must lie in [1,max_depth]")
        safe_depths = depths.masked_fill(~valid, 1)
        safe_codes = parent_codes.masked_fill(~valid.unsqueeze(-1), 0)
        centers = parent_centers(safe_codes, safe_depths)
        rope_positions = centers * (2 ** (self.max_depth - 1))
        clean_noisy = noisy.masked_fill(~valid.unsqueeze(-1), 0)
        condition = condition_tokens.masked_fill(~condition_valid.unsqueeze(-1), 0)
        dtype = self.data_embedding.weight.dtype
        x = self.data_embedding(clean_noisy.to(dtype))
        x = x + self.position_embedding(_fourier_xyz(centers).to(dtype))
        x = x + self.depth_embedding(safe_depths - 1)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        frequencies = 10_000.0 ** (
            -torch.arange(128, device=time.device, dtype=torch.float32) / 128
        )
        angles = (1000 * time.float()).unsqueeze(-1) * frequencies
        time_features = torch.cat([angles.cos(), angles.sin()], dim=-1)
        time_embedding = self.time_embedding(time_features.to(dtype))
        condition = condition.to(dtype)
        for block in self.blocks:
            args = (x, time_embedding, condition, rope_positions, valid, condition_valid)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, *args, use_reentrant=False)
            else:
                x = block(*args)
        return self.output(self.output_norm(x)).masked_fill(~valid.unsqueeze(-1), 0)
