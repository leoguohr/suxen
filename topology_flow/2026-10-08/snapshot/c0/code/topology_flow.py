"""Topology velocity network. Importing this module has no GPU side effects.

NEXUS specifies per-vertex latent flow, XYZ RoPE and condition cross-attention.
The concrete blocks and R1 initialization below are local reconstruction choices.
All sizes and RoPE units must be chosen explicitly by the caller.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import math
import os
import torch
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
from _vertex_reference import _VertexBlock, _LayerNorm, _valid_mask, _fourier_xyz, VertexConditionEncoder, farthest_point_sample


def configure_math_backend():
    """CLI setup before any CUDA allocation; does not initialize a device."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@contextmanager
def math_context(device_type):
    with torch.autocast(device_type=device_type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        yield


@dataclass(frozen=True)
class TopologyFlowConfig:
    hidden_dim: int
    num_layers: int
    num_heads: int
    condition_dim: int
    condition_layers: int
    condition_heads: int
    condition_tokens: int
    rope_scale: float
    latent_dim: int = 512
    recompute: bool = True

    def __post_init__(self):
        sizes = (self.hidden_dim, self.num_layers, self.num_heads, self.condition_dim,
                 self.condition_layers, self.condition_heads, self.condition_tokens)
        if any(x < 1 for x in sizes) or self.hidden_dim % self.num_heads or self.condition_dim % self.condition_heads:
            raise ValueError('Positive sizes and head-divisible widths are required')
        if self.hidden_dim // self.num_heads < 6:
            raise ValueError('XYZ RoPE requires at least six channels per head')
        if self.rope_scale is None or not math.isfinite(self.rope_scale) or self.rope_scale <= 0:
            raise ValueError('An explicit finite positive RoPE scale is required')


class TopologyDiT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.latent_dim != 512 or cfg.num_layers < 1 or cfg.rope_scale <= 0:
            raise ValueError('This interface requires latent512, positive depth and explicit positive RoPE scale')
        self.cfg = cfg
        self.data_embedding = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.time_embedding = nn.Sequential(nn.Linear(256, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.blocks = nn.ModuleList(_VertexBlock(cfg.hidden_dim, cfg.condition_dim, cfg.num_heads) for _ in range(cfg.num_layers))
        self.output_norm = _LayerNorm(cfg.hidden_dim, affine=False)
        self.output = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for layer in (self.time_embedding[0], self.time_embedding[2]):
            nn.init.normal_(layer.weight, std=.02)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
            nn.init.zeros_(block.cross_attention.output.weight)
            nn.init.zeros_(block.cross_attention.output.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, noisy, time, vertices, condition_tokens, vertex_mask=None, condition_mask=None):
        if noisy.ndim != 3 or noisy.shape[-1] != 512 or noisy.dtype != torch.float32:
            raise ValueError('noisy must be FP32 [B,N,512]')
        batch, count = noisy.shape[:2]
        if vertices.shape != (batch, count, 3) or vertices.dtype != torch.float32:
            raise ValueError('vertices must be FP32 [B,N,3], in the same local vertex order')
        if time.shape != (batch,) or ((time < 0) | (time > 1)).any():
            raise ValueError('time must have shape [B] and lie in [0,1]')
        if condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch or condition_tokens.shape[-1] != self.cfg.condition_dim:
            raise ValueError('condition_tokens must have shape [B,M,condition_dim]')
        mask = _valid_mask(vertex_mask, noisy.shape[:2], noisy.device)
        cmask = _valid_mask(condition_mask, condition_tokens.shape[:2], noisy.device)
        if not mask.any(1).all() or not cmask.any(1).all():
            raise ValueError('Each mesh needs at least one vertex and one condition token')
        positions = vertices.masked_fill(~mask.unsqueeze(-1), 0)*self.cfg.rope_scale
        context = condition_tokens.masked_fill(~cmask.unsqueeze(-1), 0).float()
        with math_context(noisy.device.type):
            x = self.data_embedding(noisy.masked_fill(~mask.unsqueeze(-1), 0))
            frequencies = 10_000.0**(-torch.arange(128, device=noisy.device, dtype=torch.float32)/128)
            angles = (1000*time.float()).unsqueeze(-1)*frequencies
            temb = self.time_embedding(torch.cat((angles.cos(), angles.sin()), -1))
            for block in self.blocks:
                args = (x, temb, context, positions, mask, cmask)
                if self.cfg.recompute and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, *args, use_reentrant=False,
                        context_fn=lambda: (math_context(noisy.device.type), math_context(noisy.device.type)))
                else:
                    x = block(*args)
            return self.output(self.output_norm(x)).masked_fill(~mask.unsqueeze(-1), 0)


class PointCloudTopologyFlow(nn.Module):
    """Own VecSet instance; no shared weights with any live Vertex process."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Wrap each condition block below so recompute uses the same MATH context.
        self.condition_encoder = VertexConditionEncoder(input_dim=6, hidden_dim=cfg.condition_dim,
            num_tokens=cfg.condition_tokens, num_heads=cfg.condition_heads, num_layers=cfg.condition_layers,
            use_checkpoint=False)
        self.flow = TopologyDiT(cfg)

    def encode_condition(self, points, point_mask=None, fps_indices=None):
        if points.dtype != torch.float32 or points.ndim != 3 or points.shape[-1] != 6:
            raise ValueError('point cloud condition must be FP32 [B,P,6]: XYZ plus normals')
        valid = _valid_mask(point_mask, points.shape[:2], points.device)
        if (valid.sum(1) < self.cfg.condition_tokens).any():
            raise ValueError('Not enough valid condition points for the requested tokens')
        if fps_indices is None:
            fps_indices = farthest_point_sample(points[..., :3], self.cfg.condition_tokens, valid)
        if fps_indices.shape != (len(points), self.cfg.condition_tokens) or fps_indices.dtype != torch.long:
            raise ValueError('Invalid FPS index shape or dtype')
        if (fps_indices < 0).any() or (fps_indices >= points.shape[1]).any() or not valid.gather(1, fps_indices).all():
            raise ValueError('FPS references an invalid condition point')
        if (fps_indices.sort(1).values.diff(dim=1) == 0).any():
            raise ValueError('FPS indices must be unique within each mesh')
        encoder = self.condition_encoder
        with math_context(points.device.type):
            # Identical operations to the immutable reference, with explicit contexts.
            points = points.masked_fill(~valid.unsqueeze(-1), 0)
            embedded = encoder.point_embedding(torch.cat((_fourier_xyz(points[..., :3]), points[..., 3:]), -1))
            queries = embedded.gather(1, fps_indices[..., None].expand(-1, -1, embedded.shape[-1]))
            x = queries
            for i, block in enumerate(encoder.blocks):
                args = (x, embedded, valid) if i == 0 else (x,)
                if self.cfg.recompute and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, *args, use_reentrant=False,
                        context_fn=lambda: (math_context(points.device.type), math_context(points.device.type)))
                else:
                    x = block(*args)
            return encoder.output_norm(x)

    def forward(self, noisy, time, vertices, points, vertex_mask=None, point_mask=None, fps_indices=None):
        context = self.encode_condition(points, point_mask, fps_indices)
        return self.flow(noisy, time, vertices, context, vertex_mask)


def linear_flow_target(clean, noise, time, mask):
    """Noise at t=0, posterior target at t=1; velocity=clean-noise."""
    if clean.shape != noise.shape or time.shape != (clean.shape[0],) or mask.shape != clean.shape[:2]:
        raise ValueError('Incompatible clean/noise/time/mask shapes')
    clean = clean.masked_fill(~mask.unsqueeze(-1), 0)
    noise = noise.masked_fill(~mask.unsqueeze(-1), 0)
    t = time[:, None, None]
    return (1-t)*noise+t*clean, clean-noise


def equal_mesh_velocity_loss(prediction, target, mask):
    """Average valid vertex/channel MSE within each mesh, then average meshes."""
    if prediction.shape != target.shape or mask.shape != prediction.shape[:2] or not mask.any(1).all():
        raise ValueError('Expected equal [B,N,C] tensors and a nonempty [B,N] mask')
    error = (prediction.float()-target.float()).masked_fill(~mask.unsqueeze(-1), 0).square()
    per_mesh = error.sum((1, 2))/(mask.sum(1)*prediction.shape[-1])
    return per_mesh.mean()
