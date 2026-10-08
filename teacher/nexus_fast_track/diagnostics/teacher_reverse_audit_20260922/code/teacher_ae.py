"""Inferred teacher AE, not original source.

Recovered state layout + one explicitly tested forward hypothesis.  Inference
on the 50 cached teacher samples matches 49 saved face sets and differs by one
face on sample 00 in the recorded CPU run.  Do NOT present strict loadability
or this near-parity as exact training-code recovery.

Assumed forward semantics: 6-band coordinate-major Fourier features with pi;
mean vertex-face aggregation; residual GELU(LayerNorm(self+neighbor));
4-head pre-norm GELU transformers; eval dropout=0; posterior mean for replay.
Training-time dropout, sampling, clamp and weighting must be specified by
training code, not inferred from these tensors alone.
"""
from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def fourier_positions(x: Tensor) -> Tensor:
    a = x[..., None] * (2.0 ** torch.arange(6, device=x.device, dtype=x.dtype)) * torch.pi
    return torch.cat((x, a.sin().flatten(-2), a.cos().flatten(-2)), dim=-1)


class GraphBlock(nn.Module):
    def __init__(self, width: int = 128):
        super().__init__()
        self.self_proj = nn.Linear(width, width)
        self.neighbor = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, h: Tensor, source: Tensor, dest: Tensor, degree: Tensor) -> Tensor:
        neighbor = torch.zeros_like(h).index_add_(0, dest, h[source]) / degree
        return h + F.gelu(self.norm(self.self_proj(h) + self.neighbor(neighbor)))


class TransformerBlock(nn.Module):
    """State-compatible TransformerEncoderLayer-style pre-norm block.

    Explicit eval-equivalent attention avoids relying on a framework fast-path.
    This module intentionally has no dropout policy: training code must decide
    and document it; original training dropout is not proven from state_dict.
    """
    def __init__(self, width: int = 128, heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.linear1 = nn.Linear(width, 4 * width)
        self.linear2 = nn.Linear(4 * width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.heads = heads

    def forward(self, h: Tensor) -> Tensor:
        x = self.norm1(h)
        n, d = x.shape
        q, k, v = F.linear(x, self.self_attn.in_proj_weight,
                          self.self_attn.in_proj_bias).chunk(3, dim=-1)
        q, k, v = [a.reshape(n, self.heads, d // self.heads).transpose(0, 1)[None]
                   for a in (q, k, v)]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)[0]
        x = x.transpose(0, 1).reshape(n, d)
        h = h + self.self_attn.out_proj(x)
        return h + self.linear2(F.gelu(self.linear1(self.norm2(h))))


class IndicatorState(nn.Module):
    def __init__(self):
        super().__init__()
        # A state_dict alone does not reveal parameter-vs-buffer status.
        # These constant values are retained for strict loading, NOT used as
        # additive thresholds in this tested spacetime sign-rule hypothesis.
        self.register_buffer('edge_threshold', torch.tensor(1.0))
        self.register_buffer('face_threshold', torch.tensor(1.0))


class ReconstructedTeacherAE(nn.Module):
    def __init__(self, layers: int = 8, heads: int = 4):
        super().__init__()
        self.input = nn.Linear(39, 128)
        self.graph = nn.ModuleList([GraphBlock() for _ in range(layers)])
        self.attn = nn.ModuleList([TransformerBlock(heads=heads) for _ in range(layers)])
        self.moments = nn.Linear(128, 128)
        self.decode_input = nn.Linear(64, 128)
        self.decode_blocks = nn.ModuleList([TransformerBlock(heads=heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64))
        self.indicator = IndicatorState()

    def encode(self, vertices: Tensor, faces: Tensor) -> tuple[Tensor, Tensor]:
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError('vertices must have shape [V,3]')
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
            raise ValueError('faces must be int64 [F,3]')
        n = len(vertices)
        x = torch.cat((vertices, vertices[faces].mean(dim=1)), dim=0)
        vi = faces.flatten()
        fi = torch.arange(len(faces), device=faces.device).repeat_interleave(3) + n
        source, dest = torch.cat((vi, fi)), torch.cat((fi, vi))
        degree = torch.bincount(dest, minlength=len(x)).clamp_min(1)[:, None]
        h = self.input(fourier_positions(x))
        for graph, attention in zip(self.graph, self.attn):
            h = attention(graph(h, source, dest, degree))
        return self.moments(h[:n]).chunk(2, dim=-1)

    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        h = self.decode_input(latent)
        for block in self.decode_blocks:
            h = block(h)
        return self.output(h).chunk(2, dim=-1)

    def forward(self, vertices: Tensor, faces: Tensor) -> tuple[Tensor, Tensor]:
        mu, _ = self.encode(vertices, faces)
        return self.decode(mu)


def load_teacher_ae(path: str | Path) -> ReconstructedTeacherAE:
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('method') != 'spacetime':
        raise ValueError('This reconstruction only covers the spacetime AE')
    sd = payload['state_dict']
    layers = int(payload.get('layers', payload.get('config', {}).get('ae_layers', 2)))
    model = ReconstructedTeacherAE(layers=layers)
    model.load_state_dict(sd, strict=True)
    return model.eval()


def edge_logits(z: Tensor, pairs: Tensor) -> Tensor:
    d = z[pairs[:, 0]] - z[pairs[:, 1]]
    return d[:, :16].square().sum(-1) - d[:, 16:].square().sum(-1)


def face_logits(z: Tensor, triangles: Tensor) -> Tensor:
    a = z[triangles[:, 0]] - z[triangles[:, 2]]
    b = z[triangles[:, 1]] - z[triangles[:, 2]]
    def gram(x: Tensor, y: Tensor) -> Tensor:
        return x.square().sum(-1) * y.square().sum(-1) - (x * y).sum(-1).square()
    return gram(a[:, :16], b[:, :16]) - gram(a[:, 16:], b[:, 16:])
