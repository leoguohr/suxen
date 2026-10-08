"""Proposed own-512 topology AE v2; NOT a trained or production-validated model.

Borrowed computation patterns: Fourier XYZ, post-message LN/GELU graph residual,
attention+FFN at each layer. Original large-model dimensions, separate inputs,
encoder output LN, separate 32D heads and center-only output are retained.
No teacher weights, per-UID parameters, decoder XYZ skip or external files.
One complete mesh per forward; accumulate meshes in the training loop.
The caller must configure FP32 SDPA MATH and deterministic graph reduction.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

@dataclass(frozen=True)
class Config:
    encoder_width: int = 512
    latent_width: int = 512
    decoder_width: int = 1024
    encoder_composite_blocks: int = 12
    decoder_blocks: int = 16
    heads: int = 8
    ffn_ratio: int = 4
    fourier_bands: int = 6
    embedding_width: int = 32
    dropout: float = 0.0
    layer_norm_eps: float = 1e-5

class FourierXYZ(nn.Module):
    def __init__(self, bands: int):
        super().__init__()
        self.register_buffer('frequencies', math.pi * 2.0 ** torch.arange(bands, dtype=torch.float32))
    def forward(self, xyz: Tensor) -> Tensor:
        phase = xyz.unsqueeze(-1) * self.frequencies
        return torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)

class AttentionFFN(nn.Module):
    def __init__(self, width: int, heads: int, ratio: int, eps: float, dropout: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(width, eps=eps)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(width, ratio*width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ratio*width, width))
        self.residual_dropout = nn.Dropout(dropout)
    def forward(self, h: Tensor) -> Tensor:
        x = self.attn_norm(h).unsqueeze(0)
        a = self.attn(x, x, x, need_weights=False)[0].squeeze(0)
        h = h + self.residual_dropout(a)
        return h + self.residual_dropout(self.ffn(self.ffn_norm(h)))

class GraphResidual(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.neighbor_projection = nn.Linear(width, width, bias=False)
        self.message_norm = nn.LayerNorm(width, eps=eps)
    def forward(self, h: Tensor, source: Tensor, target: Tensor, degree: Tensor) -> Tensor:
        # Raw h is aggregated. No LN is applied to neighbors before aggregation.
        sums = torch.zeros_like(h).index_add(0, target, h[source])
        message = self.self_projection(h) + self.neighbor_projection(sums/degree)
        return h + F.gelu(self.message_norm(message))

class EncoderBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.graph = GraphResidual(cfg.encoder_width, cfg.layer_norm_eps)
        self.transformer = AttentionFFN(cfg.encoder_width, cfg.heads, cfg.ffn_ratio,
                                        cfg.layer_norm_eps, cfg.dropout)
    def forward(self, h: Tensor, source: Tensor, target: Tensor, degree: Tensor) -> Tensor:
        return self.transformer(self.graph(h, source, target, degree))

class OwnTopologyAEV2(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        for width in (cfg.encoder_width, cfg.decoder_width):
            if width <= 0 or width % cfg.heads: raise ValueError('Widths must be positive multiples of heads')
        if cfg.embedding_width % 2 or cfg.embedding_width <= 0: raise ValueError('Embedding must split evenly')
        if cfg.encoder_composite_blocks < 1 or cfg.decoder_blocks < 1: raise ValueError('Positive layer counts required')
        self.cfg = cfg
        self.position_features = FourierXYZ(cfg.fourier_bands)
        input_width = 3 + 6*cfg.fourier_bands
        # Separate projections retain the user's original vertex/face interface.
        self.vertex_input = nn.Linear(input_width, cfg.encoder_width)
        self.face_input = nn.Linear(input_width, cfg.encoder_width)
        self.encoder_blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_composite_blocks))
        self.encoder_output_norm = nn.LayerNorm(cfg.encoder_width, eps=cfg.layer_norm_eps)
        self.mu = nn.Linear(cfg.encoder_width, cfg.latent_width)
        self.log_variance = nn.Linear(cfg.encoder_width, cfg.latent_width)
        self.log_variance.requires_grad_(False)  # Deterministic reconstruction stage.
        self.latent_input = nn.Linear(cfg.latent_width, cfg.decoder_width)
        self.decoder_blocks = nn.ModuleList(AttentionFFN(cfg.decoder_width, cfg.heads, cfg.ffn_ratio,
                            cfg.layer_norm_eps, cfg.dropout) for _ in range(cfg.decoder_blocks))
        self.decoder_output_norm = nn.LayerNorm(cfg.decoder_width, eps=cfg.layer_norm_eps)
        self.edge_embedding = nn.Linear(cfg.decoder_width, cfg.embedding_width)
        self.face_embedding = nn.Linear(cfg.decoder_width, cfg.embedding_width)
    def encode(self, vertices: Tensor, faces: Tensor) -> tuple[Tensor, Tensor]:
        if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
            raise ValueError('vertices must be nonempty [V,3]')
        if vertices.dtype != torch.float32: raise TypeError('FP32 vertices required')
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype != torch.long:
            raise TypeError('faces must be int64 [F,3]')
        if faces.device != vertices.device: raise ValueError('Device mismatch')
        if not bool(torch.isfinite(vertices).all()): raise ValueError('Non-finite vertices')
        if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= len(vertices)):
            raise ValueError('Face index outside local mesh')
        nv, nf = len(vertices), len(faces)
        centers = vertices[faces].mean(1)
        h = torch.cat((self.vertex_input(self.position_features(vertices)),
                       self.face_input(self.position_features(centers))), dim=0)
        v_ids = faces.flatten()
        f_ids = (torch.arange(nf, device=faces.device)+nv).repeat_interleave(3)
        source = torch.cat((v_ids, f_ids)); target = torch.cat((f_ids, v_ids))
        degree = torch.bincount(target, minlength=nv+nf).clamp_min(1).to(h.dtype).unsqueeze(-1)
        for block in self.encoder_blocks: h = block(h, source, target, degree)
        hv = self.encoder_output_norm(h[:nv])
        return self.mu(hv), self.log_variance(hv).clamp(-20, 10)
    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.latent_input(latent)
        for block in self.decoder_blocks: h = block(h)
        h = self.decoder_output_norm(h)
        e, f = self.edge_embedding(h), self.face_embedding(h)
        return e-e.mean(0, keepdim=True), f-f.mean(0, keepdim=True), h
    def forward(self, vertices: Tensor, faces: Tensor, *, sample_latent: bool = False,
                generator: torch.Generator | None = None) -> dict[str, Tensor]:
        mu, logvar = self.encode(vertices, faces)
        z = mu
        if sample_latent:
            noise = torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
            z = mu + (0.5*logvar).exp()*noise
        edge, face, hidden = self.decode(z)
        return dict(mu=mu, log_variance=logvar, latent=z, edge=edge, face=face, decoder_hidden=hidden)
