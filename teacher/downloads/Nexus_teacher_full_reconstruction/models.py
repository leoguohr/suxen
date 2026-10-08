"""Reconstructed teacher flow networks (NOT the original training source).

Weights establish tensor sizes. These global forward semantics were selected by
read-only hypothesis tests and validated by actual noise-to-mesh CPU sampling.
No per-sample rules, target points/faces or saved predictions are model inputs.
Raw-text encoder weights/tokenization are not included: inputs are text features.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from teacher_ae import ReconstructedTeacherAE, fourier_positions

FORWARD_CONTRACT = dict(heads=4, head_dim=36, rope_axis_dim=12,
    rope_layout='xyz axes; adjacent feature pairs', rope_angle_scale='pi',
    rope_frequencies='10000 ** (-arange(0,12,2)/12)',
    time_dim=64, time_scale=1000, time_order='cos then sin', time_denominator=32,
    adaln_order='shift1,scale1,gate1,shift2,scale2,gate2',
    norm_eps=1e-5, mlp_activation='SiLU except DiT feedforward GELU',
    training_dropout='not proven; these modules use 0')

def time_embedding(t: Tensor) -> Tensor:
    freq = torch.exp(-math.log(10000) * torch.arange(32, device=t.device, dtype=t.dtype) / 32)
    angle = t.reshape(-1, 1) * 1000 * freq
    return torch.cat((angle.cos(), angle.sin()), dim=-1)


def rope_3d(x: Tensor, xyz: Tensor) -> Tensor:
    """x[B,4,N,36]; positions[B,N,3]."""
    axis_dim = x.shape[-1] // 3
    if x.shape[-1] % 6:
        raise ValueError('Head width must be divisible by 6')
    value = x.reshape(*x.shape[:-1], 3, axis_dim)
    freq = torch.exp(-math.log(10000) * torch.arange(
        0, axis_dim, 2, device=x.device, dtype=x.dtype) / axis_dim)
    angle = xyz[:, None, :, :, None] * math.pi * freq
    c, s = angle.cos(), angle.sin()
    u, v = value[..., 0::2], value[..., 1::2]
    return torch.stack((u*c-v*s, u*s+v*c), -1).flatten(-2).reshape_as(x)


class DiTBlock(nn.Module):
    def __init__(self, width: int = 144):
        super().__init__()
        self.qkv = nn.Linear(width, 3*width)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width))
        self.heads = 4

    def forward(self, x: Tensor, condition: Tensor, xyz: Tensor,
                mask: Tensor | None = None) -> Tensor:
        b, n, d = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(condition).chunk(6, -1)
        y = F.layer_norm(x, (d,), eps=1e-5) * (1+scale1[:, None]) + shift1[:, None]
        q, k, v = self.qkv(y).reshape(b, n, 3, self.heads, d//self.heads).permute(2,0,3,1,4).unbind(0)
        q, k = rope_3d(q, xyz), rope_3d(k, xyz)
        attention = F.scaled_dot_product_attention(q, k, v,
            attn_mask=None if mask is None else mask[:, None, None, :], dropout_p=0.)
        x = x + gate1[:, None] * self.out(attention.transpose(1,2).reshape(b,n,d))
        y = F.layer_norm(x, (d,), eps=1e-5) * (1+scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.ff(y)


class TopologyFlow(nn.Module):
    """Vertex-conditioned velocity field: [B,N,64] -> [B,N,64]."""
    def __init__(self, layers: int = 10):
        super().__init__()
        self.input = nn.Linear(64,144)
        self.position = nn.Linear(39,144)
        self.time = nn.Sequential(nn.Linear(64,144),nn.SiLU(),nn.Linear(144,144))
        self.blocks = nn.ModuleList([DiTBlock() for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(144),nn.Linear(144,64))

    def forward(self, z: Tensor, t: Tensor, vertices: Tensor,
                mask: Tensor | None = None) -> Tensor:
        if z.shape[:2] != vertices.shape[:2] or z.shape[-1] != 64:
            raise ValueError('Expected matching z[B,N,64] and vertices[B,N,3]')
        x = self.input(z) + self.position(fourier_positions(vertices))
        condition = self.time(time_embedding(t))
        for block in self.blocks:
            x = block(x, condition, vertices, mask)
        return self.out(x)


class PointFlow(nn.Module):
    """Final learned text-coordinate prior + scaled frozen-denoiser candidate.

    state_dict does not encode requires_grad. Freeze rules belong to the trainer.
    residual_scale is declared a Parameter because candidate.pt has a scalar
    Adam slot for it. The text-coordinate prior has six matching Adam slots.
    """
    def __init__(self, layers: int = 18, prior_width: int = 512):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(1.))
        self.text = nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,144),nn.SiLU(),nn.Linear(144,144))
        self.count = nn.Sequential(nn.SiLU(),nn.Linear(144,275))
        self.slot = nn.Embedding(274,144)
        self.input = nn.Linear(39,144)
        self.time = nn.Sequential(nn.Linear(64,144),nn.SiLU(),nn.Linear(144,144))
        self.blocks = nn.ModuleList([DiTBlock() for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(144),nn.Linear(144,3))
        self.coordinate_prior = nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,prior_width),
                                             nn.SiLU(),nn.Linear(prior_width,822))

    def count_logits(self, text_features: Tensor) -> Tensor:
        return self.count(self.text(text_features))

    def prior(self, text_features: Tensor) -> Tensor:
        return self.coordinate_prior(text_features).reshape(-1,274,3)

    def denoise(self, x: Tensor, t: Tensor, text_features: Tensor,
                mask: Tensor | None = None) -> Tensor:
        if x.shape[-1] != 3 or not 0 < x.shape[1] <= 274:
            raise ValueError('Expected x[B,N,3], 1<=N<=274')
        text = self.text(text_features)
        h = self.input(fourier_positions(x)) + self.slot.weight[:x.shape[1]][None]
        h = h + text[:, None]
        condition = self.time(time_embedding(t)) + text
        for block in self.blocks:
            h = block(h, condition, x, mask)
        return self.out(h)

    def forward(self, x: Tensor, t: Tensor, text_features: Tensor,
                mask: Tensor | None = None) -> Tensor:
        return self.prior(text_features)[:, :x.shape[1]] + self.residual_scale*self.denoise(x,t,text_features,mask)


def load_ae(path: str | Path) -> tuple[ReconstructedTeacherAE, str]:
    p = torch.load(path, map_location='cpu', weights_only=True)
    cfg = p.get('config', {})
    model = ReconstructedTeacherAE(layers=int(p.get('layers',cfg.get('ae_layers',2))))
    model.load_state_dict(p['state_dict'], strict=True)
    method = p['method']
    if method not in ('cosine','euclidean','spacetime'):
        raise ValueError(f'Unsupported method {method}')
    return model.eval(), method


def load_topology(path: str | Path) -> TopologyFlow:
    p = torch.load(path, map_location='cpu', weights_only=True)
    model = TopologyFlow(int(p.get('layers',p.get('config',{}).get('flow_layers',4))))
    model.load_state_dict(p['state_dict'], strict=True)
    return model.eval()


def load_points(path: str | Path) -> PointFlow:
    p = torch.load(path, map_location='cpu', weights_only=True)
    cfg = p['config']  # final checkpoint config, NOT stale six-layer JSON
    model = PointFlow(int(cfg['layers']),int(cfg.get('prior_width',512)))
    model.load_state_dict(p['state_dict'], strict=True)
    return model.eval()


@torch.inference_mode()
def sample_points(model: PointFlow, text_feature: Tensor, generator: torch.Generator,
                  steps: int = 100) -> Tensor:
    """One condition at a time. No GT count, vertices, faces or sample ID."""
    if text_feature.shape != (1,2048):
        raise ValueError('text_feature must be [1,2048]')
    n = int(model.count_logits(text_feature).argmax(-1).item())
    if not 1 <= n <= 274:
        raise ValueError(f'Invalid model-predicted count: {n}')
    x = torch.randn((1,n,3),generator=generator,device=text_feature.device,dtype=text_feature.dtype)
    for i in range(steps):
        t = x.new_tensor([i/steps])
        x1 = model(x,t,text_feature)
        x = x + (1/steps) * (x1-x)/(1-t[:,None,None])
    return x[0]


@torch.inference_mode()
def sample_topology(model: TopologyFlow, vertices: Tensor, mean: Tensor, std: Tensor,
                    generator: torch.Generator, steps: int = 50) -> Tensor:
    """Returns unnormalized AE latent. No GT topology/AE encoder at generation."""
    z = torch.randn((1,len(vertices),64),generator=generator,device=vertices.device,dtype=vertices.dtype)
    for i in range(steps):
        z = z + (1/steps) * model(z,z.new_tensor([i/steps]),vertices[None])
    return z[0]*std+mean
