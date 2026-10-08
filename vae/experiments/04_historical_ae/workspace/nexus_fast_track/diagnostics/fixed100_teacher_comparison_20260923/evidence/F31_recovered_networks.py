"""2026-09-23 teacher-network reconstruction from original ZIP evidence.

This is newly written recovery code, not recovered original source. Tensor
names/sizes and layer counts come from ORIGINAL_CHECKPOINT_METADATA.json.
We have seen yesterday's reconstruction: non-identifiable forward conventions
are deliberately reused from its tested global hypothesis, not independently
rediscovered. See CONFIG_PROVENANCE.json for the evidence of each convention.
Only inference behavior is specified. No original dropout/trainer is claimed.
"""
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def position_features(points):
    # 39 = xyz + 6 sine and 6 cosine frequencies for each coordinate.
    bands = torch.arange(6, dtype=points.dtype, device=points.device)
    phase = points.unsqueeze(-1) * (2 ** bands) * math.pi
    return torch.cat([points, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)


def time_features(time):
    frequencies = torch.exp(torch.arange(32, dtype=time.dtype, device=time.device)
                            * (-math.log(10000) / 32))
    phase = time.reshape(-1, 1) * 1000 * frequencies
    return torch.cat([phase.cos(), phase.sin()], -1)


def rotary_xyz(tensor, points):
    # Width144 / 4 heads / xyz = 12 coordinates per spatial axis.
    shape = tensor.shape
    axes = tensor.reshape(*shape[:-1], 3, 6, 2)
    frequency = torch.exp(torch.arange(6, dtype=tensor.dtype, device=tensor.device)
                          * (-math.log(10000) / 6))
    angle = points[:, None, :, :, None] * math.pi * frequency
    real, imag = axes.unbind(-1)
    rotated = torch.stack([real * angle.cos() - imag * angle.sin(),
                           real * angle.sin() + imag * angle.cos()], -1)
    return rotated.reshape(shape)


class VertexFaceMessage(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_proj = nn.Linear(128, 128)
        self.neighbor = nn.Linear(128, 128, bias=False)
        self.norm = nn.LayerNorm(128)

    def forward(self, hidden, source, target, degree):
        aggregated = torch.zeros_like(hidden)
        aggregated.index_add_(0, target, hidden[source])
        message = self.self_proj(hidden) + self.neighbor(aggregated / degree)
        return hidden + F.gelu(self.norm(message))


class StoredIndicators(nn.Module):
    def __init__(self):
        super().__init__()
        # Original serialization cannot identify Parameter versus buffer.
        # Keep the two scalar values for strict replay; spacetime ignores them.
        self.register_buffer('edge_threshold', torch.ones(()))
        self.register_buffer('face_threshold', torch.ones(()))


def encoder_layer():
    # Original state key layout is exactly TransformerEncoderLayer-compatible.
    # Heads, pre-norm, GELU and eval dropout are tested forward conventions.
    return nn.TransformerEncoderLayer(128, 4, dim_feedforward=512,
                                     dropout=0, activation='gelu',
                                     batch_first=True, norm_first=True)


class TopologyAutoencoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.input = nn.Linear(39, 128)
        self.graph = nn.ModuleList(VertexFaceMessage() for _ in range(layers))
        self.attn = nn.ModuleList(encoder_layer() for _ in range(layers))
        self.moments = nn.Linear(128, 128)
        self.decode_input = nn.Linear(64, 128)
        self.decode_blocks = nn.ModuleList(encoder_layer() for _ in range(layers))
        self.output = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64))
        self.indicator = StoredIndicators()

    def encode(self, vertices, faces):
        count = vertices.shape[0]
        centers = vertices[faces].mean(1)
        nodes = torch.cat([vertices, centers])
        vertex_ids = faces.reshape(-1)
        face_ids = torch.arange(faces.shape[0], device=faces.device).repeat_interleave(3) + count
        source = torch.cat([vertex_ids, face_ids])
        target = torch.cat([face_ids, vertex_ids])
        degree = torch.bincount(target, minlength=len(nodes)).clamp_min(1).unsqueeze(-1)
        hidden = self.input(position_features(nodes))
        for graph, attention in zip(self.graph, self.attn):
            hidden = attention(graph(hidden, source, target, degree))
        return self.moments(hidden[:count]).chunk(2, dim=-1)

    def decode(self, latent):
        hidden = self.decode_input(latent)
        for attention in self.decode_blocks:
            hidden = attention(hidden)
        return self.output(hidden).chunk(2, dim=-1)

    def forward(self, vertices, faces):
        mean, _ = self.encode(vertices, faces)
        return self.decode(mean)


class ConditionalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(144, 432)
        self.out = nn.Linear(144, 144)
        self.ff = nn.Sequential(nn.Linear(144, 576), nn.GELU(), nn.Linear(576, 144))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(144, 864))

    def forward(self, hidden, condition, xyz, valid=None):
        batch, length, _ = hidden.shape
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.ada(condition).unsqueeze(1).chunk(6, -1)
        normalized = F.layer_norm(hidden, (144,), eps=1e-5)
        projected = self.qkv(normalized * (1 + scale_a) + shift_a)
        q, k, v = projected.reshape(batch, length, 3, 4, 36).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(rotary_xyz(q, xyz), rotary_xyz(k, xyz), v,
            attn_mask=None if valid is None else valid[:, None, None, :], dropout_p=0)
        hidden = hidden + gate_a * self.out(attended.transpose(1, 2).reshape(batch, length, 144))
        normalized = F.layer_norm(hidden, (144,), eps=1e-5)
        return hidden + gate_f * self.ff(normalized * (1 + scale_f) + shift_f)


class VertexConditionedFlow(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.input = nn.Linear(64, 144)
        self.position = nn.Linear(39, 144)
        self.time = nn.Sequential(nn.Linear(64, 144), nn.SiLU(), nn.Linear(144, 144))
        self.blocks = nn.ModuleList(ConditionalBlock() for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(144), nn.Linear(144, 64))

    def forward(self, latent, time, vertices, mask=None):
        hidden = self.input(latent) + self.position(position_features(vertices))
        condition = self.time(time_features(time))
        for block in self.blocks:
            hidden = block(hidden, condition, vertices, mask)
        return self.out(hidden)


class TextConditionedPointFlow(nn.Module):
    def __init__(self, layers, prior_width):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(()))
        self.text = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, 144),
                                  nn.SiLU(), nn.Linear(144, 144))
        self.count = nn.Sequential(nn.SiLU(), nn.Linear(144, 275))
        self.slot = nn.Embedding(274, 144)
        self.input = nn.Linear(39, 144)
        self.time = nn.Sequential(nn.Linear(64, 144), nn.SiLU(), nn.Linear(144, 144))
        self.blocks = nn.ModuleList(ConditionalBlock() for _ in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(144), nn.Linear(144, 3))
        self.coordinate_prior = nn.Sequential(nn.LayerNorm(2048), nn.Linear(2048, prior_width),
                                              nn.SiLU(), nn.Linear(prior_width, 822))

    def count_logits(self, features):
        return self.count(self.text(features))

    def prior(self, features):
        return self.coordinate_prior(features).reshape(-1, 274, 3)

    def denoise(self, points, time, features, mask=None):
        condition = self.text(features)
        hidden = self.input(position_features(points)) + self.slot.weight[:points.shape[1]]
        hidden = hidden + condition[:, None]
        modulation = self.time(time_features(time)) + condition
        for block in self.blocks:
            hidden = block(hidden, modulation, points, mask)
        return self.out(hidden)

    def forward(self, points, time, features, mask=None):
        return self.prior(features)[:, :points.shape[1]] + self.residual_scale * self.denoise(points, time, features, mask)


def load_network(path, kind):
    saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    config = saved.get('config', {})
    if kind == 'ae':
        network = TopologyAutoencoder(saved.get('layers', config.get('ae_layers', 2)))
    elif kind == 'topology':
        network = VertexConditionedFlow(saved.get('layers', config.get('flow_layers', 4)))
    elif kind == 'points':
        network = TextConditionedPointFlow(config['layers'], config['prior_width'])
    else:
        raise ValueError('kind must be ae, topology or points')
    network.load_state_dict(saved['state_dict'], strict=True)
    return network.eval()


@torch.inference_mode()
def generate_points(network, text_features, generator, steps=100):
    count = int(network.count_logits(text_features).argmax(-1).item())
    if not 1 <= count <= 274:
        raise ValueError(f'Predicted invalid point count {count}')
    points = torch.randn(1, count, 3, generator=generator,
                         device=text_features.device, dtype=text_features.dtype)
    for step in range(steps):
        time = points.new_tensor([step / steps])
        clean = network(points, time, text_features)
        points = points + (clean - points) / (1 - time[:, None, None]) / steps
    return points[0]


@torch.inference_mode()
def generate_topology_latent(network, vertices, mean, std, generator, steps=50):
    latent = torch.randn(1, len(vertices), 64, generator=generator,
                         device=vertices.device, dtype=vertices.dtype)
    for step in range(steps):
        time = latent.new_tensor([step / steps])
        latent = latent + network(latent, time, vertices[None]) / steps
    return latent[0] * std + mean
