"""Native, versioned original512/V2 models. No runtime replacement or hooks."""
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

VARIANTS = ('A_v1_recipe_control', 'B_v2_teacher_blocks')


@dataclass(frozen=True)
class Config:
    model_variant: str = VARIANTS[0]
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
    activation_checkpointing: bool = True
    coordinate_encoding: str = 'fourier'
    graph_norm_position: str = 'post_projection'
    graph_activation: str = 'gelu'
    encoder_activation: str = 'gelu'

    def to_dict(self):
        return asdict(self)


@contextmanager
def deterministic_reduction():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


class DeterministicGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nodes, source):
        ctx.save_for_backward(source)
        ctx.shape = nodes.shape
        return nodes[source]

    @staticmethod
    def backward(ctx, gradient):
        source, = ctx.saved_tensors
        with deterministic_reduction():
            result = gradient.new_zeros(ctx.shape)
            result.index_add_(0, source, gradient)
        return result, None


@dataclass
class Graph:
    source: torch.Tensor
    target: torch.Tensor
    degree: torch.Tensor

    @classmethod
    def from_faces(cls, faces, vertex_count):
        vertices = faces.reshape(-1)
        face_nodes = (torch.arange(len(faces), device=faces.device) + vertex_count).repeat_interleave(3)
        source = torch.cat((vertices, face_nodes))
        target = torch.cat((face_nodes, vertices))
        degree = torch.zeros(vertex_count + len(faces), 1, device=faces.device, dtype=torch.float32)
        with deterministic_reduction():
            degree.index_add_(0, target, torch.ones(len(target), 1, device=faces.device))
        return cls(source, target, degree.clamp_min(1))

    def to(self, device):
        return Graph(self.source.to(device), self.target.to(device), self.degree.to(device))

    def mean(self, nodes):
        with deterministic_reduction():
            total = torch.zeros_like(nodes)
            total.index_add_(0, self.target, DeterministicGather.apply(nodes, self.source))
        return total / self.degree


class FourierXYZ(nn.Module):
    def __init__(self, bands, enabled=True):
        super().__init__()
        self.enabled = enabled
        self.bands = bands
        self.register_buffer('frequencies', math.pi * 2.0 ** torch.arange(bands, dtype=torch.float32))

    def forward(self, xyz):
        if not self.enabled:
            # Keep input weights and initialization identical; only remove sin/cos features.
            return torch.cat((xyz, xyz.new_zeros(*xyz.shape[:-1], 6*self.bands)), dim=-1)
        phase = xyz.unsqueeze(-1) * self.frequencies
        return torch.cat((xyz, phase.sin().flatten(-2), phase.cos().flatten(-2)), dim=-1)


def math_attention(attention, tokens):
    """The existing math00 FP32 QKV/layout/SDPA/output-projection operations."""
    width, heads = attention.embed_dim, attention.num_heads
    with torch.autocast(device_type=tokens.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        qkv = F.linear(tokens, attention.in_proj_weight, attention.in_proj_bias)
        qkv = qkv.reshape(len(tokens), 3, heads, width // heads).contiguous()
        q, k, v = [qkv[:, j].transpose(0, 1).unsqueeze(0) for j in range(3)]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False)
        out = out.squeeze(0).transpose(0, 1).reshape(len(tokens), width)
        return F.linear(out, attention.out_proj.weight, attention.out_proj.bias)


class GraphProjection(nn.Module):
    def __init__(self, width, neighbor_bias):
        super().__init__()
        self.self_projection = nn.Linear(width, width)
        self.neighbor_projection = nn.Linear(width, width, bias=neighbor_bias)

    def forward(self, nodes, graph):
        return self.self_projection(nodes) + self.neighbor_projection(graph.mean(nodes))


class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.v2 = cfg.model_variant == VARIANTS[1]
        assert cfg.graph_norm_position in ('post_projection', 'pre_aggregation')
        assert cfg.graph_activation in ('gelu', 'silu')
        assert cfg.encoder_activation in ('gelu', 'relu')
        self.graph_norm_position = cfg.graph_norm_position
        self.graph_activation = F.gelu if cfg.graph_activation == 'gelu' else F.silu
        self.graph_norm = nn.LayerNorm(cfg.encoder_width, eps=cfg.layer_norm_eps)
        self.graph = GraphProjection(cfg.encoder_width, neighbor_bias=not self.v2)
        self.transformer = nn.TransformerEncoderLayer(cfg.encoder_width, cfg.heads,
            cfg.ffn_ratio*cfg.encoder_width, dropout=0., activation=cfg.encoder_activation if self.v2 else 'relu',
            layer_norm_eps=cfg.layer_norm_eps, batch_first=True, norm_first=True)

    def forward(self, h, graph):
        if self.v2:
            message = (self.graph_norm(self.graph(h, graph)) if self.graph_norm_position == 'post_projection'
                       else self.graph(self.graph_norm(h), graph))
            h = h + self.graph_activation(message)
        else:
            h = h + F.silu(self.graph(self.graph_norm(h), graph))
        layer = self.transformer
        h = h + math_attention(layer.self_attn, layer.norm1(h))
        return h + layer.linear2(layer.activation(layer.linear1(layer.norm2(h))))


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.decoder_width, eps=cfg.layer_norm_eps)
        self.attention = nn.MultiheadAttention(cfg.decoder_width, cfg.heads, dropout=0., batch_first=True)
        self.v2 = cfg.model_variant == VARIANTS[1]
        if self.v2:
            self.ffn_norm = nn.LayerNorm(cfg.decoder_width, eps=cfg.layer_norm_eps)
            self.ffn = nn.Sequential(nn.Linear(cfg.decoder_width, cfg.ffn_ratio*cfg.decoder_width),
                nn.GELU(), nn.Linear(cfg.ffn_ratio*cfg.decoder_width, cfg.decoder_width))

    def forward(self, h):
        h = h + math_attention(self.attention, self.norm(h))
        return h + self.ffn(self.ffn_norm(h)) if self.v2 else h


class NativeTopologyAE(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        assert cfg.model_variant in VARIANTS and cfg.dropout == 0
        assert cfg.encoder_width % cfg.heads == cfg.decoder_width % cfg.heads == 0
        assert cfg.embedding_width == 32
        self.cfg = cfg
        assert cfg.coordinate_encoding in ('fourier', 'xyz_only')
        self.position_features = FourierXYZ(cfg.fourier_bands, cfg.coordinate_encoding == 'fourier') if cfg.model_variant == VARIANTS[1] else nn.Identity()
        input_width = 3 + 6*cfg.fourier_bands if cfg.model_variant == VARIANTS[1] else 3
        self.vertex_input = nn.Linear(input_width, cfg.encoder_width)
        self.face_input = nn.Linear(input_width, cfg.encoder_width)
        self.encoder_blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.encoder_composite_blocks))
        self.encoder_output_norm = nn.LayerNorm(cfg.encoder_width, eps=cfg.layer_norm_eps)
        self.mu = nn.Linear(cfg.encoder_width, cfg.latent_width)
        self.log_variance = nn.Linear(cfg.encoder_width, cfg.latent_width)
        self.log_variance.requires_grad_(False)
        self.latent_input = nn.Linear(cfg.latent_width, cfg.decoder_width)
        self.decoder_blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.decoder_blocks))
        self.decoder_output_norm = nn.LayerNorm(cfg.decoder_width, eps=cfg.layer_norm_eps)
        self.edge_embedding = nn.Linear(cfg.decoder_width, cfg.embedding_width)
        self.face_embedding = nn.Linear(cfg.decoder_width, cfg.embedding_width)

    def forward(self, vertices, faces, *, sample_latent=False, graph=None, generator=None):
        assert vertices.dtype == torch.float32 and faces.dtype == torch.long
        assert vertices.ndim == faces.ndim == 2 and vertices.shape[1] == faces.shape[1] == 3
        graph = Graph.from_faces(faces, len(vertices)) if graph is None else graph
        h = torch.cat((self.vertex_input(self.position_features(vertices)),
            self.face_input(self.position_features(vertices[faces].mean(1)))))
        recompute = self.cfg.activation_checkpointing and torch.is_grad_enabled()
        with set_checkpoint_early_stop(False):
            for block in self.encoder_blocks:
                h = checkpoint(block, h, graph, use_reentrant=False, preserve_rng_state=False) if recompute else block(h, graph)
            hv = self.encoder_output_norm(h[:len(vertices)])
            mu = self.mu(hv)
            logvar = self.log_variance(hv).clamp(-20, 10)
            z = mu
            if sample_latent:
                z = mu + (0.5*logvar).exp()*torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
            h = self.latent_input(z)
            for block in self.decoder_blocks:
                h = checkpoint(block, h, use_reentrant=False, preserve_rng_state=False) if recompute else block(h)
        h = self.decoder_output_norm(h)
        e, f = self.edge_embedding(h), self.face_embedding(h)
        return dict(mu=mu, log_variance=logvar, latent=z,
            edge=e-e.mean(0, keepdim=True), face=f-f.mean(0, keepdim=True), decoder_hidden=h)
