"""Paper-scale topology flow: the vertex chat's S0 DiT with per-vertex latents in place of octree bits.

Nexus Sec. 4.1: the topology DiT "shares the architecture and hyperparameters as the vertex diffusion
model", takes vertex positions through 3D RoPE and the condition C as in the vertex stage. In this
project the vertex DiT is S0, so every block below is S0's `_VertexBlock` and the condition encoder is
S0's `VertexConditionEncoder`, imported unchanged from vendor/vertex_s0.py. Changes from S0:

  * input  Linear(512 -> 1536) on the normalized latent (S0: Linear(8 -> 1536) on octree bits)
  * output Linear(1536 -> 512), zero-initialized (S0: 1536 -> 8)
  * no depth embedding (topology has no octree depth)
  * positions are the GT/generated vertex XYZ in [-1,1]; Fourier input and RoPE are computed exactly as S0
    does for parent centres (RoPE in D9 grid units = XYZ * 256)

`forward_padded` runs S0's own block forward on a padded batch (the reference). `forward_packed` runs the
same parameters on all objects packed into one token axis; self-attention is block-diagonal per object
and cross-attention uses each object's own condition tokens. tests/test_cpu.py checks they agree.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from data import Segments
from vendor.vertex_s0 import (VertexConditionEncoder, _LayerNorm, _VertexBlock, _fourier_xyz,
                              apply_vertex_rope)

try:  # FlashAttention-2 varlen kernels; optional (falls back to SDPA per object)
    from flash_attn import flash_attn_varlen_func
except Exception:  # noqa: BLE001
    flash_attn_varlen_func = None


@dataclass(frozen=True)
class FlowConfig:
    latent_dim: int = 512
    hidden_dim: int = 1536
    num_layers: int = 36
    num_heads: int = 12
    condition_dim: int = 2048
    condition_layers: int = 8
    condition_heads: int = 16
    condition_tokens: int = 1024
    rope_unit: float = 256.0  # XYZ in [-1,1] * 2**(9-1): D9 grid units, as S0 (rope_reference_depth=9)

    def to_dict(self):
        return asdict(self)


PAPER = FlowConfig()
TINY = FlowConfig(latent_dim=8, hidden_dim=48, num_layers=2, num_heads=4, condition_dim=32,
                  condition_layers=2, condition_heads=4, condition_tokens=8)


def time_features(time: Tensor) -> Tensor:
    """S0's 256-dim sinusoid of 1000*t (cos then sin)."""
    frequencies = 10_000.0 ** (-torch.arange(128, device=time.device, dtype=torch.float32) / 128)
    angles = (1000 * time.float()).unsqueeze(-1) * frequencies
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class TopologyDiT(nn.Module):
    def __init__(self, cfg: FlowConfig):
        super().__init__()
        self.cfg = cfg
        self.data_embedding = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.position_embedding = nn.Linear(51, cfg.hidden_dim)  # XYZ + 8 sin/cos bands (S0 _fourier_xyz)
        self.time_embedding = nn.Sequential(nn.Linear(256, cfg.hidden_dim), nn.SiLU(),
                                            nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.blocks = nn.ModuleList([_VertexBlock(cfg.hidden_dim, cfg.condition_dim, cfg.num_heads)
                                     for _ in range(cfg.num_layers)])
        self.output_norm = _LayerNorm(cfg.hidden_dim, affine=False)
        self.output = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        # S0 initialization: Xavier-uniform linears with zero bias, zero AdaLN output, zero velocity head.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    # ----- reference path: S0's own block forward on a padded batch -----
    def forward_padded(self, noisy: Tensor, time: Tensor, vertices: Tensor, condition: Tensor,
                       mask: Tensor, condition_mask: Tensor | None = None) -> Tensor:
        valid = mask
        cvalid = (torch.ones(condition.shape[:2], dtype=torch.bool, device=condition.device)
                  if condition_mask is None else condition_mask)
        xyz = vertices.masked_fill(~valid.unsqueeze(-1), 0)
        dtype = self.data_embedding.weight.dtype
        x = self.data_embedding(noisy.masked_fill(~valid.unsqueeze(-1), 0).to(dtype))
        x = x + self.position_embedding(_fourier_xyz(xyz).to(dtype))
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        temb = self.time_embedding(time_features(time).to(dtype))
        positions = xyz * self.cfg.rope_unit
        condition = condition.masked_fill(~cvalid.unsqueeze(-1), 0).to(dtype)
        for block in self.blocks:
            x = block(x, temb, condition, positions, valid, cvalid)
        return self.output(self.output_norm(x)).masked_fill(~valid.unsqueeze(-1), 0)

    # ----- training / sampling path: all objects packed in one token axis -----
    def forward_packed(self, noisy: Tensor, time: Tensor, vertices: Tensor, seg: Segments,
                       condition: Tensor, attention: str = "auto", use_checkpoint: bool = False) -> Tensor:
        """noisy [T,L], time [K], vertices [T,3], condition [K,M,C] -> velocity [T,L]."""
        if noisy.shape[0] != seg.tokens or vertices.shape != (seg.tokens, 3):
            raise ValueError("packed tokens do not match the segments")
        if time.shape != (len(seg.lengths),) or condition.shape[0] != len(seg.lengths):
            raise ValueError("one time and one condition set per object")
        dtype = self.data_embedding.weight.dtype
        x = self.data_embedding(noisy.to(dtype)) + self.position_embedding(_fourier_xyz(vertices).to(dtype))
        temb = self.time_embedding(time_features(time).to(dtype))
        positions = vertices.float() * self.cfg.rope_unit
        condition = condition.to(dtype)
        for block in self.blocks:
            if use_checkpoint and torch.is_grad_enabled():
                x = checkpoint(packed_block, block, x, temb, condition, positions, seg, attention,
                               use_reentrant=False)
            else:
                x = packed_block(block, x, temb, condition, positions, seg, attention)
        return self.output(self.output_norm(x))


def _head_rms(norm: nn.Module, values: Tensor) -> Tensor:
    """vendor.vertex_s0._HeadRMSNorm applied to [T,H,D] instead of [B,H,N,D] (same arithmetic)."""
    if isinstance(norm, nn.Identity):
        return values
    x = values.float()
    x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)
    return (x * norm.weight.float()[None, :, :]).to(values.dtype)


def _use_flash(attention: str, q: Tensor) -> bool:
    if attention == "sdpa":
        return False
    available = (flash_attn_varlen_func is not None and q.is_cuda
                 and q.dtype in (torch.float16, torch.bfloat16))
    if attention == "flash" and not available:
        raise RuntimeError("attention=flash requested but flash_attn varlen is unavailable for this input")
    return available


def varlen_attention(q: Tensor, k: Tensor, v: Tensor, cu_q: Tensor, cu_k: Tensor,
                     max_q: int, max_k: int, attention: str) -> Tensor:
    """Block-diagonal attention. q [Tq,H,D], k/v [Tk,H,D]; segment i of q attends to segment i of k."""
    if _use_flash(attention, q):
        return flash_attn_varlen_func(q.contiguous(), k.contiguous(), v.contiguous(), cu_q, cu_k,
                                      max_q, max_k, dropout_p=0.0, causal=False)
    bq, bk = cu_q.tolist(), cu_k.tolist()
    outputs = []
    for i in range(len(bq) - 1):
        qs = q[bq[i]:bq[i + 1]].transpose(0, 1).unsqueeze(0)
        ks = k[bk[i]:bk[i + 1]].transpose(0, 1).unsqueeze(0)
        vs = v[bk[i]:bk[i + 1]].transpose(0, 1).unsqueeze(0)
        outputs.append(F.scaled_dot_product_attention(qs, ks, vs, dropout_p=0.0)[0].transpose(0, 1))
    return torch.cat(outputs, dim=0)


def packed_self_attention(attn, h: Tensor, positions: Tensor, seg: Segments, attention: str) -> Tensor:
    tokens, width = h.shape
    q, k, v = attn.qkv(h).reshape(tokens, 3, attn.num_heads, attn.head_dim).unbind(dim=1)
    q = _head_rms(attn.query_norm, q)
    k = _head_rms(attn.key_norm, k)
    q4, k4 = apply_vertex_rope(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                               positions.unsqueeze(0))
    q, k = q4[0].transpose(0, 1), k4[0].transpose(0, 1)
    out = varlen_attention(q, k, v, seg.cu, seg.cu, seg.max_len, seg.max_len, attention)
    return attn.output(out.reshape(tokens, width))


def packed_cross_attention(attn, h: Tensor, condition: Tensor, seg: Segments, attention: str) -> Tensor:
    tokens, width = h.shape
    objects, context = condition.shape[:2]
    q = attn.query(h).reshape(tokens, attn.num_heads, attn.head_dim)
    k, v = attn.key_value(condition).reshape(objects, context, 2, attn.num_heads, attn.head_dim).unbind(dim=2)
    q = _head_rms(attn.query_norm, q)
    k = _head_rms(attn.key_norm, k.reshape(objects * context, attn.num_heads, attn.head_dim))
    v = v.reshape(objects * context, attn.num_heads, attn.head_dim)
    cu_k = torch.arange(objects + 1, device=h.device, dtype=torch.int32) * context
    out = varlen_attention(q, k, v, seg.cu, cu_k, seg.max_len, context, attention)
    return attn.output(out.reshape(tokens, width))


def packed_block(block: _VertexBlock, x: Tensor, temb: Tensor, condition: Tensor, positions: Tensor,
                 seg: Segments, attention: str) -> Tensor:
    """vendor.vertex_s0._VertexBlock.forward with per-object AdaLN on packed tokens (no padding)."""
    modulation = block.modulation(temb)[seg.index]
    shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = modulation.chunk(6, dim=-1)
    h = block.self_norm(x) * (1 + scale_sa) + shift_sa
    x = x + gate_sa * packed_self_attention(block.self_attention, h, positions, seg, attention)
    x = x + packed_cross_attention(block.cross_attention, block.cross_norm(x), condition, seg, attention)
    h = block.ffn_norm(x) * (1 + scale_ff) + shift_ff
    return x + gate_ff * block.ffn(h)


class TopologyFlowSystem(nn.Module):
    """VecSet condition encoder + topology DiT. Attribute names match S0 (condition_encoder, flow)."""

    def __init__(self, cfg: FlowConfig = PAPER):
        super().__init__()
        self.cfg = cfg
        self.condition_encoder = VertexConditionEncoder(
            input_dim=6, hidden_dim=cfg.condition_dim, num_tokens=cfg.condition_tokens,
            num_heads=cfg.condition_heads, num_layers=cfg.condition_layers)
        self.flow = TopologyDiT(cfg)

    def encode_condition(self, points: Tensor, valid: Tensor, fps_indices: Tensor,
                         use_checkpoint: bool = False) -> Tensor:
        """VertexConditionEncoder.forward with precomputed FPS indices (fixed inputs)."""
        enc = self.condition_encoder
        points = points.masked_fill(~valid.unsqueeze(-1), 0)
        features = torch.cat([_fourier_xyz(points[..., :3]), points[..., 3:].float()], dim=-1)
        embedded = enc.point_embedding(features.to(enc.point_embedding.weight.dtype))
        queries = embedded.gather(1, fps_indices.unsqueeze(-1).expand(-1, -1, enc.hidden_dim))
        grad = use_checkpoint and torch.is_grad_enabled()
        x = (checkpoint(enc.blocks[0], queries, embedded, valid, use_reentrant=False) if grad
             else enc.blocks[0](queries, embedded, valid))
        for block in enc.blocks[1:]:
            x = checkpoint(block, x, use_reentrant=False) if grad else block(x)
        return enc.output_norm(x)

    def velocity(self, noisy, time, data, condition, attention="auto", use_checkpoint=False):
        return self.flow.forward_packed(noisy, time, data.vertices, data.seg, condition, attention,
                                        use_checkpoint)


NEW_LAYER_PREFIXES = ("flow.data_embedding.", "flow.output.")
REQUIRED_TRANSFER_PREFIXES = ("flow.blocks.", "flow.time_embedding.", "flow.position_embedding.",
                              "condition_encoder.")


def transfer_vertex_weights(system: TopologyFlowSystem, s0_state: dict) -> dict:
    """Run B warm start: copy every S0 tensor with the same name and shape, except the new latent
    input/output layers. S0's depth embedding and octree in/out layers are not used."""
    own = system.state_dict()
    copied, kept = [], []
    update = {}
    for name, tensor in own.items():
        source = s0_state.get(name)
        if (not name.startswith(NEW_LAYER_PREFIXES) and source is not None
                and tuple(source.shape) == tuple(tensor.shape)):
            update[name] = source.to(tensor.dtype)
            copied.append(name)
        else:
            kept.append(name)
    missing = [n for n in own if n.startswith(REQUIRED_TRANSFER_PREFIXES) and n not in update]
    if missing:
        raise ValueError(f"S0 state lacks {len(missing)} required tensors, e.g. {missing[:3]}")
    system.load_state_dict({**own, **update}, strict=True)
    unused = sorted(set(s0_state) - set(copied))
    return dict(copied=len(copied), kept_fresh=kept, s0_unused=unused)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
