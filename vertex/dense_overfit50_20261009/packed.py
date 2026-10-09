"""Packed forward for the S0 VertexDiT: many (object, depth, t) items per call.

Several items of the SAME object share one row. Self-attention is block-diagonal,
so tokens only see tokens of their own item, exactly as in the original
one-item-per-row forward. Time conditioning (AdaLN) is per item, depth/position
embeddings are per token, and every row cross-attends to its object's VecSet
tokens. Parameters are the unchanged S0 parameters; tests/test_cpu.py checks
numerical equivalence against the original VertexDiT.forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mini_nexus.vertex import _fourier_xyz, apply_vertex_rope, parent_centers


@dataclass
class Item:
    obj: int          # index into the objects of this update (0..K-1)
    depth: int        # target child depth 1..9
    codes: Tensor     # [n,3] long parent codes at depth-1
    target: Tensor    # [n,8] float multi-hot child occupancy
    time: float
    noise: Tensor     # [n,8] float
    weight: float     # loss weight; all items of one update sum to 1


@dataclass
class Chunk:
    noisy: Tensor        # [B,N,8]
    velocity: Tensor     # [B,N,8] target velocity (x1 - x0)
    codes: Tensor        # [B,N,3] long
    depths: Tensor       # [B,N] long, 1 on padding
    item_ids: Tensor     # [B,N] long, -1 on padding, else index into item_* arrays
    item_times: Tensor   # [I]
    item_weights: Tensor  # [I]
    item_counts: Tensor  # [I]
    item_depths: Tensor  # [I]
    row_object: Tensor   # [B] long
    tokens: int          # padded tokens B*N
    item_index: list     # chunk-local item id -> index into the input item list


def pack_rows(items: list[Item], row_tokens: int) -> list[list[int]]:
    """First-fit-decreasing packing of item indices into rows, one object per row."""
    rows: list[list[int]] = []
    loads: list[int] = []
    owners: list[int] = []
    order = sorted(range(len(items)), key=lambda i: (items[i].obj, -len(items[i].codes)))
    for index in order:
        item = items[index]
        size = len(item.codes)
        if size > row_tokens:
            raise ValueError(f"item with {size} tokens exceeds row_tokens={row_tokens}")
        for row, (load, owner) in enumerate(zip(loads, owners)):
            if owner == item.obj and load + size <= row_tokens:
                rows[row].append(index)
                loads[row] += size
                break
        else:
            rows.append([index])
            loads.append(size)
            owners.append(item.obj)
    return rows


def make_chunks(items: list[Item], rows: list[list[int]], chunk_tokens: int,
                device: torch.device) -> list[Chunk]:
    """Group rows (longest first) so that padded tokens per chunk <= chunk_tokens."""
    lengths = [sum(len(items[i].codes) for i in row) for row in rows]
    order = sorted(range(len(rows)), key=lambda r: -lengths[r])
    groups: list[list[int]] = []
    for r in order:
        if groups:
            current = groups[-1]
            width = max(lengths[x] for x in current + [r])
            if width * (len(current) + 1) <= chunk_tokens:
                current.append(r)
                continue
        groups.append([r])
    chunks = []
    for group in groups:
        width = max(lengths[r] for r in group)
        batch = len(group)
        noisy = torch.zeros((batch, width, 8), device=device)
        velocity = torch.zeros((batch, width, 8), device=device)
        codes = torch.zeros((batch, width, 3), dtype=torch.long, device=device)
        depths = torch.ones((batch, width), dtype=torch.long, device=device)
        item_ids = torch.full((batch, width), -1, dtype=torch.long, device=device)
        times, weights, counts, item_depths, row_object, item_index = [], [], [], [], [], []
        local = 0
        for b, r in enumerate(group):
            offset = 0
            row_object.append(items[rows[r][0]].obj)
            for index in rows[r]:
                item = items[index]
                n = len(item.codes)
                sl = slice(offset, offset + n)
                t = item.time
                noisy[b, sl] = (1.0 - t) * item.noise + t * item.target
                velocity[b, sl] = item.target - item.noise
                codes[b, sl] = item.codes
                depths[b, sl] = item.depth
                item_ids[b, sl] = local
                times.append(t)
                weights.append(item.weight)
                counts.append(n)
                item_depths.append(item.depth)
                item_index.append(index)
                local += 1
                offset += n
        chunks.append(Chunk(
            noisy=noisy, velocity=velocity, codes=codes, depths=depths, item_ids=item_ids,
            item_times=torch.tensor(times, device=device, dtype=torch.float32),
            item_weights=torch.tensor(weights, device=device, dtype=torch.float32),
            item_counts=torch.tensor(counts, device=device, dtype=torch.float32),
            item_depths=torch.tensor(item_depths, device=device, dtype=torch.long),
            row_object=torch.tensor(row_object, device=device, dtype=torch.long),
            tokens=batch * width, item_index=item_index))
    return chunks


def _self_attention(attention, x: Tensor, positions: Tensor, attn_mask: Tensor) -> Tensor:
    """Same math as mini_nexus.vertex._Attention.forward (self case), full [B,1,N,N] mask."""
    batch_size, token_count, hidden_dim = x.shape
    q, k, v = attention.qkv(x).reshape(
        batch_size, token_count, 3, attention.num_heads, attention.head_dim).unbind(dim=2)
    q = attention.query_norm(q.transpose(1, 2))
    k = attention.key_norm(k.transpose(1, 2))
    v = v.transpose(1, 2)
    q, k = apply_vertex_rope(q, k, positions)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
    return attention.output(attended.transpose(1, 2).reshape(batch_size, token_count, hidden_dim))


def _block_forward(block, x, item_time_embedding, gather_index, condition, condition_valid,
                   positions, attn_mask, valid):
    """mini_nexus.vertex._VertexBlock.forward with per-item (not per-row) AdaLN."""
    modulation = block.modulation(item_time_embedding)[gather_index]
    shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = modulation.chunk(6, dim=-1)
    h = block.self_norm(x) * (1 + scale_sa) + shift_sa
    x = x + gate_sa * _self_attention(block.self_attention, h, positions, attn_mask)
    x = x + block.cross_attention(block.cross_norm(x), condition, key_mask=condition_valid)
    h = block.ffn_norm(x) * (1 + scale_ff) + shift_ff
    x = x + gate_ff * block.ffn(h)
    return x.masked_fill(~valid.unsqueeze(-1), 0)


def packed_flow_forward(flow, noisy: Tensor, codes: Tensor, depths: Tensor, item_ids: Tensor,
                        item_times: Tensor, condition_rows: Tensor,
                        use_checkpoint: bool = False) -> Tensor:
    """Velocity prediction [B,N,8] for packed rows. condition_rows is [B,M,C]."""
    if hasattr(flow, "final_modulation") or hasattr(flow, "global_condition"):
        raise NotImplementedError("packed forward implements the S0 architecture only")
    valid = item_ids >= 0
    batch_size, token_count = item_ids.shape
    safe_depths = depths.masked_fill(~valid, 1)
    safe_codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
    centers = parent_centers(safe_codes, safe_depths)
    positions = centers * (2 ** (flow.rope_reference_depth - 1))
    dtype = flow.data_embedding.weight.dtype
    x = flow.data_embedding(noisy.masked_fill(~valid.unsqueeze(-1), 0).to(dtype))
    x = x + flow.position_embedding(_fourier_xyz(centers).to(dtype))
    x = x + flow.depth_embedding(safe_depths - 1)
    x = x.masked_fill(~valid.unsqueeze(-1), 0)
    frequencies = 10_000.0 ** (-torch.arange(128, device=noisy.device, dtype=torch.float32) / 128)
    angles = (1000 * item_times.float()).unsqueeze(-1) * frequencies
    item_time_embedding = flow.time_embedding(
        torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype))
    gather_index = item_ids.clamp_min(0)
    same_item = (item_ids.unsqueeze(2) == item_ids.unsqueeze(1)) & valid.unsqueeze(1)
    diagonal = torch.eye(token_count, dtype=torch.bool, device=noisy.device).unsqueeze(0)
    attn_mask = (same_item | diagonal).unsqueeze(1)  # padded queries see only themselves
    condition = condition_rows.to(dtype)
    condition_valid = torch.ones(condition.shape[:2], dtype=torch.bool, device=condition.device)
    for block in flow.blocks:
        args = (block, x, item_time_embedding, gather_index, condition, condition_valid,
                positions, attn_mask, valid)
        if use_checkpoint and torch.is_grad_enabled():
            x = checkpoint(_block_forward, *args, use_reentrant=False)
        else:
            x = _block_forward(*args)
    return flow.output(flow.output_norm(x)).masked_fill(~valid.unsqueeze(-1), 0)


def chunk_loss(prediction: Tensor, chunk: Chunk) -> tuple[Tensor, Tensor]:
    """Weighted sum over items of per-item velocity MSE; also per-item MSE (detached)."""
    valid = chunk.item_ids >= 0
    error = (prediction.float() - chunk.velocity).square().sum(-1).masked_fill(~valid, 0)
    per_item_sum = torch.zeros_like(chunk.item_times).index_add(
        0, chunk.item_ids[valid], error[valid])
    per_item = per_item_sum / (chunk.item_counts * 8)
    return (per_item * chunk.item_weights).sum(), per_item.detach()
