"""Bind a Flow-only checkpoint policy without changing T1 parameters or source files."""
from __future__ import annotations

from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint


def _forward_packed(self, noisy, time, vertices, seg, condition,
                    attention="auto", use_checkpoint=False):
    # Same input preparation, packed_block and output as T1 model.py:108-126.
    from model import _fourier_xyz, packed_block, time_features

    if noisy.shape[0] != seg.tokens or vertices.shape != (seg.tokens, 3):
        raise ValueError("packed tokens do not match the segments")
    if time.shape != (len(seg.lengths),) or condition.shape[0] != len(seg.lengths):
        raise ValueError("one time and one condition set per object")
    dtype = self.data_embedding.weight.dtype
    x = self.data_embedding(noisy.to(dtype)) + self.position_embedding(_fourier_xyz(vertices).to(dtype))
    temb = self.time_embedding(time_features(time).to(dtype))
    positions = vertices.float() * self.cfg.rope_unit
    condition = condition.to(dtype)
    for index, block in enumerate(self.blocks):
        if use_checkpoint and torch.is_grad_enabled() and index < self._checkpoint_flow_layers:
            x = checkpoint(packed_block, block, x, temb, condition, positions, seg, attention,
                           use_reentrant=False)
        else:
            x = packed_block(block, x, temb, condition, positions, seg, attention)
    return self.output(self.output_norm(x))


def bind_checkpoint_policy(system, checkpoint_flow_layers):
    """First N Flow blocks recompute; other Flow blocks and all weights stay unchanged.

    The condition encoder is untouched. Call training with checkpoint_activations=1
    to retain its original checkpointing. Evaluation still disables recomputation.
    """
    count = len(system.flow.blocks)
    if not isinstance(checkpoint_flow_layers, int) or not 0 <= checkpoint_flow_layers <= count:
        raise ValueError(f"checkpoint_flow_layers must be an integer from 0 to {count}")
    system.flow._checkpoint_flow_layers = checkpoint_flow_layers
    system.flow.forward_packed = MethodType(_forward_packed, system.flow)
    return system
