"""Generate normalized topology latents from Gaussian noise (t = 0) for packed objects.

Inputs are only the vertex XYZ and the point-cloud condition; no posterior, face or edge enters.
The initial noise of object `uid` under evaluation seed `s` comes from a private CPU generator keyed by
sha256("topology-overfit50/eval/{s}/{uid}"), so it does not depend on batching, device or training RNG.
"""
from __future__ import annotations

import contextlib

import torch

from common import generator
from vendor.samplers import dpm_solver_pp_2m, euler

SAMPLERS = {"euler": euler, "dpm2m": dpm_solver_pp_2m}


def eval_noise(data, seed: int) -> torch.Tensor:
    parts = [torch.randn((n, data.mu.shape[1]), generator=generator(f"topology-overfit50/eval/{seed}/{uid}"),
                         dtype=torch.float32)
             for uid, n in zip(data.uids, data.lengths)]
    return torch.cat(parts).to(data.device)


def autocast_context(device, dtype: str):
    if dtype == "fp32" or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def generate(system, data, seed: int, steps: int, sampler: str = "euler", dtype: str = "bf16",
             attention: str = "auto", max_tokens: int = 60000) -> torch.Tensor:
    """Normalized latents [T,512] in data order. Objects are processed in groups of <= max_tokens."""
    if data.fps_indices is None:
        raise ValueError("data.compute_fps(...) first")
    was_training = system.training
    system.eval()
    noise = eval_noise(data, seed)
    out = torch.empty_like(noise)
    groups, current, size = [], [], 0
    for i, n in enumerate(data.lengths):
        if current and size + n > max_tokens:
            groups.append(current)
            current, size = [], 0
        current.append(i)
        size += n
    groups.append(current)
    try:
        for group in groups:
            sub = data.select(group)
            lo, hi = data.token_slice(group[0]).start, data.token_slice(group[-1]).stop
            with autocast_context(data.device, dtype):
                condition = system.encode_condition(sub.points, sub.point_valid, sub.fps_indices)

            def velocity(x, s):
                time = torch.full((sub.count,), float(s), device=data.device)
                with autocast_context(data.device, dtype):
                    v = system.velocity(x, time, sub, condition, attention)
                v = v.float()
                if not torch.isfinite(v).all():
                    raise FloatingPointError("nonfinite velocity")
                return v

            out[lo:hi] = SAMPLERS[sampler](velocity, noise[lo:hi], steps)
    finally:
        system.train(was_training)
    return out
