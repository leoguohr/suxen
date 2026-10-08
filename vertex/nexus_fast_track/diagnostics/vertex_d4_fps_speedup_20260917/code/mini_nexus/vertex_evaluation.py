"""Deterministic Vertex diagnostics; Euler sampling is an engineering baseline."""

from __future__ import annotations

import torch
from torch import Tensor

from .octree import expand_occupied_children
from .flow import euler_integrate


def binary_metrics(predicted: Tensor, target: Tensor) -> dict[str, float | int]:
    predicted, target = predicted.bool(), target.bool()
    tp = int((predicted & target).sum())
    fp = int((predicted & ~target).sum())
    fn = int((~predicted & target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "predicted_positive_fraction": float(predicted.float().mean()),
        "target_positive_fraction": float(target.float().mean()),
    }


def cell_metrics(predicted: Tensor, target: Tensor) -> dict[str, float | int]:
    predicted_set = set(map(tuple, predicted.cpu().tolist()))
    target_set = set(map(tuple, target.cpu().tolist()))
    tp = len(predicted_set & target_set)
    return {
        "generated_vertices": len(predicted_set), "target_vertices": len(target_set),
        "precision": tp / max(len(predicted_set), 1),
        "recall": tp / max(len(target_set), 1),
        "f1": 2 * tp / max(len(predicted_set) + len(target_set), 1),
        "iou": tp / max(len(predicted_set | target_set), 1),
    }


@torch.no_grad()
def denoise_level(model, level, context: Tensor, *, time: float, seed: int) -> dict:
    """One-step data estimate using true parents and a fixed Gaussian draw.

    This is a denoising diagnostic, not a generated-vertex quality measurement.
    The zero-velocity baseline uses exactly the same noisy input and time.
    """
    target = level.target.to(context.device).unsqueeze(0)
    codes = level.parent_codes.to(context.device).unsqueeze(0)
    depths = torch.tensor([level.depth], device=context.device)
    generator = torch.Generator(device=context.device).manual_seed(seed)
    noise = torch.randn(target.shape, generator=generator, device=context.device)
    noisy = (1 - time) * noise + time * target
    times = torch.tensor([time], device=context.device)
    prediction = model.flow(noisy, times, codes, depths, context).float()
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("non-finite denoising velocity")
    estimate = noisy + (1 - time) * prediction
    return {
        "time": time, "depth": level.depth, "parents": codes.shape[1],
        "velocity_mse": float((prediction - (target - noise)).square().mean()),
        "zero_velocity_mse": float((target - noise).square().mean()),
        **binary_metrics(estimate >= 0.5, target.bool()),
        "zero_velocity_baseline": binary_metrics(noisy >= 0.5, target.bool()),
    }


@torch.no_grad()
def trace_denoising(model, level, context: Tensor, *, time: float, seed: int) -> dict:
    """Observe residual/modulation scale without modifying weights or outputs.

    Finite loss can hide large residual activations after final LayerNorm.
    Hooks are removed even if the diagnostic forward fails.
    """
    activations = {}
    flow = model.flow
    first = flow.blocks[0]
    modules = {
        "data_embedding": flow.data_embedding,
        "position_embedding": flow.position_embedding,
        "depth_embedding": flow.depth_embedding,
        "time_embedding": flow.time_embedding,
        "first_self_attention_before_gate": first.self_attention,
        "first_cross_attention": first.cross_attention,
        "first_ffn_before_gate": first.ffn,
        "first_block_output": first,
        "last_block_output": flow.blocks[-1],
    }

    def statistics(value):
        value = value.float()
        return {"rms": float(value.square().mean().sqrt()),
                "max_abs": float(value.abs().max())}

    def capture(name):
        def hook(module, inputs, output):
            activations[name] = statistics(output)
        return hook

    def capture_modulation(module, inputs, output):
        names = ("shift_sa", "scale_sa", "gate_sa", "shift_ff", "scale_ff", "gate_ff")
        activations["first_modulation"] = {
            name: statistics(value) for name, value in zip(names, output.chunk(6, -1))
        }

    handles = [module.register_forward_hook(capture(name)) for name, module in modules.items()]
    handles.append(first.modulation.register_forward_hook(capture_modulation))
    try:
        metrics = denoise_level(model, level, context, time=time, seed=seed)
    finally:
        for handle in handles:
            handle.remove()
    return {"depth": level.depth, "time": time, "activations": activations,
            "denoising": metrics}


@torch.no_grad()
def sample_level(model, context: Tensor, parents: Tensor, depth: int,
                 noise: Tensor, *, steps: int) -> Tensor:
    """Sample one level with the same ODE used by full-tree generation.

    Parents may be GT for a single-level diagnostic or previously generated
    cells. Return continuous occupancies; the common decision threshold is 0.5.
    """
    depths = torch.tensor([depth], device=context.device)
    return euler_integrate(
        lambda value, times: model.flow(
            value, times, parents.unsqueeze(0), depths, context
        ).float(), noise, steps=steps,
    )


@torch.no_grad()
def generate_cells(model, context: Tensor, *, steps: int, seed: int,
                   max_parents: int = 10000) -> tuple[Tensor | None, dict]:
    """Euler flow from root to leaves, with no forced child or top-k repair.

    A capacity abort returns None, so a partial tree cannot be scored as leaves.
    An empty tree is a valid empty prediction and receives zero recall.
    """
    if steps < 1 or max_parents < 1 or context.shape[0] != 1:
        raise ValueError("positive steps/cap and a single-object context are required")
    device = context.device
    parents = torch.zeros(1, 3, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = []
    for depth in range(1, model.flow.max_depth + 1):
        if len(parents) > max_parents:
            return None, {"status": "capacity_abort", "depth": depth,
                          "parent_count": len(parents), "levels": rows}
        if len(parents) == 0:
            return parents, {"status": "empty", "depth": depth, "levels": rows}
        value = torch.randn((1, len(parents), 8), device=device, generator=generator)
        value = sample_level(model, context, parents, depth, value, steps=steps)
        if not torch.isfinite(value).all():
            raise FloatingPointError("non-finite Euler occupancy")
        occupancy = value[0] >= 0.5
        rows.append({"depth": depth, "parents": len(parents),
                     "children": int(occupancy.sum()),
                     "empty_parent_fraction": float((~occupancy.any(-1)).float().mean())})
        parents = expand_occupied_children(parents, occupancy)
        if len(parents) > max_parents:
            return None, {"status": "capacity_abort", "depth": depth,
                          "child_count": len(parents), "levels": rows}
    return parents, {"status": "complete" if len(parents) else "empty", "levels": rows}


@torch.no_grad()
def chamfer_mean_distance(predicted: Tensor, target: Tensor, depth: int) -> float | None:
    """Mean of both directional Euclidean NN distances in normalized space."""
    if len(predicted) == 0 or len(target) == 0:
        return None
    first = -1 + 2 * (predicted.float() + 0.5) / 2**depth
    second = -1 + 2 * (target.float() + 0.5) / 2**depth

    def direction(source, destination):
        return torch.cat([
            torch.cdist(chunk.unsqueeze(0), destination.unsqueeze(0))[0].amin(-1)
            for chunk in source.split(512)
        ]).mean()

    return float((direction(first, second) + direction(second, first)) / 2)
