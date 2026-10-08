#!/usr/bin/env python3
"""Profile one real FP32 Topology-AE optimizer step by major subsystem.

This is a diagnostic entry point, not a training entry point.  It uses the same
20 meshes, model, fixed face negatives, all-pair edge loss, per-mesh weighting,
gradient clipping and Adam update as the packed overfit experiment.  Runtime
wrappers add profiler labels without changing the model or loss implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Callable

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset  # noqa: E402
from mini_nexus.negative_candidates import TopologyNegativeCandidateStore  # noqa: E402
from mini_nexus.packed_topology import collate_packed_topology  # noqa: E402
from mini_nexus.training_2k import Nexus2KTopologyAESystem  # noqa: E402
import mini_nexus.training_2k as training_2k  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--negative-candidate-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument("--pair-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def stable_uid_seed(base_seed: int, step: int, uid: str) -> int:
    uid_value = int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little")
    return (base_seed + step * 1_000_003 + uid_value) % (2**63 - 1)


def run_step(
    model: Nexus2KTopologyAESystem,
    batches: list[object],
    optimizer: torch.optim.Optimizer,
    *,
    seed: int,
    step: int,
    expected_samples: int,
) -> tuple[float, float, float]:
    """Run one complete optimizer step and return loss, grad norm and seconds."""

    torch.cuda.synchronize()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device="cuda")
    for batch in batches:
        sample_seeds = tuple(stable_uid_seed(seed, step, uid) for uid in batch.uids)
        with record_function("nexus::forward_total"):
            loss = model(
                batch,
                negative_seed=seed,
                fixed_negatives=True,
                packed=True,
                sample_seeds=sample_seeds,
            )
        if loss.dtype != torch.float32:
            raise RuntimeError(f"topology-ae loss is not FP32: {loss.dtype}")
        sample_count = len(batch.uids)
        with record_function("nexus::backward_total"):
            (loss * sample_count / expected_samples).backward()
        loss_sum += loss.detach() * sample_count
    if any(
        parameter.grad is not None and parameter.grad.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise RuntimeError("topology-ae gradients are not FP32")
    with record_function("nexus::clip_and_adam"):
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if any(
        torch.is_tensor(value)
        and value.is_floating_point()
        and value.dtype != torch.float32
        for state in optimizer.state.values()
        for value in state.values()
    ):
        raise RuntimeError("topology-ae optimizer states are not FP32")
    torch.cuda.synchronize()
    return (
        float(loss_sum / expected_samples),
        float(gradient_norm),
        time.perf_counter() - started,
    )


def label_method(module: torch.nn.Module, label: str) -> None:
    """Wrap one module instance with a profiler-only user annotation."""

    original = module.forward

    def wrapped(self: torch.nn.Module, *args: object, **kwargs: object):
        with record_function(label):
            return original(*args, **kwargs)

    module.forward = MethodType(wrapped, module)


def label_function(module: object, name: str, label: str) -> None:
    original: Callable[..., object] = getattr(module, name)

    def wrapped(*args: object, **kwargs: object):
        with record_function(label):
            return original(*args, **kwargs)

    setattr(module, name, wrapped)


def event_times(event: object) -> tuple[float, float]:
    """Return inclusive CPU/device time in microseconds across PyTorch versions."""

    cpu_us = float(getattr(event, "cpu_time_total", 0.0))
    device_us = float(
        getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))
    )
    return cpu_us, device_us


def main() -> int:
    args = parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("failed to require highest FP32 matmul precision")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("failed to disable TF32 for topology-ae")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = Nexus2KManifestDataset(args.manifest, "train")
    if len(dataset) != args.expected_samples:
        raise ValueError(
            f"expected {args.expected_samples} rows, found {len(dataset)}"
        )
    samples = [dataset[index] for index in range(len(dataset))]
    # Singleton groups match the fastest validated FP32 baseline.  The model
    # still uses the packed path, so GraphSAGE indices and masks are identical.
    batches = [collate_packed_topology([sample]).to("cuda") for sample in samples]
    if any(batch.vertices.dtype != torch.float32 for batch in batches):
        raise RuntimeError("topology-ae vertex inputs are not FP32")
    store = TopologyNegativeCandidateStore(args.negative_candidate_root, "mixed_medium")
    model = Nexus2KTopologyAESystem(
        hidden_dim=512,
        latent_dim=64,
        spacetime_dim=32,
        num_heads=8,
        encoder_layers=24,
        decoder_layers=16,
        decoder_hidden_dim=1024,
        encoder_dropout=0.0,
        pair_chunk_size=args.pair_chunk_size,
        negative_candidate_store=store,
        fixed_overfit_face_negatives=True,
        face_interval_factor=0.25,
    ).to(device="cuda", dtype=torch.float32).train()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("topology-ae parameters are not FP32")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    # Warm up kernels and populate immutable positive-edge/face-negative caches.
    warmup = run_step(
        model,
        batches,
        optimizer,
        seed=args.seed,
        step=1,
        expected_samples=args.expected_samples,
    )

    for block in model.autoencoder.encoder_blocks:
        label_method(block.graph, "nexus::encoder_graphsage")
        label_method(block.transformer, "nexus::encoder_attention_ffn")
    for block in model.autoencoder.decoder_blocks:
        label_method(block.attention, "nexus::decoder_attention")
    label_function(training_2k, "paper_edge_loss_all_pairs", "nexus::edge_loss_all_pairs")
    label_function(training_2k, "face_interval_logits", "nexus::face_candidate_head")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        measured = run_step(
            model,
            batches,
            optimizer,
            seed=args.seed,
            step=2,
            expected_samples=args.expected_samples,
        )

    labels = {}
    all_events = []
    for event in profiler.key_averages():
        cpu_us, device_us = event_times(event)
        row = {
            "name": event.key,
            "calls": int(event.count),
            "cpu_total_ms": cpu_us / 1000.0,
            "device_total_ms": device_us / 1000.0,
        }
        all_events.append(row)
        if event.key.startswith("nexus::"):
            labels[event.key] = row
    top_cuda = sorted(
        all_events, key=lambda row: row["device_total_ms"], reverse=True
    )[:30]
    result = {
        "manifest_sha256": dataset.manifest_sha256,
        "uids": list(dataset.uids),
        "precision": "fp32",
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "pair_chunk_size": args.pair_chunk_size,
        "scoring_contract": model.scoring_contract(),
        "logit_scale_calibration": "not_run_in_profiler; explicit unit scales",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "warmup": {"loss": warmup[0], "gradient_norm": warmup[1], "seconds": warmup[2]},
        "profiled_step": {
            "loss": measured[0],
            "gradient_norm": measured[1],
            "wall_seconds": measured[2],
        },
        "labeled_events": labels,
        "top_device_events": top_cuda,
        "interpretation": (
            "Labeled module times cover forward kernels. backward_total is the "
            "whole backward pass and is not attributed to individual modules."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["profiled_step"], ensure_ascii=False), flush=True)
    for name, row in labels.items():
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
