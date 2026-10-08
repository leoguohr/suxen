#!/usr/bin/env python3
"""Measure one real 20-mesh Topology-AE optimizer step without saving a checkpoint.

This is a bounded performance smoke test, not a training entry point.  It uses the
same model, dynamic face negatives, per-mesh loss averaging, backward, gradient
clipping and Adam update as the overfit run, then writes timing evidence only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples  # noqa: E402
from mini_nexus.negative_candidates import TopologyNegativeCandidateStore  # noqa: E402
from mini_nexus.training_2k import Nexus2KTopologyAESystem  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--negative-candidate-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--pair-chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


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
    if args.micro_batch_size <= 0 or args.micro_batch_size > len(dataset):
        raise ValueError("micro_batch_size must be in [1, expected_samples]")
    batches = []
    for start in range(0, len(dataset), args.micro_batch_size):
        samples = [
            dataset[index]
            for index in range(start, min(start + args.micro_batch_size, len(dataset)))
        ]
        batches.append((collate_nexus2k_samples(samples).to("cuda"), len(samples), start))
    if any(batch.vertices.dtype != torch.float32 for batch, _, _ in batches):
        raise RuntimeError("topology-ae vertex inputs are not FP32")
    store = TopologyNegativeCandidateStore(args.negative_candidate_root, "mixed_medium")
    model_kwargs = {
        "hidden_dim": 512,
        "latent_dim": 64,
        "spacetime_dim": 32,
        "num_heads": 8,
        "encoder_layers": 24,
        "decoder_layers": 16,
        "decoder_hidden_dim": 1024,
        "encoder_dropout": 0.0,
        "face_negative_ratio": 4.0,
        "negative_candidate_store": store,
        "pair_chunk_size": args.pair_chunk_size,
        "face_interval_factor": 0.25,
    }
    model = (
        Nexus2KTopologyAESystem(**model_kwargs)
        .to(device="cuda", dtype=torch.float32)
        .train()
    )
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("topology-ae parameters are not FP32")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

    records = []
    total_steps = args.warmup_steps + args.measured_steps
    for local_step in range(1, total_steps + 1):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for batch, sample_count, sample_seed_offset in batches:
            loss = model(
                batch,
                negative_seed=args.seed + local_step,
                fixed_negatives=False,
                sample_seed_offset=sample_seed_offset,
            )
            if loss.dtype != torch.float32:
                raise RuntimeError(f"topology-ae loss is not FP32: {loss.dtype}")
            batch_weight = sample_count / args.expected_samples
            (loss * batch_weight).backward()
            loss_value += float(loss.detach()) * batch_weight
        if any(
            parameter.grad is not None and parameter.grad.dtype != torch.float32
            for parameter in model.parameters()
        ):
            raise RuntimeError("topology-ae gradients are not FP32")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if local_step == 1 and any(
            torch.is_tensor(value)
            and value.is_floating_point()
            and value.dtype != torch.float32
            for state in optimizer.state.values()
            for value in state.values()
        ):
            raise RuntimeError("topology-ae optimizer states are not FP32")
        torch.cuda.synchronize()
        record = {
            "phase": "warmup" if local_step <= args.warmup_steps else "measured",
            "step": local_step,
            "seconds": time.perf_counter() - started,
            "loss": loss_value,
            "gradient_norm_before_clip": float(gradient_norm.detach()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "manifest_sha256": dataset.manifest_sha256,
                "uids": list(dataset.uids),
                "micro_batch_size": args.micro_batch_size,
                "pair_chunk_size": args.pair_chunk_size,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "precision": "fp32",
                "scoring_contract": model.scoring_contract(),
                "logit_scale_calibration": "not_run_in_benchmark; explicit unit scales",
                "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
