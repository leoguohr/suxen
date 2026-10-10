"""Acceptance evaluation: 50 objects x 2 fixed seeds = 100 complete D9 trees.

Protocol identical to S0's evaluation (native_export.export_tree + evaluate_ab.score_tree):
root -> depth 9, the model's own parents at every level, 20 Euler steps per depth,
threshold 0.5, per-depth noise seeded with seed + 1000*depth, capacity abort above 4096
parents. All predictions are saved before ground truth is loaded.

PASS = 100/100 trees exact at every depth (vertex count exact and every vertex in the
correct depth-9 cell). Teacher-style numbers (count accuracy, matched RMSE, max error /
minimum GT spacing) are reported alongside.
"""
from __future__ import annotations

import os
# nexus-algo pairs protobuf 4.x with system onnx 1.16; torch.optim lazily imports onnx.
# S0 ran with the same setting. Must be set before torch is imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from common import build_model, cache_fixed_features, load_data, load_weights, run_evaluation, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--weights", choices=("raw", "ema"), default="raw")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--sampler", choices=("euler", "dpm2m"), default="euler",
                   help="per-level ODE solver; the paper uses 20 DPM-Solver steps (dpm2m = DPM-Solver++(2M))")
    p.add_argument("--hash-checkpoint", action="store_true")
    args = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to ONE GPU"
    device = torch.device("cuda:0")
    assert not args.output.exists(), "output must be a new directory"
    manifest, conditions, leaves, _levels, raw_gt = load_data(device)
    weights, state = load_weights(args.checkpoint, args.weights)
    model = build_model(weights, device).eval().requires_grad_(False)
    info = {"checkpoint": str(args.checkpoint), "weights": args.weights, "update": state.get("update"),
            "step": state.get("step"), "steps_per_depth": args.steps, "sampler": args.sampler,
            "checkpoint_sha256": sha(args.checkpoint) if args.hash_checkpoint else None}
    del weights, state
    cache_fixed_features(model, conditions)
    report = run_evaluation(model, manifest, conditions, leaves, raw_gt, args.output, steps=args.steps, info=info,
                            sampler=args.sampler)
    print(f"EVALUATION_COMPLETE full_trees_exact={report['full_trees_exact']}/100 "
          f"per-depth exact={[d['full_level_exact'] for d in report['per_depth']]} "
          f"first-mismatch={report['first_mismatch_depth_histogram']} "
          f"by-size={report['exact_by_vertex_count']} "
          f"passed={report['acceptance']['passed']}", flush=True)


if __name__ == "__main__":
    main()
