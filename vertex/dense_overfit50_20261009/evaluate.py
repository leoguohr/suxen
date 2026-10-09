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

from common import (EVAL_SEEDS, build_model, cache_fixed_features, generate_tree, load_data,
                    load_weights, score_tree, summarize, write_json, sha)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--weights", choices=("raw", "ema"), default="raw")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--hash-checkpoint", action="store_true")
    args = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to ONE GPU"
    device = torch.device("cuda:0")
    assert not args.output.exists(), "output must be a new directory"
    (args.output / "predictions").mkdir(parents=True)
    started = time.monotonic()
    manifest, conditions, leaves, _levels, raw_gt = load_data(device)
    weights, state = load_weights(args.checkpoint, args.weights)
    model = build_model(weights, device).eval().requires_grad_(False)
    info = {"checkpoint": str(args.checkpoint), "weights": args.weights, "update": state.get("update"),
            "step": state.get("step"), "steps_per_depth": args.steps, "seeds": list(EVAL_SEEDS),
            "checkpoint_sha256": sha(args.checkpoint) if args.hash_checkpoint else None}
    del weights, state
    cache_fixed_features(model, conditions)
    generated = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for seed in EVAL_SEEDS:
            for index, uid in enumerate(manifest["train_uids"]):
                context = model.condition_encoder(conditions[index])
                status, levels, q = generate_tree(model, context, seed, steps=args.steps)
                path = args.output / "predictions" / f"seed-{seed}_{uid}.npz"
                np.savez_compressed(path, integer_vertices=q, status=np.asarray(status),
                                    **{f"depth{lv['depth']}_{k}": lv[k] for lv in levels
                                       for k in ("parents", "occupancy", "estimate", "predicted_cells")})
                generated.append({"seed": seed, "uid": uid, "index": index, "status": status,
                                  "levels": levels, "q": q,
                                  "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                write_json(args.output / "progress.json", {"generated": len(generated), "expected": 100,
                                                           "minutes": (time.monotonic() - started) / 60})
    write_json(args.output / "generation_manifest.json",
               {"gt_read": False, "rows": [{k: g[k] for k in ("seed", "uid", "status", "prediction_sha256")}
                                           for g in generated]})
    rows = []
    for g in generated:
        scored = score_tree(g["status"], g["levels"], g["q"], leaves[g["index"]], raw_gt[g["index"]])
        rows.append({"seed": g["seed"], "uid": g["uid"], **scored})
    report = summarize(rows)
    report["acceptance"] = {"criterion": "100/100 trees exact at every depth (50 objects x 2 seeds)",
                            "passed": report["full_trees_exact"] == 100}
    report["s0_reference"] = {"full_trees_exact": 7, "per_depth_full_level_exact": [98, 88, 58, 32, 16, 14, 12, 7, 7]}
    write_json(args.output / "evaluation.json", {**info, **report, "rows": rows})
    write_json(args.output / "summary.json", {**info, **report, "minutes": (time.monotonic() - started) / 60})
    print(f"EVALUATION_COMPLETE full_trees_exact={report['full_trees_exact']}/100 "
          f"per-depth exact={[d['full_level_exact'] for d in report['per_depth']]} "
          f"first-mismatch={report['first_mismatch_depth_histogram']} "
          f"passed={report['acceptance']['passed']}", flush=True)


if __name__ == "__main__":
    main()
