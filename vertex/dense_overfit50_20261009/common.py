"""Shared data/model/evaluation helpers. Generation and scoring are copied from the
S0 evaluation path (native_export.export_tree, evaluate_ab.score_tree/summarize) so
results are directly comparable with S0's 7/100."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
from architecture_variants import construct_model, cache_fixed_features  # noqa: E402
from mini_nexus.octree import build_octree_levels, decode_leaf_centers, expand_occupied_children  # noqa: E402
from mini_nexus.vertex_evaluation import sample_level  # noqa: E402
from samplers import SAMPLERS, dpm_solver_pp_2m  # noqa: E402

DATA = HERE / "data"
DEPTH = 9
EVAL_SEEDS = (97029000, 97029001)
OFFSETS = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.int64)
S0_SHA256 = "3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935"


def sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, value) -> None:
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_data(device, data=DATA):
    """The fixed 50 objects: point clouds [1,8192,6], D9 leaves, per-depth octree levels."""
    manifest = json.loads((data / "prepared_d9" / "manifest.json").read_text())
    assert len(manifest["records"]) == 50 and manifest["train_uids"] == [r["uid"] for r in manifest["records"]]
    conditions, leaves, levels, raw_gt = [], [], [], []
    for record in manifest["records"]:
        condition = data / record["condition_file"]
        label = data / "prepared_d9" / record["label_file"]
        source_label = data / record["source_label_file"]
        assert sha(condition) == record["condition_sha256"], condition
        assert sha(label) == record["label_sha256"], label
        assert sha(source_label) == record["source_label_sha256"], source_label
        with np.load(condition, allow_pickle=False) as item:
            cloud = np.concatenate((item["points"], item["normals"]), axis=-1).astype(np.float32)
        assert cloud.shape == (8192, 6) and np.isfinite(cloud).all()
        conditions.append(torch.from_numpy(cloud)[None].to(device))
        with np.load(label, allow_pickle=False) as item:
            assert int(item["depth"]) == DEPTH
            cells = item["leaves"].copy()
        with np.load(source_label, allow_pickle=False) as item:
            raw_gt.append(item["gt_normalized"].copy())
        leaves.append(cells)
        native = build_octree_levels(torch.from_numpy(cells), DEPTH)
        levels.append([(lv.parent_codes.to(device), lv.target.to(device)) for lv in native])
    return manifest, conditions, leaves, levels, raw_gt


def build_model(state_dict, device):
    """S0 architecture (36x1536 DiT + 8-block VecSet), strict load."""
    with torch.device("meta"):
        model = construct_model("S0")
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.to(device)
    for p in model.parameters():
        assert p.dtype == torch.float32
    return model


def load_weights(path, which="raw"):
    """Accepts the S0 sweep checkpoint ('model') or this script's checkpoints ('model'/'ema')."""
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if which == "ema":
        if "ema" not in state:
            raise KeyError("checkpoint has no EMA weights")
        return state["ema"], state
    return state["model"], state


def noise_for(shape, seed, device):
    """Identical to scripts.train_vertex_staged.noise_for used by S0's evaluation."""
    return torch.randn(shape, device=device, generator=torch.Generator(device=device).manual_seed(seed))


def dpm_sample_level(model, context, parents, depth, noise, *, steps):
    """One level with DPM-Solver++(2M) (samplers.py); same model call as mini_nexus sample_level."""
    depths = torch.tensor([depth], device=context.device)
    def velocity(x, s):
        times = torch.full((x.shape[0],), s, device=x.device, dtype=x.dtype)
        return model.flow(x, times, parents.unsqueeze(0), depths, context).float()
    return dpm_solver_pp_2m(velocity, noise, steps)


@torch.no_grad()
def generate_tree(model, context, seed, *, steps=20, max_depth=DEPTH, capacity=4096, sampler="euler"):
    """Same protocol as native_export.export_tree: root -> D9, own parents, Euler, >=0.5.
    sampler='dpm2m' swaps only the per-level ODE solver (the paper uses 20 DPM-Solver steps)."""
    assert sampler in SAMPLERS, sampler
    parents = torch.zeros((1, 3), device=context.device, dtype=torch.long)
    levels, status = [], "complete"
    for depth in range(1, max_depth + 1):
        if len(parents) > capacity:
            status = "capacity_abort"
            break
        noise = noise_for((1, len(parents), 8), seed + depth * 1000, context.device)
        solve = sample_level if sampler == "euler" else dpm_sample_level
        value = solve(model, context, parents, depth, noise, steps=steps) if len(parents) else noise
        if not torch.isfinite(value).all():
            raise FloatingPointError("nonfinite occupancy")
        occupancy = value[0] >= 0.5
        predicted = expand_occupied_children(parents, occupancy)
        levels.append({"depth": depth, "parents": parents.cpu().numpy(), "estimate": value.cpu().numpy(),
                       "occupancy": occupancy.cpu().numpy(), "predicted_cells": predicted.cpu().numpy()})
        parents = predicted
    q = parents.cpu().numpy() if status == "complete" else np.empty((0, 3), dtype=np.int64)
    return status, levels, q


def point_metrics(p, g):
    """Teacher metric (teacher_candidate/replay.py): Hungarian match, RMSE, max error / min GT spacing."""
    from scipy.optimize import linear_sum_assignment
    p, g = np.asarray(p, dtype=np.float64), np.asarray(g, dtype=np.float64)
    if len(p) != len(g) or len(p) == 0:
        return {"status": "not_applicable_count_mismatch_or_empty", "predicted_count": len(p),
                "target_count": len(g), "matched_coordinate_rmse": None}
    dist = ((p[:, None] - g[None, :]) ** 2).sum(-1)
    row, col = linear_sum_assignment(dist)
    pred_to_gt = np.empty(len(p), np.int64)
    pred_to_gt[row] = col
    delta = p - g[pred_to_gt]
    gd = ((g[:, None] - g[None, :]) ** 2).sum(-1)
    np.fill_diagonal(gd, np.inf)
    spacing = float(np.sqrt(gd.min())) if len(g) > 1 else None
    max_error = float(np.sqrt((delta ** 2).sum(-1)).max())
    return {"status": "applicable_equal_count", "predicted_count": len(p), "target_count": len(g),
            "matched_coordinate_rmse": float(np.sqrt(np.mean(delta ** 2))), "max_error": max_error,
            "minimum_spacing": spacing,
            "spacing_ratio": max_error / spacing if spacing else None}


def score_tree(status, levels, q, target_leaves, raw_gt):
    """Copy of evaluate_ab.score_tree, working on in-memory arrays."""
    out_levels = []
    by_depth = {lv["depth"]: lv for lv in levels}
    for depth in range(1, DEPTH + 1):
        truth = set(map(tuple, np.unique(target_leaves >> (DEPTH - depth), axis=0)))
        record = by_depth.get(depth)
        if record is None:
            predicted, parent_bits = set(), None
        else:
            predicted = set(map(tuple, record["predicted_cells"]))
            parents, bits = record["parents"], record["occupancy"].astype(bool)
            children = parents[:, None] * 2 + OFFSETS[None]
            actual = np.array([tuple(c) in truth for c in children.reshape(-1, 3)], dtype=bool).reshape(len(parents), 8)
            parent_bits = {"tp": int((bits & actual).sum()), "fp": int((bits & ~actual).sum()),
                           "fn": int((~bits & actual).sum()), "tn": int((~bits & ~actual).sum())}
        tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
        out_levels.append({"depth": depth, "generated": record is not None,
                           "exact": record is not None and tp == len(truth) and fp == 0,
                           "global_cells_tp": tp, "global_cells_fp": fp, "global_cells_fn": fn,
                           "global_cells_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
                           "generated_parent_bits": parent_bits})
    complete = status == "complete" and len(levels) == DEPTH
    native_exact = complete and set(map(tuple, q)) == set(map(tuple, target_leaves))
    all_exact = complete and all(lv["exact"] for lv in out_levels)
    first = next((lv["depth"] for lv in out_levels if not lv["exact"]), None)
    if complete:
        xyz = decode_leaf_centers(torch.from_numpy(q), DEPTH).numpy()
        d9_centers = decode_leaf_centers(torch.from_numpy(target_leaves), DEPTH).numpy()
        d9_metric = point_metrics(xyz, d9_centers)
        unique_metric = point_metrics(xyz, np.unique(raw_gt, axis=0))
    else:
        d9_metric = unique_metric = {"status": "not_applicable_incomplete_generation", "matched_coordinate_rmse": None}
    return {"status": status, "predicted_n": int(len(q)), "target_n": int(len(target_leaves)),
            "count_correct": bool(complete and len(q) == len(target_leaves)),
            "native_final_set_exact": bool(native_exact), "all_levels_exact": bool(all_exact),
            "first_mismatch_depth": first, "levels": out_levels,
            "d9_gt_xyz": d9_metric, "unique_float_gt_xyz": unique_metric}


def summarize(rows):
    depths = []
    for depth in range(1, DEPTH + 1):
        lv = [row["levels"][depth - 1] for row in rows]
        tp = sum(x["global_cells_tp"] for x in lv)
        fp = sum(x["global_cells_fp"] for x in lv)
        fn = sum(x["global_cells_fn"] for x in lv)
        depths.append({"depth": depth, "full_level_exact": sum(x["exact"] for x in lv),
                       "global_cells_tp": tp, "global_cells_fp": fp, "global_cells_fn": fn,
                       "global_cells_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0})
    first = {}
    for row in rows:
        key = str(row["first_mismatch_depth"])
        first[key] = first.get(key, 0) + 1
    by_size = {}
    for low, high in ((0, 300), (300, 700), (700, 1200), (1200, 10**9)):
        bucket = [r for r in rows if low <= r["target_n"] < high]
        label = f"{low}-{high}" if high < 10**9 else f"{low}+"
        by_size[label] = {"trees": len(bucket), "exact": sum(r["all_levels_exact"] for r in bucket)}
    ratios = [row["d9_gt_xyz"].get("spacing_ratio") for row in rows]
    teacher_rule = sum(1 for row, r in zip(rows, ratios) if row["count_correct"] and r is not None and r < 0.1)
    return {"trees": len(rows), "full_trees_exact": sum(r["all_levels_exact"] for r in rows),
            "native_final_sets_exact": sum(r["native_final_set_exact"] for r in rows),
            "capacity_abort_trees": sum(r["status"] == "capacity_abort" for r in rows),
            "count_accuracy": sum(r["count_correct"] for r in rows) / max(len(rows), 1),
            "teacher_rule_vs_d9_gt_pass": teacher_rule,
            "first_mismatch_depth_histogram": first, "exact_by_vertex_count": by_size, "per_depth": depths}


@torch.no_grad()
def coarse_probe(model, conditions, leaves, *, seed=EVAL_SEEDS[0], max_depth=5, steps=20):
    """Cheap in-training probe: root -> depth `max_depth` for all 50 objects with own parents."""
    was_training = model.training
    model.eval()
    exact = [0] * max_depth
    first_error = {}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=conditions[0].is_cuda):
        for index, cloud in enumerate(conditions):
            context = model.condition_encoder(cloud)
            status, levels, _ = generate_tree(model, context, seed, steps=steps, max_depth=max_depth)
            first = None
            for lv in levels:
                d = lv["depth"]
                truth = set(map(tuple, np.unique(leaves[index] >> (DEPTH - d), axis=0)))
                ok = set(map(tuple, lv["predicted_cells"])) == truth
                if ok and first is None:
                    exact[d - 1] += 1
                elif first is None:
                    first = d
            first_error[str(first)] = first_error.get(str(first), 0) + 1
    model.train(was_training)
    return {"seed": seed, "max_depth": max_depth, "objects": len(conditions),
            "exact_through_depth": exact, "first_error_depth": first_error}


def run_evaluation(model, manifest, conditions, leaves, raw_gt, output, *, steps=20, info=None, sampler="euler"):
    """The 100-tree acceptance evaluation (50 objects x EVAL_SEEDS). Saves every prediction
    before any ground truth is read, then scores. Returns the summary dict."""
    import hashlib
    import time
    output = Path(output)
    (output / "predictions").mkdir(parents=True)
    started = time.monotonic()
    was_training = model.training
    model.eval()
    generated = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=conditions[0].is_cuda):
        for seed in EVAL_SEEDS:
            for index, uid in enumerate(manifest["train_uids"]):
                context = model.condition_encoder(conditions[index])
                status, levels, q = generate_tree(model, context, seed, steps=steps, sampler=sampler)
                path = output / "predictions" / f"seed-{seed}_{uid}.npz"
                np.savez_compressed(path, integer_vertices=q, status=np.asarray(status),
                                    **{f"depth{lv['depth']}_{k}": lv[k] for lv in levels
                                       for k in ("parents", "occupancy", "estimate", "predicted_cells")})
                generated.append({"seed": seed, "uid": uid, "index": index, "status": status,
                                  "levels": levels, "q": q,
                                  "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                write_json(output / "progress.json", {"generated": len(generated), "expected": 2 * len(conditions),
                                                      "minutes": (time.monotonic() - started) / 60})
    model.train(was_training)
    write_json(output / "generation_manifest.json",
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
    info = dict(info or {})
    write_json(output / "evaluation.json", {**info, **report, "rows": rows})
    summary = {**info, **report, "minutes": (time.monotonic() - started) / 60}
    write_json(output / "summary.json", summary)
    return summary
