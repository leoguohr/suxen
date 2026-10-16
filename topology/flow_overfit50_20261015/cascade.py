"""Gate 3, full pipeline: vertex-generated D9 cells -> topology flow -> frozen decoder -> edges/faces,
scored against the GT mesh. No GT vertex, edge or face enters generation.

Inputs per tree: the vertex chat's saved prediction `predictions/seed-<vertex seed>_<uid>.npz`
(`integer_vertices` = generated D9 cells, `status`). Cells become XYZ centres -1 + (q + 0.5) / 256, the
same frame as the topology cache (its GT vertices are exact D9 centres). The topology flow sees these
XYZ and the object's 8,192-point condition, samples latents from Gaussian noise (Euler, 20 steps), the
frozen OwnAE-v2 decoder turns them into embeddings, and edges/faces are recovered exactly as in gate 2.

Scoring (evaluation only, after generation): predicted vertices are matched to GT vertices.
  strict    a predicted vertex matches the GT vertex in the identical D9 cell
  one_cell  additionally matches leftover vertices within Chebyshev distance 1 cell
            (minimum total squared distance; the vertex chat's "within one cell" idea)
GT faces/edges are mapped to predicted indices; those touching an unmatched GT vertex can never be
predicted and count as misses; predicted faces/edges touching an unmatched predicted vertex are false.
A tree that did not complete counts every GT face and edge as missed.

  --mode cascade        vertex predictions from --vertex-eval
  --mode gt_reference   GT vertices in cache order with the gate-2 noise keys: must reproduce the
                        gate-2 evaluation of the same checkpoint exactly (self-check of this script)
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import micro_f1, write_json  # noqa: E402
from data import TopologyData, load_cache  # noqa: E402
from metrics import decode, default_cap, load_frozen_vae, score_mesh  # noqa: E402
from sampling import generate  # noqa: E402

D9 = 512


def cells_to_xyz(cells: np.ndarray) -> np.ndarray:
    return (-1.0 + (cells.astype(np.float64) + 0.5) * (2.0 / D9)).astype(np.float32)


def xyz_to_cells(vertices: np.ndarray) -> np.ndarray:
    cells = np.rint((vertices.astype(np.float64) + 1.0) * (D9 / 2.0) - 0.5).astype(np.int64)
    if len(vertices) and np.abs(cells_to_xyz(cells).astype(np.float64) - vertices).max() > 1e-6:
        raise ValueError("GT vertices are not D9 cell centres")
    return cells


def _keys(cells: np.ndarray) -> np.ndarray:
    return (cells[:, 0] * D9 + cells[:, 1]) * D9 + cells[:, 2]


def match_vertices(pred: np.ndarray, gt: np.ndarray, one_cell: bool, max_leftover: int = 2500) -> np.ndarray:
    """pred_to_gt [M], -1 where unmatched. Identical cells first; with one_cell, leftovers within
    Chebyshev distance 1 by minimum total squared distance (Hungarian; greedy if SciPy is missing)."""
    index = {int(k): j for j, k in enumerate(_keys(gt))}
    pred_to_gt = np.full(len(pred), -1, dtype=np.int64)
    used = set()
    for i, k in enumerate(_keys(pred)):
        j = index.get(int(k), -1)
        if j >= 0 and j not in used:
            pred_to_gt[i] = j
            used.add(j)
    if not one_cell:
        return pred_to_gt
    left_p = np.flatnonzero(pred_to_gt < 0)
    left_g = np.array([j for j in range(len(gt)) if j not in used], dtype=np.int64)
    if not len(left_p) or not len(left_g) or max(len(left_p), len(left_g)) > max_leftover:
        return pred_to_gt
    diff = pred[left_p][:, None, :] - gt[left_g][None, :, :]
    near = np.abs(diff).max(-1) <= 1
    cost = np.where(near, (diff.astype(np.float64) ** 2).sum(-1), 1e9)
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(cost)
        pairs = [(r, c) for r, c in zip(rows, cols) if near[r, c]]
    except ImportError:
        pairs, taken = [], set()
        for flat in np.argsort(cost, axis=None):
            r, c = divmod(int(flat), cost.shape[1])
            if not near[r, c]:
                break
            if r in {p for p, _ in pairs} or c in taken:
                continue
            pairs.append((r, c))
            taken.add(c)
    for r, c in pairs:
        pred_to_gt[left_p[r]] = left_g[c]
    return pred_to_gt


def map_gt_to_pred(pred_to_gt: np.ndarray, gt_count: int, gt_edges: np.ndarray, gt_faces: np.ndarray):
    """GT edges/faces whose vertices are all matched, re-indexed to predicted vertices (sorted, unique),
    plus the numbers of GT edges/faces that cannot be predicted at all."""
    gt_to_pred = np.full(gt_count, -1, dtype=np.int64)
    matched = np.flatnonzero(pred_to_gt >= 0)
    gt_to_pred[pred_to_gt[matched]] = matched

    def remap(items, width):
        mapped = gt_to_pred[items] if len(items) else np.zeros((0, width), dtype=np.int64)
        ok = (mapped >= 0).all(axis=1) if len(mapped) else np.zeros(0, dtype=bool)
        kept = np.sort(mapped[ok], axis=1)
        kept = np.unique(kept, axis=0) if len(kept) else np.zeros((0, width), dtype=np.int64)
        return kept, int((~ok).sum())

    edges, lost_edges = remap(gt_edges, 2)
    faces, lost_faces = remap(gt_faces, 3)
    return edges, faces, lost_edges, lost_faces


def score_tree(edge_emb, face_emb, pred_to_gt, gt_count, gt_edges, gt_faces) -> dict:
    edges, faces, lost_edges, lost_faces = map_gt_to_pred(pred_to_gt, gt_count, gt_edges, gt_faces)
    row = score_mesh(edge_emb, face_emb, edges, faces, default_cap(gt_faces))
    row["edge"]["fn"] += lost_edges
    row["face"]["fn"] += lost_faces
    row["face_fn_unmatched_vertex"] = lost_faces
    for task in ("edge", "face"):
        row[task]["micro_f1"] = micro_f1(row[task]["tp"], row[task]["fp"], row[task]["fn"])
    row["edge_strict"] = row["edge"]["fp"] == row["edge"]["fn"] == 0
    row["joint_strict"] = row["edge_strict"] and row["face"]["fp"] == row["face"]["fn"] == 0 and not row["overflow"]
    row["matched_vertices"] = int((pred_to_gt >= 0).sum())
    return row


def failed_tree(gt_edges, gt_faces) -> dict:
    row = dict(edge=dict(tp=0, fp=0, fn=len(gt_edges), tn=0), face=dict(tp=0, fp=0, fn=len(gt_faces), tn=0),
               candidates=0, overflow=False, face_fn_missing=len(gt_faces), face_fn_present=0,
               face_fn_unmatched_vertex=len(gt_faces), matched_vertices=0, edge_strict=False, joint_strict=False)
    for task in ("edge", "face"):
        row[task]["micro_f1"] = 0.0
    return row


def aggregate_trees(rows: list) -> dict:
    def block(group):
        out = dict(trees=len(group))
        for task in ("edge", "face"):
            sums = {k: sum(r[task][k] for r in group) for k in ("tp", "fp", "fn")}
            sums["micro_f1"] = micro_f1(sums["tp"], sums["fp"], sums["fn"])
            out[task] = sums
        out["joint_strict"] = sum(r["joint_strict"] for r in group)
        out["overflow_trees"] = sum(r["overflow"] for r in group)
        out["face_fn_unmatched_vertex"] = sum(r["face_fn_unmatched_vertex"] for r in group)
        return out
    result = dict(all=block(rows))
    result["exact_vertex_trees"] = block([r for r in rows if r["vertex_set_exact"]])
    result["inexact_vertex_trees"] = block([r for r in rows if not r["vertex_set_exact"]])
    for seed in sorted({r["vertex_seed"] for r in rows}):
        result[f"vertex_seed_{seed}"] = block([r for r in rows if r["vertex_seed"] == seed])
    return result


def predicted_records(cache: TopologyData, vertex_eval: Path, vertex_seed: int) -> tuple[list, list]:
    """Records for the trees that completed; plus a status list for all 50 objects."""
    records, status = [], []
    for i, uid in enumerate(cache.uids):
        path = vertex_eval / "predictions" / f"seed-{vertex_seed}_{uid}.npz"
        with np.load(path, allow_pickle=False) as item:
            cells = item["integer_vertices"].astype(np.int64).reshape(-1, 3)
            state = str(item["status"])
        if len(np.unique(cells, axis=0)) != len(cells):
            raise ValueError(f"duplicate generated cells: {path}")
        status.append(dict(uid=uid, status=state, predicted_n=len(cells)))
        if state != "complete" or not len(cells):
            continue
        src = cache.records[i]
        records.append(dict(uid=f"{uid}@vertex{vertex_seed}", source_index=i, cells=cells,
                            vertices=cells_to_xyz(cells), faces=np.zeros((0, 3), dtype=np.int64),
                            edges=np.zeros((0, 2), dtype=np.int64), points=src["points"],
                            point_mask=src["point_mask"], mu=np.zeros((len(cells), cache.mu.shape[1]), np.float32),
                            logvar=np.zeros((len(cells), cache.mu.shape[1]), np.float32)))
    return records, status


def check_vertex_gt(cache: TopologyData, vertex_data: Path) -> list:
    """Compare the topology GT cells with the vertex chat's GT D9 leaves (prepared_d9)."""
    manifest = json.loads((vertex_data / "manifest.json").read_text())
    by_uid = {r["uid"]: r for r in manifest["records"]}
    differences = []
    for i, uid in enumerate(cache.uids):
        with np.load(vertex_data / by_uid[uid]["label_file"], allow_pickle=False) as item:
            leaves = item["leaves"].astype(np.int64)
        topo = xyz_to_cells(cache.records[i]["vertices"])
        a, b = set(map(tuple, leaves.tolist())), set(map(tuple, topo.tolist()))
        if a != b:
            differences.append(dict(uid=uid, vertex_gt_cells=len(a), topology_gt_cells=len(b),
                                    only_in_vertex_gt=len(a - b), only_in_topology_gt=len(b - a)))
    return differences


@torch.no_grad()
def run_cascade(system, vae, cache, records, flow_seed, args) -> list:
    """Generate for all completed trees of one vertex seed, then decode and score each."""
    if not records:
        return []
    data = TopologyData(records, cache.mean, cache.std, cache.device)
    data.compute_fps(system.cfg.condition_tokens)
    latents = generate(system, data, seed=flow_seed, steps=args.steps, sampler=args.sampler, dtype=args.dtype,
                       attention=args.attention)
    rows = []
    for t, record in enumerate(records):
        i = record["source_index"]
        sl = data.token_slice(t)
        z_norm = latents[sl]
        edge_emb, face_emb = decode(vae, z_norm * cache.std + cache.mean)
        gt_cells = xyz_to_cells(cache.records[i]["vertices"])
        exact = (len(record["cells"]) == len(gt_cells)
                 and set(map(tuple, record["cells"].tolist())) == set(map(tuple, gt_cells.tolist())))
        out = dict(uid=cache.uids[i], predicted_n=len(record["cells"]), gt_n=len(gt_cells), vertex_set_exact=exact)
        for name, one_cell in (("strict", False), ("one_cell", True)):
            pred_to_gt = match_vertices(record["cells"], gt_cells, one_cell)
            row = score_tree(edge_emb, face_emb, pred_to_gt, len(gt_cells), cache.gt_edges[i], cache.gt_faces[i])
            matched = np.flatnonzero(pred_to_gt >= 0)
            if len(matched):
                gt_sl = cache.token_slice(i)
                ref = cache.mu_normalized[gt_sl][torch.from_numpy(pred_to_gt[matched]).to(cache.device)]
                row["latent_rmse_matched"] = float((z_norm[torch.from_numpy(matched).to(cache.device)] - ref)
                                                   .square().mean().sqrt())
            out[name] = row
        rows.append(out)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("cascade", "gt_reference"), default="cascade")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="topology flow final.pt")
    p.add_argument("--weights", choices=("raw", "ema"), default="raw")
    p.add_argument("--vertex-eval", type=Path, help="vertex evaluation folder holding predictions/")
    p.add_argument("--vertex-seeds", default="97029000,97029001")
    p.add_argument("--vertex-data", type=Path, help="optional: vertex chat's data/prepared_d9 for a GT cell check")
    p.add_argument("--flow-seeds", default="0")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--sampler", choices=("euler", "dpm2m"), default="euler")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--attention", choices=("auto", "flash", "sdpa"), default="auto")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.is_available(), "runs on one GPU"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    from evaluate import load_system
    cache = load_cache(args.cache, device)
    vae = load_frozen_vae(args.vae, device)
    system = load_system(args.checkpoint, args.weights, device)
    report = dict(mode=args.mode, checkpoint=str(args.checkpoint), weights=args.weights, steps=args.steps,
                  sampler=args.sampler, dtype=args.dtype, vertex_eval=str(args.vertex_eval), runs={})
    if args.vertex_data is not None:
        report["vertex_gt_cell_differences"] = check_vertex_gt(cache, args.vertex_data)
        print(f"[vertex GT check] objects whose GT cells differ: {report['vertex_gt_cell_differences']}", flush=True)
    flow_seeds = [int(s) for s in args.flow_seeds.split(",")]

    if args.mode == "gt_reference":
        for flow_seed in flow_seeds:
            records = [dict(r, source_index=i, cells=xyz_to_cells(r["vertices"])) for i, r in enumerate(cache.records)]
            rows = run_cascade(system, vae, cache, records, flow_seed, args)
            for row in rows:
                row["vertex_seed"] = "gt"
            name = f"gt_reference_flow{flow_seed}"
            summary = {m: aggregate_trees([dict(r[m], vertex_set_exact=r["vertex_set_exact"], vertex_seed="gt")
                                           for r in rows]) for m in ("strict", "one_cell")}
            report["runs"][name] = summary
            write_json(args.output / name / "summary.json", summary)
            with (args.output / name / "trees.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            f = summary["strict"]["all"]["face"]
            print(f"[{name}] face TP/FP/FN {f['tp']}/{f['fp']}/{f['fn']} micro-F1 {f['micro_f1']:.6f} "
                  f"(must equal the gate-2 evaluation of this checkpoint, seed {flow_seed})", flush=True)
    else:
        if args.vertex_eval is None:
            raise SystemExit("--vertex-eval is required for --mode cascade")
        for flow_seed in flow_seeds:
            all_rows, statuses = [], {}
            for vertex_seed in [int(s) for s in args.vertex_seeds.split(",")]:
                records, status = predicted_records(cache, args.vertex_eval, vertex_seed)
                statuses[vertex_seed] = status
                rows = run_cascade(system, vae, cache, records, flow_seed, args)
                done = {r["uid"] for r in rows}
                for i, uid in enumerate(cache.uids):
                    if uid not in done:  # vertex tree did not complete: everything missed
                        fail = failed_tree(cache.gt_edges[i], cache.gt_faces[i])
                        rows.append(dict(uid=uid, predicted_n=0, gt_n=len(cache.records[i]["vertices"]),
                                         vertex_set_exact=False, strict=fail, one_cell=dict(fail)))
                for row in rows:
                    row["vertex_seed"] = vertex_seed
                all_rows.extend(rows)
            name = f"cascade_flow{flow_seed}"
            summary = {m: aggregate_trees([dict(r[m], vertex_set_exact=r["vertex_set_exact"],
                                                vertex_seed=r["vertex_seed"]) for r in all_rows])
                       for m in ("strict", "one_cell")}
            summary["vertex_status"] = {str(k): v for k, v in statuses.items()}
            report["runs"][name] = summary
            write_json(args.output / name / "summary.json", summary)
            with (args.output / name / "trees.jsonl").open("w") as stream:
                for row in all_rows:
                    stream.write(json.dumps(row) + "\n")
            for m in ("strict", "one_cell"):
                s = summary[m]
                print(f"[{name} {m}] face micro-F1 all {s['all']['face']['micro_f1']:.6f} "
                      f"(exact-vertex trees {s['exact_vertex_trees']['trees']}: "
                      f"{s['exact_vertex_trees']['face']['micro_f1']:.6f}; inexact "
                      f"{s['inexact_vertex_trees']['trees']}: {s['inexact_vertex_trees']['face']['micro_f1']:.6f}) "
                      f"edge {s['all']['edge']['micro_f1']:.6f} strict {s['all']['joint_strict']}/"
                      f"{s['all']['trees']}", flush=True)
    report["minutes"] = (time.monotonic() - started) / 60
    write_json(args.output / "results.json", report)
    print("CASCADE_DONE", flush=True)


if __name__ == "__main__":
    main()
