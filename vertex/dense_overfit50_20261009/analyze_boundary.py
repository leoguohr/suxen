"""CPU-only (numpy + scipy): how far off are the wrong depth-9 vertices? No GPU, no torch.

1. Tolerance curve. Per tree, match missed GT cells (FN) one-to-one to wrong predicted cells (FP) at
   depth 9. A tree is "exact within k cells" when the counts agree and a perfect matching exists using
   only pairs at Chebyshev distance <= k (k = 0 is the strict exact test).
2. Boundary test. For matched pairs one cell apart whose GT cell holds a single vertex, the depth-15
   label gives the vertex position inside its depth-9 cell (1/64 cell steps). We report how far the
   vertex is from the cell face the prediction crossed. Uniform positions would give 10% within 0.1
   cell and 25% within 0.25 cell. Much higher shares mean the misses are boundary near-misses.
3. Paper-style distances. Nexus reports Chamfer/Hausdorff distances, not exact vertex sets. Per tree we
   compute vertex-set Hausdorff and Chamfer (mean of the two directional mean nearest-neighbour
   distances) between predicted and GT depth-9 cell centres, in the paper's [-1, 1] units
   (one cell = 2/512 = 0.0039). These are vertex-set numbers, not the paper's surface metrics.

    python analyze_boundary.py --eval-dir <run>/evals/u003600_raw [--eval-dir ...] [--output boundary.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
OFFSETS = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.int64)
TOLERANCES = (0, 1, 2, 4, 8)
CELL = 2.0 / 512  # one depth-9 cell in the paper's [-1, 1] normalisation


def load_labels(data=HERE / "data"):
    """uid -> (d9 leaves [n,3], {d9 cell: [d15 positions]})."""
    manifest = json.loads((data / "prepared_d9" / "manifest.json").read_text())
    out = {}
    for record in manifest["records"]:
        with np.load(data / "prepared_d9" / record["label_file"], allow_pickle=False) as z:
            d9 = z["leaves"].copy()
        with np.load(data / record["source_label_file"], allow_pickle=False) as z:
            assert int(z["depth"]) == 15, "source labels are expected at depth 15"
            d15 = z["leaves"].copy()
        inside = {}
        for v in d15:
            inside.setdefault(tuple((v >> 6).tolist()), []).append(v)
        assert set(inside) == set(map(tuple, d9.tolist())), record["uid"]
        out[record["uid"]] = (d9, inside)
    return out


def predicted_d9(z):
    parents = z["depth9_parents"].reshape(-1, 3)
    occupancy = z["depth9_occupancy"].astype(bool).reshape(len(parents), 8)
    return (parents[:, None] * 2 + OFFSETS[None])[occupancy]


def chebyshev(a, b):
    return np.abs(a[:, None, :] - b[None, :, :]).max(-1)


def matchable(dist, k):
    cost = (dist > k).astype(np.int64)
    rows, cols = linear_sum_assignment(cost)
    return int(cost[rows, cols].sum()) == 0


def analyze(eval_dir: Path, labels: dict) -> dict:
    files = sorted((eval_dir / "predictions").glob("seed-*_*.npz"))
    assert files, f"no predictions under {eval_dir / 'predictions'}"
    within = dict.fromkeys(TOLERANCES, 0)
    trees = count_mismatch = incomplete = 0
    crossing, nearest_face, multi_vertex, multi_axis = [], 0, 0, 0
    hausdorff, chamfer = [], []
    for path in files:
        uid = path.stem.split("_", 1)[1]
        d9, inside = labels[uid]
        trees += 1
        with np.load(path, allow_pickle=False) as z:
            if "depth9_parents" not in z.files:
                incomplete += 1
                continue
            pred = predicted_d9(z)
        truth = set(map(tuple, d9.tolist()))
        predicted = set(map(tuple, pred.tolist()))
        if len(pred):
            to_gt = cKDTree(d9).query(pred)[0]
            to_pred = cKDTree(pred).query(d9)[0]
            hausdorff.append(max(to_gt.max(), to_pred.max()) * CELL)
            chamfer.append(0.5 * (to_gt.mean() + to_pred.mean()) * CELL)
        else:
            hausdorff.append(float("inf"))
            chamfer.append(float("inf"))
        fn = np.array(sorted(truth - predicted), dtype=np.int64).reshape(-1, 3)
        fp = np.array(sorted(predicted - truth), dtype=np.int64).reshape(-1, 3)
        if len(fn) != len(fp):
            count_mismatch += 1
        elif len(fn) == 0:
            for k in TOLERANCES:
                within[k] += 1
        else:
            dist = chebyshev(fn, fp)
            for k in TOLERANCES[1:]:
                within[k] += int(matchable(dist, k))
        if len(fn) and len(fp):
            dist = chebyshev(fn, fp)
            rows, cols = linear_sum_assignment(dist)
            for r, c in zip(rows, cols):
                if dist[r, c] != 1:
                    continue
                vertices = inside[tuple(fn[r].tolist())]
                if len(vertices) != 1:
                    multi_vertex += 1
                    continue
                offset = fp[c] - fn[r]
                frac = ((vertices[0] & 63) + 0.5) / 64.0
                moved = np.flatnonzero(offset)
                multi_axis += int(len(moved) > 1)
                gaps = [1.0 - frac[a] if offset[a] > 0 else frac[a] for a in moved]
                crossing.append(max(gaps))
                if len(moved) == 1:
                    sides = np.concatenate([frac, 1.0 - frac])
                    a = moved[0]
                    nearest_face += int(np.argmin(sides) == (a + 3 if offset[a] > 0 else a))
    crossing = np.array(crossing)
    shifted = len(crossing)
    hd, cd = np.array(hausdorff), np.array(chamfer)
    report = {
        "eval_dir": str(eval_dir), "trees": trees, "incomplete_trees": incomplete,
        "count_mismatch_trees": count_mismatch,
        "exact_within_cells": {str(k): within[k] for k in TOLERANCES},
        "shifted_pairs_single_vertex": shifted, "shifted_pairs_multi_vertex_cell": multi_vertex,
        "shifted_pairs_diagonal": multi_axis,
        "crossing_distance_share_le_0.1": float((crossing <= 0.1).mean()) if shifted else None,
        "crossing_distance_share_le_0.25": float((crossing <= 0.25).mean()) if shifted else None,
        "crossing_distance_median": float(np.median(crossing)) if shifted else None,
        "crossing_distance_histogram_0.1": np.histogram(crossing, bins=10, range=(0, 1))[0].tolist(),
        "crossed_nearest_face_share_single_axis": (nearest_face / (shifted - multi_axis)
                                                   if shifted - multi_axis else None),
        "vertex_hausdorff_median": float(np.median(hd)) if len(hd) else None,
        "vertex_hausdorff_max": float(hd.max()) if len(hd) else None,
        "vertex_hausdorff_trees_le_cells": {str(k): int((hd <= k * CELL * 1.0001).sum()) for k in (0, 1, 2, 4, 8, 16)},
        "vertex_chamfer_mean": float(cd[np.isfinite(cd)].mean()) if np.isfinite(cd).any() else None,
        "vertex_chamfer_max": float(cd.max()) if len(cd) else None,
    }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    labels = load_labels()
    reports = [analyze(d, labels) for d in args.eval_dir]
    for r in reports:
        w = r["exact_within_cells"]
        print(f"== {r['eval_dir']} ({r['trees']} trees, {r['count_mismatch_trees']} with wrong vertex count, "
              f"{r['incomplete_trees']} incomplete)")
        print("   exact within 0/1/2/4/8 depth-9 cells: " + " / ".join(str(w[str(k)]) for k in TOLERANCES))
        if r["shifted_pairs_single_vertex"]:
            print(f"   one-cell shifts: {r['shifted_pairs_single_vertex']} (diagonal {r['shifted_pairs_diagonal']}); "
                  f"vertex within 0.1 / 0.25 cell of the crossed face: {r['crossing_distance_share_le_0.1']:.0%} / "
                  f"{r['crossing_distance_share_le_0.25']:.0%} (uniform: 10% / 25%); median "
                  f"{r['crossing_distance_median']:.2f}; crossed the nearest face: "
                  f"{r['crossed_nearest_face_share_single_axis']:.0%} (random: 17%)")
            print(f"   histogram (0-1 cell, 0.1 bins): {r['crossing_distance_histogram_0.1']}")
        h = r["vertex_hausdorff_trees_le_cells"]
        print(f"   vertex-set Hausdorff ([-1,1] units, 1 cell = {CELL:.4f}): median {r['vertex_hausdorff_median']:.4f}, "
              f"max {r['vertex_hausdorff_max']:.4f}; trees within 0/1/2/4/8/16 cells: "
              + " / ".join(str(h[k]) for k in ("0", "1", "2", "4", "8", "16")))
        print(f"   vertex-set Chamfer: mean over trees {r['vertex_chamfer_mean']:.5f}, worst tree {r['vertex_chamfer_max']:.5f}")
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
