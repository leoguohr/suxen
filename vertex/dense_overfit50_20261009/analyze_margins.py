"""CPU-only (numpy) margin check on saved evaluation predictions. No GPU, no torch.

For every tree that failed, take the FIRST depth whose cell set is wrong (its parents are
still exact there). For the wrong occupancy bits, report the continuous value the sampler
produced (thresholded at 0.5). Values close to 0.5 mean near-misses that a better sampler,
more steps or EMA weights might fix. Values near 0 or 1 mean the model was confidently wrong.
Correct bits that sit close to 0.5 are reported too, as bits "at risk".

    python analyze_margins.py --eval-dir <run>/eval_raw [--eval-dir <run2>/eval_raw ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPTH = 9
OFFSETS = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.int64)
BINS = np.linspace(0.0, 1.0, 11)


def load_leaves():
    manifest = json.loads((HERE / "data" / "prepared_d9" / "manifest.json").read_text())
    leaves = {}
    for record in manifest["records"]:
        with np.load(HERE / "data" / "prepared_d9" / record["label_file"], allow_pickle=False) as item:
            leaves[record["uid"]] = item["leaves"].copy()
    return leaves


def analyze(eval_dir: Path, leaves: dict) -> dict:
    files = sorted((eval_dir / "predictions").glob("seed-*_*.npz"))
    assert files, f"no predictions under {eval_dir / 'predictions'}"
    wrong_values, risky_correct, total_bits = [], 0, 0
    per_tree = []
    for path in files:
        seed, uid = path.stem.split("_", 1)
        cells = leaves[uid]
        with np.load(path, allow_pickle=False) as z:
            first = None
            for d in range(1, DEPTH + 1):
                key = f"depth{d}_predicted_cells"
                if key not in z.files:
                    first = d
                    break
                truth = set(map(tuple, np.unique(cells >> (DEPTH - d), axis=0)))
                if set(map(tuple, z[key])) != truth:
                    first = d
                    break
            if first is None:
                per_tree.append({"file": path.name, "exact": True})
                continue
            if f"depth{first}_parents" not in z.files:
                per_tree.append({"file": path.name, "exact": False, "first_bad_depth": first, "note": "level missing"})
                continue
            parents = z[f"depth{first}_parents"]
            estimate = z[f"depth{first}_estimate"].reshape(len(parents), 8)
            occupancy = z[f"depth{first}_occupancy"].astype(bool).reshape(len(parents), 8)
            truth = set(map(tuple, np.unique(cells >> (DEPTH - first), axis=0)))
            children = parents[:, None] * 2 + OFFSETS[None]
            actual = np.array([tuple(c) in truth for c in children.reshape(-1, 3)], dtype=bool).reshape(len(parents), 8)
            wrong = occupancy != actual
            missing_parents = len(truth - set(map(tuple, children.reshape(-1, 3))))
            values = estimate[wrong]
            wrong_values.extend(values.tolist())
            risky_correct += int((np.abs(estimate[~wrong] - 0.5) < 0.1).sum())
            total_bits += wrong.size
            per_tree.append({"file": path.name, "exact": False, "first_bad_depth": first, "parents": int(len(parents)),
                             "target_cells": len(cells), "wrong_bits": int(wrong.sum()),
                             "wrong_bit_values": [round(float(v), 4) for v in values[:20]],
                             "truth_cells_outside_children": missing_parents})
    wrong_values = np.asarray(wrong_values, dtype=np.float64)
    hist, _ = np.histogram(wrong_values, bins=BINS)
    failing = [t for t in per_tree if not t["exact"]]
    near = lambda m: int((np.abs(wrong_values - 0.5) < m).sum())  # noqa: E731
    return {
        "eval_dir": str(eval_dir), "trees": len(per_tree), "exact_trees": len(per_tree) - len(failing),
        "failing_trees": len(failing),
        "wrong_bits_at_first_bad_depth": int(len(wrong_values)),
        "wrong_bit_value_histogram_0_to_1_step_0.1": hist.tolist(),
        "wrong_bits_within_0.05_of_threshold": near(0.05),
        "wrong_bits_within_0.1_of_threshold": near(0.1),
        "wrong_bits_within_0.2_of_threshold": near(0.2),
        "correct_bits_within_0.1_of_threshold_at_those_depths": risky_correct,
        "bits_scored_at_those_depths": total_bits,
        "failing_trees_with_1_wrong_bit": sum(t.get("wrong_bits") == 1 for t in failing),
        "failing_trees_with_le_2_wrong_bits": sum((t.get("wrong_bits") or 99) <= 2 for t in failing),
        "per_tree": per_tree,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, default=None, help="optional JSON output path")
    args = p.parse_args()
    leaves = load_leaves()
    reports = [analyze(d, leaves) for d in args.eval_dir]
    for r in reports:
        print(f"{r['eval_dir']}: exact {r['exact_trees']}/{r['trees']}; wrong bits at first bad depth "
              f"{r['wrong_bits_at_first_bad_depth']} (within 0.05/0.1/0.2 of 0.5: "
              f"{r['wrong_bits_within_0.05_of_threshold']}/{r['wrong_bits_within_0.1_of_threshold']}/"
              f"{r['wrong_bits_within_0.2_of_threshold']}); histogram {r['wrong_bit_value_histogram_0_to_1_step_0.1']}; "
              f"failing trees with 1 / <=2 wrong bits: {r['failing_trees_with_1_wrong_bit']} / "
              f"{r['failing_trees_with_le_2_wrong_bits']}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
