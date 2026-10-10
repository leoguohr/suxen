"""CPU-only (numpy) error-cascade analysis on saved evaluation predictions. No GPU, no torch.

At every depth, every predicted parent is either CORRECT (a ground-truth cell) or EXTRA (wrong).
This script splits each depth's errors into:
  local FN      - GT children of correct parents that the model missed
  inherited FN  - GT cells whose parent was never generated (a deleted subtree)
  local FP      - wrong children predicted under correct parents
  inherited FP  - children predicted under extra parents (a spurious subtree)
It also reports whether extra parents die out (predict no children, i.e. self-correction) and
how large the deleted subtrees are at depth 9.

    python analyze_cascade.py --eval-dir <run>/eval_raw [--eval-dir ...] [--output cascade.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEPTH = 9
OFFSETS = np.array([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], dtype=np.int64)
RING26 = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
RING6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def as_set(cells) -> set:
    return set(map(tuple, np.asarray(cells).reshape(-1, 3).tolist()))


def load_leaves():
    manifest = json.loads((HERE / "data" / "prepared_d9" / "manifest.json").read_text())
    out = {}
    for record in manifest["records"]:
        with np.load(HERE / "data" / "prepared_d9" / record["label_file"], allow_pickle=False) as item:
            out[record["uid"]] = item["leaves"].copy()
    return out


def analyze(eval_dir: Path, leaves: dict) -> dict:
    files = sorted((eval_dir / "predictions").glob("seed-*_*.npz"))
    assert files, f"no predictions under {eval_dir / 'predictions'}"
    keys = ("local_fn", "inherited_fn", "local_fp", "inherited_fp", "correct_parents", "extra_parents",
            "extra_parents_with_no_children", "missing_parents", "first_loss_cells", "d9_vertices_under_first_loss",
            "missing_parents_next_to_predicted_6", "missing_parents_next_to_predicted_26",
            "predicted_parents", "ring6_candidates", "ring26_candidates",
            "local_fn_with_adjacent_fp", "local_fp_with_adjacent_fn", "local_fn_with_sibling_fp")
    per_depth = {d: dict.fromkeys(keys, 0) for d in range(1, DEPTH + 1)}
    trees = 0
    for path in files:
        uid = path.stem.split("_", 1)[1]
        cells = leaves[uid]
        truth = {d: as_set(np.unique(cells >> (DEPTH - d), axis=0)) for d in range(0, DEPTH + 1)}
        trees += 1
        with np.load(path, allow_pickle=False) as z:
            for d in range(1, DEPTH + 1):
                if f"depth{d}_parents" not in z.files:
                    break
                parents = z[f"depth{d}_parents"].reshape(-1, 3)
                occupancy = z[f"depth{d}_occupancy"].astype(bool).reshape(len(parents), 8)
                stats = per_depth[d]
                predicted_all, local_fn_cells, local_fp_cells = set(), [], []
                for row, parent in enumerate(map(tuple, parents.tolist())):
                    children = np.asarray(parent)[None] * 2 + OFFSETS
                    predicted = as_set(children[occupancy[row]])
                    predicted_all |= predicted
                    if parent in truth[d - 1]:
                        stats["correct_parents"] += 1
                        missed = (as_set(children) & truth[d]) - predicted
                        stats["local_fn"] += len(missed)
                        stats["local_fp"] += len(predicted - truth[d])
                        local_fn_cells += list(missed)
                        local_fp_cells += list(predicted - truth[d])
                        for cell in missed:  # first loss: everything below this cell is gone
                            stats["first_loss_cells"] += 1
                            under = (cells >> (DEPTH - d) == np.asarray(cell)).all(axis=1).sum()
                            stats["d9_vertices_under_first_loss"] += int(under)
                    else:
                        stats["extra_parents"] += 1
                        stats["inherited_fp"] += len(predicted)
                        stats["extra_parents_with_no_children"] += int(len(predicted) == 0)
                # "Shifted" errors: a missed cell with a wrong predicted cell right next to it (26-ring,
                # same depth), i.e. the vertex was put one cell off rather than lost.
                all_fp, all_fn = predicted_all - truth[d], truth[d] - predicted_all
                near = lambda c, pool: any((c[0] + a, c[1] + b, c[2] + e) in pool for a, b, e in RING26)
                stats["local_fn_with_adjacent_fp"] += sum(1 for c in local_fn_cells if near(c, all_fp))
                stats["local_fp_with_adjacent_fn"] += sum(1 for c in local_fp_cells if near(c, all_fn))
                stats["local_fn_with_sibling_fp"] += sum(
                    1 for c in local_fn_cells
                    if any((c[0] >> 1 << 1 | o[0], c[1] >> 1 << 1 | o[1], c[2] >> 1 << 1 | o[2]) in all_fp for o in OFFSETS))
                predicted_parents = as_set(parents)
                missing = truth[d - 1] - predicted_parents
                stats["missing_parents"] += len(missing)
                stats["predicted_parents"] += len(predicted_parents)
                # Could a "1-ring" candidate set (predicted cells plus their neighbours) bring back missing cells?
                for ring, key in ((RING6, "6"), (RING26, "26")):
                    stats[f"missing_parents_next_to_predicted_{key}"] += sum(
                        1 for m in missing if any((m[0] + a, m[1] + b, m[2] + c) in predicted_parents for a, b, c in ring))
                    limit = 1 << (d - 1)
                    neighbours = {(x + a, y + b, z + c) for x, y, z in predicted_parents for a, b, c in ring
                                  if 0 <= x + a < limit and 0 <= y + b < limit and 0 <= z + c < limit}
                    stats[f"ring{key}_candidates"] += len(neighbours - predicted_parents)
                stats["inherited_fn"] += sum(1 for c in truth[d] if (c[0] >> 1, c[1] >> 1, c[2] >> 1) in missing)
    for stats in per_depth.values():
        fn = stats["local_fn"] + stats["inherited_fn"]
        stats["inherited_fn_share"] = stats["inherited_fn"] / fn if fn else None
        stats["extra_parent_dieout_rate"] = (stats["extra_parents_with_no_children"] / stats["extra_parents"]
                                             if stats["extra_parents"] else None)
        stats["inherited_fp_per_extra_parent"] = (stats["inherited_fp"] / stats["extra_parents"]
                                                  if stats["extra_parents"] else None)
        stats["d9_vertices_lost_per_first_loss"] = (stats["d9_vertices_under_first_loss"] / stats["first_loss_cells"]
                                                    if stats["first_loss_cells"] else None)
        stats["shifted_fn_share"] = (stats["local_fn_with_adjacent_fp"] / stats["local_fn"]
                                     if stats["local_fn"] else None)
        stats["shifted_fp_share"] = (stats["local_fp_with_adjacent_fn"] / stats["local_fp"]
                                     if stats["local_fp"] else None)
        stats["sibling_fn_share"] = (stats["local_fn_with_sibling_fp"] / stats["local_fn"]
                                     if stats["local_fn"] else None)
        for key in ("6", "26"):
            stats[f"missing_recoverable_share_{key}"] = (stats[f"missing_parents_next_to_predicted_{key}"] / stats["missing_parents"]
                                                         if stats["missing_parents"] else None)
            stats[f"ring{key}_token_overhead"] = (stats[f"ring{key}_candidates"] / stats["predicted_parents"]
                                                  if stats["predicted_parents"] else None)
    return {"eval_dir": str(eval_dir), "trees": trees, "per_depth": per_depth}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    leaves = load_leaves()
    reports = [analyze(d, leaves) for d in args.eval_dir]
    for r in reports:
        print(f"== {r['eval_dir']} ({r['trees']} trees)")
        print(" d | local FN | inherited FN (share) | local FP | inherited FP | extra parents | die-out | FP/extra | D9 lost per local miss")
        for d, s in r["per_depth"].items():
            share = "-" if s["inherited_fn_share"] is None else f"{s['inherited_fn_share']:.0%}"
            die = "-" if s["extra_parent_dieout_rate"] is None else f"{s['extra_parent_dieout_rate']:.0%}"
            per = "-" if s["inherited_fp_per_extra_parent"] is None else f"{s['inherited_fp_per_extra_parent']:.2f}"
            lost = "-" if s["d9_vertices_lost_per_first_loss"] is None else f"{s['d9_vertices_lost_per_first_loss']:.1f}"
            print(f" {d} | {s['local_fn']:7d} | {s['inherited_fn']:7d} ({share:>4}) | {s['local_fp']:7d} | "
                  f"{s['inherited_fp']:7d} | {s['extra_parents']:7d} | {die:>5} | {per} | {lost}")
        pct = lambda v: "-" if v is None else f"{v:.0%}"
        print(" d | local FN with a wrong cell next to it (26-ring / same parent) | local FP with a missed cell next to it")
        for d, s in r["per_depth"].items():
            if s["local_fn"] or s["local_fp"]:
                print(f" {d} | {pct(s['shifted_fn_share'])} / {pct(s['sibling_fn_share'])} of {s['local_fn']} | "
                      f"{pct(s['shifted_fp_share'])} of {s['local_fp']}")
        print(" d | missing parents | next to a predicted cell (6-ring / 26-ring) | extra candidate tokens (6 / 26) per predicted cell")
        for d, s in r["per_depth"].items():
            if s["missing_parents"]:
                print(f" {d} | {s['missing_parents']:7d} | {s['missing_recoverable_share_6']:.0%} / {s['missing_recoverable_share_26']:.0%} | "
                      f"{s['ring6_token_overhead']:.2f} / {s['ring26_token_overhead']:.2f}")
    if args.output:
        args.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
