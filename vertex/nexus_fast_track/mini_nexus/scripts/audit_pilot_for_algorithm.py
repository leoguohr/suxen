#!/usr/bin/env python3
"""Audit Stage-2 pilot outputs before any algorithm training starts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data import load_stage2_sample, prepare_vertex_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=PROJECT_ROOT.parent / "data_pilot32" / "stage2_outputs",
    )
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "pilot_algorithm_audit.json"
    )
    args = parser.parse_args()

    records = []
    for sample_directory in sorted(path for path in args.stage2_root.iterdir() if path.is_dir()):
        record = {"uid": sample_directory.name, "depth": args.depth}
        try:
            sample = load_stage2_sample(sample_directory)
            cells, levels = prepare_vertex_stage(sample, args.depth)
            record.update(
                {
                    "status": "ready",
                    "vertex_count": len(sample.vertices),
                    "face_count": len(sample.faces),
                    "condition_point_count": len(sample.condition),
                    "leaf_count": len(cells),
                    "parents_per_depth": [len(level.parent_codes) for level in levels],
                }
            )
        except ValueError as error:
            record.update({"status": "quarantine", "reason": str(error)})
        records.append(record)

    report = {
        "stage2_root": str(args.stage2_root.resolve()),
        "depth": args.depth,
        "ready_count": sum(record["status"] == "ready" for record in records),
        "quarantine_count": sum(record["status"] == "quarantine" for record in records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

