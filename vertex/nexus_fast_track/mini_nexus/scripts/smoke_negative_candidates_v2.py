#!/usr/bin/env python3
"""Read-only smoke for the audited topology-negative sidecars."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mini_nexus.negative_candidates import TopologyNegativeCandidateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--preset", default="mixed_medium")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_uids = {row["uid"]: row["split"] for row in rows}
    store = TopologyNegativeCandidateStore(
        args.candidate_root, args.preset, expected_uids=expected_uids
    )
    uid = store.protocol_uids("fixed_sampled_validation")[0]
    first = store.sample(
        uid,
        positive_edge_count=100,
        positive_face_count=100,
        seed=1,
        fixed=True,
    )
    second = store.sample(
        uid,
        positive_edge_count=100,
        positive_face_count=100,
        seed=999,
        fixed=True,
    )
    report = {
        **store.metadata(),
        "uid": uid,
        "edge_shape": list(first.edges.shape),
        "face_shape": list(first.faces.shape),
        "fixed_edges_seed_independent": first.edges.equal(second.edges),
        "fixed_faces_seed_independent": first.faces.equal(second.faces),
        "edges_globally_unique": len({tuple(row) for row in first.edges.tolist()})
        == len(first.edges),
        "faces_globally_unique": len({tuple(row) for row in first.faces.tolist()})
        == len(first.faces),
    }
    if not all(
        report[key]
        for key in (
            "fixed_edges_seed_independent",
            "fixed_faces_seed_independent",
            "edges_globally_unique",
            "faces_globally_unique",
        )
    ):
        raise RuntimeError("negative-candidate smoke failed")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
