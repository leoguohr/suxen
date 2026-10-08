import csv
import json
from pathlib import Path

import numpy as np
import torch

from mini_nexus.negative_candidates import TopologyNegativeCandidateStore


def _write_store(root: Path) -> TopologyNegativeCandidateStore:
    root.mkdir()
    sample = root / "sample.npz"
    np.savez_compressed(
        sample,
        edge_negative_knn=np.array([[0, 2], [0, 3], [1, 3], [1, 4]]),
        edge_negative_two_hop=np.array([[0, 2], [0, 4], [2, 4], [3, 4]]),
        edge_negative_uniform=np.array([[0, 4], [1, 3], [1, 4], [2, 4]]),
        face_negative_cycle=np.array([[0, 1, 3], [0, 2, 4]]),
        face_negative_wedge=np.array([[0, 1, 3], [1, 2, 4]]),
        face_negative_uniform=np.array([[0, 3, 4], [1, 3, 4]]),
    )
    with (root / "candidate_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "split", "candidate_path"])
        writer.writeheader()
        writer.writerow({"uid": "sample", "split": "val", "candidate_path": sample})
    (root / "presets.json").write_text(
        json.dumps(
            {
                "presets": {
                    "test": {
                        "edge_negative_ratio": 3,
                        "face_negative_ratio": 2,
                        "edge_mix": {"knn": 0.5, "two_hop": 0.25, "uniform": 0.25},
                        "face_mix": {"cycle": 0.5, "wedge": 0.25, "uniform": 0.25},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "evaluation_protocol.json").write_text(
        json.dumps({"fixed_sampled_validation": {"uids": ["sample"]}}),
        encoding="utf-8",
    )
    return TopologyNegativeCandidateStore(
        root, "test", expected_uids={"sample": "val"}
    )


def test_candidate_sampling_is_unique_and_fixed_validation_is_seed_independent(tmp_path):
    store = _write_store(tmp_path / "candidates")
    first = store.sample(
        "sample",
        positive_edge_count=2,
        positive_face_count=2,
        seed=10,
        fixed=True,
    )
    second = store.sample(
        "sample",
        positive_edge_count=2,
        positive_face_count=2,
        seed=999,
        fixed=True,
    )
    assert len(first.edges) == len({tuple(row) for row in first.edges.tolist()})
    assert len(first.faces) == len({tuple(row) for row in first.faces.tolist()})
    assert first.edges.equal(second.edges)
    assert first.faces.equal(second.faces)
    assert store.protocol_uids("fixed_sampled_validation") == ("sample",)


def test_candidate_training_sampling_is_reproducible_for_one_seed(tmp_path):
    store = _write_store(tmp_path / "candidates")
    first = store.sample(
        "sample",
        positive_edge_count=1,
        positive_face_count=1,
        seed=10,
        fixed=False,
    )
    second = store.sample(
        "sample",
        positive_edge_count=1,
        positive_face_count=1,
        seed=10,
        fixed=False,
    )
    assert first.edges.equal(second.edges)
    assert first.faces.equal(second.faces)


def test_face_only_sampling_preserves_faces_and_skips_unused_edges(tmp_path):
    store = _write_store(tmp_path / "candidates")
    full = store.sample(
        "sample",
        positive_edge_count=1,
        positive_face_count=1,
        seed=10,
        fixed=False,
    )
    face_only = store.sample(
        "sample",
        positive_edge_count=1,
        positive_face_count=1,
        seed=10,
        fixed=False,
        include_edges=False,
    )

    assert face_only.edges.shape == (0, 2)
    assert face_only.edge_source_counts == {}
    assert face_only.faces.equal(full.faces)
    assert face_only.face_source_counts == full.face_source_counts


def test_overfit_faces_include_all_false_cycles_and_fixed_sidecar_subsets(tmp_path):
    store = _write_store(tmp_path / "candidates")
    # K4 has four three-cycles.  One is a positive face, so the other three must
    # all be retained regardless of the sidecar cycle cap/order.
    positive_edges = torch.tensor(
        [[0, 0, 0, 1, 1, 2], [1, 2, 3, 2, 3, 3]], dtype=torch.long
    )
    positive_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    first = store.sample_fixed_overfit_faces(
        "sample",
        positive_edges=positive_edges,
        positive_faces=positive_faces,
        vertex_count=5,
        seed=7,
    )
    second = store.sample_fixed_overfit_faces(
        "sample",
        positive_edges=positive_edges,
        positive_faces=positive_faces,
        vertex_count=5,
        seed=7,
    )

    rows = {tuple(row) for row in first.faces.tolist()}
    assert {(0, 1, 3), (0, 2, 3), (1, 2, 3)}.issubset(rows)
    assert (0, 1, 2) not in rows
    assert first.face_source_counts == {"cycle": 3, "wedge": 1, "uniform": 1}
    assert first.faces.equal(second.faces)
