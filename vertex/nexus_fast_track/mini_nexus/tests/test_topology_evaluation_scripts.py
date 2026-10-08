from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scripts.evaluate_nexus2k_topology_ae as topology_evaluator
from scripts.audit_nexus2k_topology_recovery import (
    best_inclusive_threshold,
    threshold_mask,
)
from scripts.export_train_topology_comparison import (
    canonical_face_set,
    comparison_title,
)


def test_export_metrics_ignore_face_winding_and_cyclic_order():
    truth = torch.tensor([[0, 1, 2], [3, 4, 5]])
    oriented_prediction = torch.tensor([[2, 1, 0], [4, 5, 3]])

    assert canonical_face_set(oriented_prediction) == canonical_face_set(truth)


def test_export_title_reports_actual_checkpoint_step_and_thresholds():
    title = comparison_title(
        Path("/tmp/checkpoint-0000500.pt"),
        training_step=500,
        edge_threshold=0.125,
        face_threshold=-0.25,
    )

    assert "/tmp/checkpoint-0000500.pt" in title
    assert "step=500" in title
    assert "edge_threshold=0.125" in title
    assert "face_threshold=-0.25" in title
    assert "Final checkpoint" not in title
    assert "paper zero thresholds" not in title


def test_face_threshold_calibration_handles_no_candidate_cycles():
    threshold = best_inclusive_threshold(
        torch.empty(0), torch.empty(0, dtype=torch.bool), total_true_positives=7
    )

    assert threshold is None


def test_threshold_calibration_prefers_empty_prediction_when_candidates_are_negative():
    threshold = best_inclusive_threshold(
        torch.tensor([0.9, 0.2]),
        torch.tensor([False, False]),
        total_true_positives=3,
    )

    assert threshold is None


def test_threshold_calibration_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="must be finite"):
        best_inclusive_threshold(
            torch.tensor([0.9, float("nan")]),
            torch.tensor([True, False]),
            total_true_positives=1,
        )


def test_face_threshold_calibration_selects_attainable_inclusive_prefix():
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.tensor([True, False, True])

    threshold = best_inclusive_threshold(scores, labels, total_true_positives=2)

    assert threshold == pytest.approx(0.7)
    assert threshold_mask(scores, threshold, inclusive=True).tolist() == [
        True,
        True,
        True,
    ]


def test_threshold_calibration_only_selects_complete_tie_groups():
    scores = torch.tensor([0.9, 0.8, 0.8, 0.8, 0.8])
    labels = torch.tensor([True, True, False, False, False])

    threshold = best_inclusive_threshold(scores, labels, total_true_positives=2)

    assert threshold == pytest.approx(0.9)
    assert threshold_mask(scores, threshold, inclusive=True).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_paper_zero_threshold_remains_strict():
    scores = torch.tensor([-0.1, 0.0, 0.1])

    assert threshold_mask(scores, 0.0, inclusive=False).tolist() == [False, False, True]


def test_evaluate_checkpoint_appends_and_aggregates_one_sample_on_cpu(monkeypatch):
    class CpuValue:
        def __init__(self, value):
            self.value = value

        def cuda(self):
            return self.value

    class Dataset:
        manifest_sha256 = "evaluation-manifest"

        def __getitem__(self, index):
            assert index == 0
            return SimpleNamespace(
                uid="triangle",
                vertices=CpuValue(
                    torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
                ),
                faces=CpuValue(torch.tensor([[0, 1, 2]], dtype=torch.long)),
            )

    class Packed:
        def to(self, device):
            assert device == "cuda"
            return self

    class Model:
        edge_logit_scale = 1.0
        face_logit_scale = 1.0
        face_interval_factor = 1.0

        def topology_embedding_rows(self, _batch, *, sample_seeds):
            assert sample_seeds is None
            embedding = torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            )
            latent = torch.zeros((3, 2))
            return (latent,), (latent,), (embedding,), (embedding,)

        def scoring_contract(self):
            return {"test_scoring": True}

        def runtime_contract(self):
            return {"test_backend": "cpu_mock"}

    checkpoint = {
        "step": 17,
        "manifest_sha256": "evaluation-manifest",
        "args": {"precision": "fp32"},
        "runtime": {"training_system": "Nexus2KTopologyAESystem"},
    }
    monkeypatch.setattr(
        topology_evaluator, "load_topology_checkpoint", lambda _path: checkpoint
    )
    monkeypatch.setattr(
        topology_evaluator,
        "load_topology_system_from_checkpoint",
        lambda _checkpoint, *, device: Model(),
    )
    monkeypatch.setattr(
        topology_evaluator, "collate_packed_topology", lambda _samples: Packed()
    )

    result = topology_evaluator.evaluate_checkpoint(
        Path("checkpoint.pt"), Dataset(), [0]
    )

    assert result["training_step"] == 17
    assert result["manifest_matches_checkpoint"] is True
    assert result["edge"]["f1"] == 1.0
    assert result["face"]["f1"] == 1.0
    assert result["samples"] == [
        {
            "uid": "triangle",
            "vertices": 3,
            "faces": 1,
            "edge": result["edge"],
            "face": result["face"],
        }
    ]

    mismatched_dataset = Dataset()
    mismatched_dataset.manifest_sha256 = "different-manifest"
    with pytest.raises(ValueError, match="evaluation manifest does not match"):
        topology_evaluator.evaluate_checkpoint(
            Path("checkpoint.pt"), mismatched_dataset, [0]
        )
