"""Shared, orientation-independent metrics for Topology-AE recovery scripts."""

from __future__ import annotations

import torch
from torch import Tensor


def canonical_face_set(faces: Tensor) -> set[tuple[int, ...]]:
    """Return orientation-independent triangle membership for metrics."""

    canonical = torch.sort(faces, dim=1).values
    return {tuple(value) for value in canonical.cpu().tolist()}


def _metrics_from_counts(
    *, predicted: int, truth: int, true_positive: int, false_positive: int, false_negative: int
) -> dict[str, float | int]:
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "predicted": predicted,
        "truth": truth,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
    }


def set_metrics(
    predicted: set[tuple[int, ...]], truth: set[tuple[int, ...]]
) -> dict[str, float | int]:
    """Compute precision, recall, and F1 from two edge or face sets."""

    true_positive = len(predicted & truth)
    false_positive = len(predicted - truth)
    false_negative = len(truth - predicted)
    return _metrics_from_counts(
        predicted=len(predicted),
        truth=len(truth),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def aggregate_set_metrics(
    rows: list[dict[str, float | int]],
) -> dict[str, float | int]:
    """Aggregate counts across meshes, then compute micro metrics once."""

    predicted = sum(int(row["predicted"]) for row in rows)
    truth = sum(int(row["truth"]) for row in rows)
    true_positive = sum(int(row["tp"]) for row in rows)
    false_positive = sum(int(row["fp"]) for row in rows)
    false_negative = sum(int(row["fn"]) for row in rows)
    return _metrics_from_counts(
        predicted=predicted,
        truth=truth,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
