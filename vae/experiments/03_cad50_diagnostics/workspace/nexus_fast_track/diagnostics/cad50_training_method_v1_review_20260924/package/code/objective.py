"""CAD50 face-finish objective -- standalone, network-agnostic.

Extracted verbatim (semantically) from the archived `data_objective.py`.
Only the objective / loss / classifier math lives here; nothing about the
network architecture.

Public API
----------
    edge_logits(embedding, pairs)             -> Tensor[P]
    face_logits(embedding, triples)           -> Tensor[T]
    hard4_sums(logits, labels)                -> (group_bce_sums[4], group_counts[4])
    hard4_chunks(logits_fn, embedding, ids, labels, chunk) -> (loss, stats)
    objective(outputs, item, negatives, ...)  -> (loss, stats)

Constants are exact; do not "clean them up".
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

# --------------------------------------------------------------------------
# Frozen constants (exact values from the archived source)
# --------------------------------------------------------------------------
EDGE_SCALE = 0.9306077080970389
FACE_SCALE = 0.39804385828730726
FACE_FACTOR = 0.25

GROUPS = ("tp", "tn", "fp", "fn")

# --------------------------------------------------------------------------
# Analytic classifiers  --  NOTE: these are FIXED formulas, not learned heads.
# --------------------------------------------------------------------------


def edge_logits(embedding: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Edge logit = EDGE_SCALE * (||delta||^2_space - ||delta||^2_time).

    `embedding` is [N, D] with D even; D is split into two halves
    (first half "space", second half "time").
    """
    delta = embedding[pairs[:, 0]] - embedding[pairs[:, 1]]
    space, time = delta.chunk(2, dim=-1)
    return (space.square().sum(-1) - time.square().sum(-1)) * EDGE_SCALE


def face_logits(embedding: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    """Face logit = FACE_SCALE * (Area_space^2 - Area_time^2).

    The inner quantity (|u|^2|v|^2 - (u.v)^2) is a Gram determinant, i.e.
    |u x v|^2 = (2*Area)^2, in ANY dimension (Lagrange identity).
    Multiplying by FACE_FACTOR=0.25 therefore yields the true Area^2.

    NOTE: there is NO bias term. The decision threshold is hard-pinned at 0.
    See FAILURE_ANALYSIS.md -- this is the single most impactful omission.
    """
    first, second, third = [embedding[triples[:, i]] for i in range(3)]
    areas = []
    for a, b, c in zip(first.chunk(2, -1), second.chunk(2, -1), third.chunk(2, -1)):
        u, v = b - a, c - a
        areas.append(
            (u.square().sum(-1) * v.square().sum(-1) - (u * v).sum(-1).square()).clamp_min(0)
        )
    return FACE_SCALE * (FACE_FACTOR * (areas[0] - areas[1]))


# --------------------------------------------------------------------------
# Hard4 -- 4-group BCE, mean-of-group-means
# --------------------------------------------------------------------------


def hard4_sums(logits: torch.Tensor, labels: torch.Tensor):
    """Split samples into TP/TN/FP/FN by (detached sign of logits) x label.

    Returns per-group summed BCE and per-group counts.
    Group assignment is a HARD assignment: gradients do NOT flow through the
    group boundary (logits are detached for the mask), but BCE itself is
    differentiable.
    """
    positive, truth = logits.detach() > 0, labels == 1
    masks = (
        truth & positive,    # tp : ground-truth & predicted positive
        ~truth & ~positive,  # tn
        ~truth & positive,   # fp
        truth & ~positive,   # fn
    )
    bce = F.binary_cross_entropy_with_logits(
        logits.float(), labels.to(torch.float32), reduction="none"
    )
    return (
        torch.stack([bce[m].sum(dtype=torch.float32) for m in masks]),
        torch.stack([m.sum() for m in masks]),
    )


def hard4_chunks(logits_fn, embedding, ids, labels, chunk):
    """Chunked Hard4 over a large candidate set.

    loss = mean over the 4 groups of (group BCE sum / group count)

    *** Each group is weighted 1/4 REGARDLESS OF GROUP SIZE. ***
    With fp=47 vs tn~8400 this amplifies a single false positive by ~179x
    relative to a single true negative. This is the root cause of the
    observed fp-down / fn-up boundary drift (see FAILURE_ANALYSIS.md).
    """
    numerator = embedding.new_zeros(4)
    counts = torch.zeros(4, dtype=torch.long, device=embedding.device)
    for start in range(0, len(ids), chunk):
        logits = logits_fn(embedding, ids[start:start + chunk])
        n, c = hard4_sums(logits, labels[start:start + chunk])
        numerator, counts = numerator + n, counts + c
    loss = (numerator / counts.clamp_min(1)).sum() / 4
    return loss, dict(
        counts=dict(zip(GROUPS, counts.detach().cpu().tolist())),
        group_bce_sums=numerator.detach().cpu().tolist(),
        loss=float(loss.detach()),
    )


# --------------------------------------------------------------------------
# Whole-mesh objective
# --------------------------------------------------------------------------


def objective(outputs, item, negatives, pair_chunk=32768, face_chunk=32768):
    """Whole-mesh loss = edge Hard4 + face Hard4.

    `item` needs: pairs, edge_labels, gt_faces.
    `negatives` is the merged negative triple set for this mesh.
    """
    device = outputs["edge"].device

    le, es = hard4_chunks(
        edge_logits, outputs["edge"], item["pairs"].to(device),
        item["edge_labels"].to(device), pair_chunk,
    )

    positive = item["gt_faces"].to(device)
    triples = torch.cat((positive, negatives.to(device)))
    labels = torch.cat((
        torch.ones(len(positive), device=device),
        torch.zeros(len(negatives), device=device),
    ))
    lf, fs = hard4_chunks(face_logits, outputs["face"], triples, labels, face_chunk)

    return le + lf, dict(
        edge=es, face=fs,
        edge_loss=float(le.detach()), face_loss=float(lf.detach()),
        face_positives=len(positive), face_negatives=len(negatives),
    )
