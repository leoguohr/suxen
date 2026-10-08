#!/usr/bin/env python3
"""Audit end-to-end edge-to-cycle-to-face recovery on small validation meshes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset  # noqa: E402
from mini_nexus.packed_topology import collate_packed_topology  # noqa: E402
from mini_nexus.topology import (  # noqa: E402
    all_vertex_pairs,
    edge_interval_logits,
    enumerate_edge_triangles,
    face_interval_logits,
    mesh_edges,
)
from mini_nexus.topology_evaluation import (  # noqa: E402
    aggregate_set_metrics,
    set_metrics,
)
from mini_nexus.topology_checkpoint import (  # noqa: E402
    load_topology_checkpoint,
    load_topology_system_from_checkpoint,
    topology_runtime_resolution,
)


DEFAULT_UIDS = (
    "nexus_2k_000028",
    "nexus_2k_000271",
    "nexus_2k_000799",
    "nexus_2k_001273",
    "nexus_2k_001460",
    "nexus_2k_001748",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uids", nargs="+", default=DEFAULT_UIDS)
    parser.add_argument("--allow-manifest-mismatch", action="store_true")
    return parser.parse_args()


def best_inclusive_threshold(
    scores: torch.Tensor, labels: torch.Tensor, total_true_positives: int
) -> float | None:
    """Return the best attainable ``score >= threshold`` F1 cutoff."""

    if not len(scores):
        return None
    if scores.shape != labels.shape:
        raise ValueError("threshold scores and labels must have equal shape")
    if not torch.isfinite(scores).all():
        raise ValueError("threshold calibration scores must be finite")
    positive_candidates = int(labels.sum())
    if total_true_positives < positive_candidates:
        raise ValueError("total true positives cannot be smaller than positive labels")
    if positive_candidates == 0:
        return None
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order].long()
    tp = sorted_labels.cumsum(0)
    fp = (~sorted_labels.bool()).long().cumsum(0)
    fn = total_true_positives - tp
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1)
    # Equal scores cannot be separated by one inclusive threshold. Only evaluate
    # the end of each tie group so the selected F1 is actually attainable.
    group_ends = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
    group_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    candidate_indices = torch.nonzero(group_ends, as_tuple=False).flatten()
    best_index = candidate_indices[torch.argmax(f1[candidate_indices])]
    return float(sorted_scores[best_index])


def threshold_mask(
    scores: torch.Tensor, threshold: float, *, inclusive: bool
) -> torch.Tensor:
    """Apply the declared audit comparator without changing paper zero semantics."""

    return scores >= threshold if inclusive else scores > threshold


@torch.no_grad()
def main() -> int:
    args = parse_args()
    checkpoint = load_topology_checkpoint(args.checkpoint)
    model = load_topology_system_from_checkpoint(checkpoint, device="cuda")
    scoring_contract = model.scoring_contract()
    dataset = Nexus2KManifestDataset(args.manifest, "val")
    checkpoint_manifest = checkpoint.get("manifest_sha256")
    if (
        checkpoint_manifest is not None
        and checkpoint_manifest != dataset.manifest_sha256
        and not args.allow_manifest_mismatch
    ):
        raise ValueError(
            "audit manifest does not match checkpoint; pass "
            "--allow-manifest-mismatch only for an intentional cross-manifest audit"
        )

    samples = []
    edge_scores = []
    edge_labels = []
    for uid in args.uids:
        sample = dataset[dataset.index_for_uid(uid)]
        vertices = sample.vertices.cuda()
        faces = sample.faces.cuda()
        packed = collate_packed_topology([sample]).to("cuda")
        _, _, edge_rows, face_rows = model.topology_embedding_rows(packed)
        edge_embedding, face_embedding = edge_rows[0], face_rows[0]
        pairs = all_vertex_pairs(len(vertices), vertices.device)
        scores = edge_interval_logits(
            edge_embedding[pairs[:, 0]],
            edge_embedding[pairs[:, 1]],
            logit_scale=model.edge_logit_scale,
        )
        true_edges = {tuple(value) for value in mesh_edges(faces).cpu().tolist()}
        labels = torch.tensor(
            [tuple(value) in true_edges for value in pairs.cpu().tolist()],
            dtype=torch.bool,
            device=vertices.device,
        )
        edge_scores.append(scores.cpu())
        edge_labels.append(labels.cpu())
        samples.append(
            {
                "uid": uid,
                "vertices": len(vertices),
                "faces": len(faces),
                "vertices_tensor": vertices,
                "faces_tensor": faces,
                "pairs": pairs,
                "edge_scores": scores,
                "edge_embedding": edge_embedding,
                "face_embedding": face_embedding,
                "true_edges": true_edges,
                "true_faces": {
                    tuple(value)
                    for value in torch.sort(faces, dim=1).values.cpu().tolist()
                },
            }
        )

    all_edge_scores = torch.cat(edge_scores)
    all_edge_labels = torch.cat(edge_labels)
    edge_threshold = best_inclusive_threshold(
        all_edge_scores, all_edge_labels, int(all_edge_labels.sum())
    )
    if edge_threshold is None:
        raise ValueError("edge threshold calibration requires at least one vertex pair")

    candidate_scores = []
    candidate_labels = []
    total_true_faces = 0
    for sample in samples:
        predicted_edges = sample["pairs"][
            threshold_mask(sample["edge_scores"], edge_threshold, inclusive=True)
        ]
        candidates = enumerate_edge_triangles(predicted_edges, int(sample["vertices"]))
        if len(candidates):
            scores = face_interval_logits(
                sample["face_embedding"][candidates[:, 0]],
                sample["face_embedding"][candidates[:, 1]],
                sample["face_embedding"][candidates[:, 2]],
                logit_scale=model.face_logit_scale,
                area_factor=model.face_interval_factor,
            )
            labels = torch.tensor(
                [
                    tuple(value) in sample["true_faces"]
                    for value in candidates.cpu().tolist()
                ],
                dtype=torch.bool,
                device=scores.device,
            )
            candidate_scores.append(scores.cpu())
            candidate_labels.append(labels.cpu())
        total_true_faces += len(sample["true_faces"])
    if candidate_scores:
        face_threshold = best_inclusive_threshold(
            torch.cat(candidate_scores), torch.cat(candidate_labels), total_true_faces
        )
        face_threshold_status = (
            "calibrated"
            if face_threshold is not None
            else "no_positive_candidate_cycles"
        )
    else:
        face_threshold = None
        face_threshold_status = "no_candidate_cycles"

    modes = {
        "paper_zero_threshold": (0.0, 0.0, False),
        "optimistic_audit_calibration": (edge_threshold, face_threshold, True),
    }
    mode_reports = {}
    for mode, (edge_cutoff, face_cutoff, inclusive) in modes.items():
        edge_rows = []
        face_rows = []
        sample_rows = []
        for sample in samples:
            predicted_edges_tensor = sample["pairs"][
                threshold_mask(sample["edge_scores"], edge_cutoff, inclusive=inclusive)
            ]
            predicted_edges = {
                tuple(value) for value in predicted_edges_tensor.cpu().tolist()
            }
            cycles = enumerate_edge_triangles(
                predicted_edges_tensor, int(sample["vertices"])
            )
            if len(cycles):
                scores = face_interval_logits(
                    sample["face_embedding"][cycles[:, 0]],
                    sample["face_embedding"][cycles[:, 1]],
                    sample["face_embedding"][cycles[:, 2]],
                    logit_scale=model.face_logit_scale,
                    area_factor=model.face_interval_factor,
                )
                predicted_faces = (
                    {
                        tuple(value)
                        for value in cycles[
                            threshold_mask(scores, face_cutoff, inclusive=inclusive)
                        ]
                        .cpu()
                        .tolist()
                    }
                    if face_cutoff is not None
                    else set()
                )
            else:
                predicted_faces = set()
            edge_result = set_metrics(predicted_edges, sample["true_edges"])
            face_result = set_metrics(predicted_faces, sample["true_faces"])
            edge_rows.append(edge_result)
            face_rows.append(face_result)
            sample_rows.append(
                {
                    "uid": sample["uid"],
                    "vertices": sample["vertices"],
                    "faces": sample["faces"],
                    "candidate_cycles": len(cycles),
                    "edge": edge_result,
                    "face": face_result,
                }
            )
        mode_reports[mode] = {
            "edge_threshold": edge_cutoff,
            "face_threshold": face_cutoff,
            "threshold_comparator": ">=" if inclusive else ">",
            "edge": aggregate_set_metrics(edge_rows),
            "face": aggregate_set_metrics(face_rows),
            "samples": sample_rows,
        }

    report = {
        "claim_boundary": "small validation audit; calibrated thresholds are fitted and evaluated on the same six meshes",
        "checkpoint": str(args.checkpoint),
        "training_step": int(checkpoint["step"]),
        "scoring_profile": checkpoint.get("scoring_profile", "legacy_unspecified"),
        "scoring_profile_status": "informational; scoring_contract validated separately",
        "scoring_contract": scoring_contract,
        "checkpoint_runtime": checkpoint.get("runtime", "legacy_unspecified"),
        "evaluation_runtime": model.runtime_contract(),
        "runtime_resolution": topology_runtime_resolution(checkpoint),
        "metric_aggregation": "micro_counts_across_meshes",
        "face_threshold_calibration_status": face_threshold_status,
        "checkpoint_manifest_sha256": checkpoint.get("manifest_sha256"),
        "evaluation_manifest_sha256": dataset.manifest_sha256,
        "manifest_matches_checkpoint": (
            checkpoint.get("manifest_sha256") == dataset.manifest_sha256
        ),
        "modes": mode_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
