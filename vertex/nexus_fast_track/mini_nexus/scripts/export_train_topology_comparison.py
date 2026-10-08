#!/usr/bin/env python3
"""Export training-set ground truth beside final-checkpoint Topology AE recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset  # noqa: E402
from mini_nexus.packed_topology import collate_packed_topology  # noqa: E402
from mini_nexus.topology import (  # noqa: E402
    mesh_edges,
    orient_faces_consistently,
    recover_topology,
)
from mini_nexus.topology_evaluation import canonical_face_set, set_metrics  # noqa: E402
from mini_nexus.topology_checkpoint import (  # noqa: E402
    load_topology_checkpoint,
    load_topology_system_from_checkpoint,
    topology_runtime_resolution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uids", nargs="+", required=True)
    parser.add_argument("--edge-threshold", type=float, default=0.0)
    parser.add_argument("--face-threshold", type=float, default=0.0)
    return parser.parse_args()


def comparison_title(
    checkpoint: Path,
    training_step: int,
    edge_threshold: float,
    face_threshold: float,
) -> str:
    """Describe the exact checkpoint and threshold values shown in the figure."""

    return (
        "Topology Autoencoder — Training-set Reconstruction\n"
        f"checkpoint={checkpoint}; step={training_step}; "
        f"edge_threshold={edge_threshold:g}; face_threshold={face_threshold:g}"
    )


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(f"v {x:.8f} {y:.8f} {z:.8f}\n" for x, y, z in vertices)
        handle.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)


def draw_mesh(
    ax: object, vertices: np.ndarray, faces: np.ndarray, title: str, color: str
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces):
        collection = Poly3DCollection(
            vertices[faces],
            facecolor=color,
            edgecolor="#243447",
            linewidth=0.25,
            alpha=0.72,
        )
        ax.add_collection3d(collection)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=5, c=color)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=38)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


@torch.no_grad()
def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = load_topology_checkpoint(args.checkpoint)
    model = load_topology_system_from_checkpoint(checkpoint, device="cuda")
    scoring_contract = model.scoring_contract()
    dataset = Nexus2KManifestDataset(args.manifest, "train")
    if checkpoint.get("manifest_sha256") != dataset.manifest_sha256:
        raise ValueError("training-set export manifest does not match checkpoint")

    reports = []
    figure = plt.figure(figsize=(11, 4.4 * len(args.uids)), constrained_layout=True)
    for row_index, uid in enumerate(args.uids):
        sample = dataset[dataset.index_for_uid(uid)]
        vertices = sample.vertices.cuda()
        faces = sample.faces.cuda()
        packed = collate_packed_topology([sample]).to("cuda")
        _, _, edge_rows, face_rows = model.topology_embedding_rows(packed)
        edge_embedding, face_embedding = edge_rows[0], face_rows[0]

        predicted_edges_tensor, predicted_faces_tensor = recover_topology(
            edge_embedding,
            face_embedding,
            edge_threshold=args.edge_threshold,
            face_threshold=args.face_threshold,
            edge_logit_scale=model.edge_logit_scale,
            face_logit_scale=model.face_logit_scale,
            face_interval_factor=model.face_interval_factor,
        )
        predicted_faces_tensor = orient_faces_consistently(
            vertices, predicted_faces_tensor
        )

        truth_edges = {tuple(value) for value in mesh_edges(faces).cpu().tolist()}
        predicted_edges = {
            tuple(value) for value in predicted_edges_tensor.cpu().tolist()
        }
        truth_faces = canonical_face_set(faces)
        predicted_faces = canonical_face_set(predicted_faces_tensor)
        edge_metrics = set_metrics(predicted_edges, truth_edges)
        face_metrics = set_metrics(predicted_faces, truth_faces)

        vertices_np = vertices.cpu().numpy()
        truth_faces_np = faces.cpu().numpy()
        predicted_faces_np = predicted_faces_tensor.cpu().numpy()
        sample_dir = args.output / uid
        sample_dir.mkdir(exist_ok=True)
        write_obj(sample_dir / "ground_truth.obj", vertices_np, truth_faces_np)
        write_obj(sample_dir / "prediction.obj", vertices_np, predicted_faces_np)

        left = figure.add_subplot(len(args.uids), 2, 2 * row_index + 1, projection="3d")
        right = figure.add_subplot(
            len(args.uids), 2, 2 * row_index + 2, projection="3d"
        )
        draw_mesh(
            left,
            vertices_np,
            truth_faces_np,
            f"{uid} — Ground Truth ({len(truth_faces_np)} faces)",
            "#4C78A8",
        )
        draw_mesh(
            right,
            vertices_np,
            predicted_faces_np,
            f"Prediction ({len(predicted_faces_np)} faces, face F1={face_metrics['f1']:.3f})",
            "#F58518",
        )
        reports.append(
            {
                "uid": uid,
                "split": sample.split,
                "vertices": len(vertices),
                "edge": edge_metrics,
                "face": face_metrics,
            }
        )

    training_step = int(checkpoint.get("step", -1))
    figure.suptitle(
        comparison_title(
            args.checkpoint,
            training_step,
            args.edge_threshold,
            args.face_threshold,
        ),
        fontsize=14,
    )
    figure.savefig(
        args.output / "ground_truth_vs_prediction.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)
    payload = {
        "claim_boundary": "Topology AE training-set reconstruction, not end-to-end Nexus generation",
        "checkpoint": str(args.checkpoint),
        "training_step": training_step,
        "scoring_profile": checkpoint.get("scoring_profile", "legacy_unspecified"),
        "scoring_profile_status": "informational; scoring_contract validated separately",
        "scoring_contract": scoring_contract,
        "checkpoint_runtime": checkpoint.get("runtime", "legacy_unspecified"),
        "evaluation_runtime": model.runtime_contract(),
        "runtime_resolution": topology_runtime_resolution(checkpoint),
        "metric_aggregation": "per_sample_and_micro_counts_not_training_loss_weighting",
        "manifest": str(args.manifest),
        "checkpoint_manifest_sha256": checkpoint.get("manifest_sha256"),
        "evaluation_manifest_sha256": dataset.manifest_sha256,
        "manifest_matches_checkpoint": True,
        "edge_threshold": args.edge_threshold,
        "face_threshold": args.face_threshold,
        "threshold_comparator": ">",
        "samples": reports,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
