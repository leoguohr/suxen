#!/usr/bin/env python3
"""Evaluate three mini-Nexus stages on one real pilot sample.

This script deliberately reports vertex generation and topology generation
separately.  The current independent reimplementation has no learned mapping
from the unordered octree leaves to the canonical vertex identities used by
the topology stage, so it must not silently present the two outputs as a
verified end-to-end mesh.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data import load_stage2_sample, prepare_vertex_stage  # noqa: E402
from mini_nexus.flow import euler_integrate  # noqa: E402
from mini_nexus.octree import (  # noqa: E402
    decode_leaf_centers,
    expand_occupied_children,
)
from mini_nexus.topology import (  # noqa: E402
    TopologyAutoencoder,
    all_vertex_pairs,
    first_order_interval,
    mesh_edges,
    second_order_interval,
)
from mini_nexus.training import (  # noqa: E402
    TopologyAESystem,
    TopologyFlowSystem,
    VertexStageSystem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--vertex-checkpoint", type=Path, required=True)
    parser.add_argument("--topology-ae-checkpoint", type=Path, required=True)
    parser.add_argument("--topology-flow-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-generated-parents", type=int, default=4096)
    parser.add_argument("--max-recovery-edges", type=int, default=10000)
    parser.add_argument("--max-recovery-cycles", type=int, default=100000)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def checkpoint_dimensions(checkpoint: dict[str, Any]) -> dict[str, int]:
    arguments = checkpoint["args"]
    return {
        "hidden_dim": int(arguments["hidden_dim"]),
        "condition_tokens": int(arguments["condition_tokens"]),
        "num_layers": int(arguments["num_layers"]),
        "num_heads": int(arguments["num_heads"]),
        "latent_dim": int(arguments["latent_dim"]),
        "spacetime_dim": int(arguments["spacetime_dim"]),
        "max_depth": int(arguments["max_depth"]),
    }


def build_models(
    vertex_checkpoint: dict[str, Any],
    ae_checkpoint: dict[str, Any],
    flow_checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[VertexStageSystem, TopologyAESystem, TopologyFlowSystem, dict[str, int]]:
    vertex_args = vertex_checkpoint["args"]
    if vertex_args.get("vertex_architecture") != VertexStageSystem.architecture:
        raise ValueError(
            "Vertex checkpoint uses the old prototype or an unknown architecture; "
            "train a point_cloud_vertex_v1 checkpoint with the new Vertex entrypoint"
        )
    vertex_dimensions = checkpoint_dimensions(vertex_checkpoint)
    ae_dimensions = checkpoint_dimensions(ae_checkpoint)
    flow_dimensions = checkpoint_dimensions(flow_checkpoint)
    for name in ("hidden_dim", "num_heads", "latent_dim", "spacetime_dim"):
        if ae_dimensions[name] != flow_dimensions[name]:
            raise ValueError(f"topology checkpoint mismatch for {name}")

    vertex = VertexStageSystem(
        hidden_dim=vertex_dimensions["hidden_dim"],
        condition_tokens=vertex_dimensions["condition_tokens"],
        num_layers=vertex_dimensions["num_layers"],
        num_heads=vertex_dimensions["num_heads"],
        max_depth=vertex_dimensions["max_depth"],
        condition_dim=int(vertex_args["condition_dim"]),
        condition_heads=int(vertex_args["condition_heads"]),
        condition_layers=int(vertex_args["condition_layers"]),
    ).to(device)
    vertex.load_state_dict(vertex_checkpoint["model"], strict=True)

    ae = TopologyAESystem(
        hidden_dim=ae_dimensions["hidden_dim"],
        latent_dim=ae_dimensions["latent_dim"],
        spacetime_dim=ae_dimensions["spacetime_dim"],
        num_heads=ae_dimensions["num_heads"],
        num_layers=max(1, ae_dimensions["num_layers"] // 2),
    ).to(device)
    ae.load_state_dict(ae_checkpoint["model"], strict=True)

    frozen_autoencoder = TopologyAutoencoder(
        hidden_dim=flow_dimensions["hidden_dim"],
        latent_dim=flow_dimensions["latent_dim"],
        spacetime_dim=flow_dimensions["spacetime_dim"],
        num_heads=flow_dimensions["num_heads"],
        num_layers=max(1, flow_dimensions["num_layers"] // 2),
    )
    flow = TopologyFlowSystem(
        frozen_autoencoder,
        hidden_dim=flow_dimensions["hidden_dim"],
        latent_dim=flow_dimensions["latent_dim"],
        condition_tokens=flow_dimensions["condition_tokens"],
        num_layers=flow_dimensions["num_layers"],
        num_heads=flow_dimensions["num_heads"],
    ).to(device)
    flow.load_state_dict(flow_checkpoint["model"], strict=True)

    vertex.eval()
    ae.eval()
    flow.eval()
    return vertex, ae, flow, vertex_dimensions


def row_set(tensor: torch.Tensor) -> set[tuple[int, ...]]:
    return {tuple(int(value) for value in row) for row in tensor.detach().cpu().tolist()}


def set_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    predicted_set = row_set(predicted)
    target_set = row_set(target)
    true_positive = len(predicted_set & target_set)
    precision = true_positive / max(len(predicted_set), 1)
    recall = true_positive / max(len(target_set), 1)
    return {
        "predicted": len(predicted_set),
        "target": len(target_set),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "iou": true_positive / max(len(predicted_set | target_set), 1),
    }


def generate_vertices(
    model: VertexStageSystem,
    condition: torch.Tensor,
    target_cells: torch.Tensor,
    *,
    max_depth: int,
    ode_steps: int,
    threshold: float,
    max_generated_parents: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    device = condition.device
    condition_tokens = model.condition_encoder(condition)
    parents = torch.zeros((1, 3), dtype=torch.long, device=device)
    depth_metrics: list[dict[str, Any]] = []
    interventions: list[dict[str, Any]] = []

    for depth in range(1, max_depth + 1):
        positions = parents.unsqueeze(0)
        depths = torch.full(
            (1, len(parents)), depth, dtype=torch.long, device=device
        )
        mask = torch.ones((1, len(parents)), dtype=torch.bool, device=device)
        noise = torch.randn((1, len(parents), 8), device=device)

        def velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return model.flow(
                value,
                time,
                positions,
                depths,
                condition_tokens,
                mask,
            )

        scores = euler_integrate(velocity, noise, steps=ode_steps).squeeze(0)
        occupied = scores > threshold
        missing = ~occupied.any(dim=1)
        if missing.any():
            occupied[missing, scores[missing].argmax(dim=1)] = True
            interventions.append(
                {
                    "depth": depth,
                    "type": "ensure_one_child_per_parent",
                    "parent_count": int(missing.sum()),
                }
            )
        selected_count = int(occupied.sum())
        if selected_count > max_generated_parents:
            selected_scores = scores.masked_fill(~occupied, -torch.inf).flatten()
            keep = selected_scores.topk(max_generated_parents).indices
            occupied.zero_()
            occupied.view(-1)[keep] = True
            interventions.append(
                {
                    "depth": depth,
                    "type": "safety_topk_cap",
                    "before": selected_count,
                    "after": max_generated_parents,
                }
            )

        parents = torch.unique(expand_occupied_children(parents, occupied), dim=0)
        target_at_depth = torch.unique(
            target_cells.to(device) >> (max_depth - depth), dim=0
        )
        depth_metrics.append(
            {"depth": depth, **set_metrics(parents, target_at_depth)}
        )
    return parents, depth_metrics, interventions


def evaluate_teacher_forced_vertex(
    model: VertexStageSystem,
    condition: torch.Tensor,
    levels: list[Any],
    *,
    max_depth: int,
    ode_steps: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate each depth on its true parents to isolate cascading errors."""

    condition_tokens = model.condition_encoder(condition)
    results: list[dict[str, Any]] = []
    for level in levels:
        parents = level.parent_codes.to(condition.device)
        positions = parents.unsqueeze(0)
        depths = torch.full(
            (1, len(parents)), level.depth, dtype=torch.long, device=condition.device
        )
        mask = torch.ones((1, len(parents)), dtype=torch.bool, device=condition.device)
        noise = torch.randn((1, len(parents), 8), device=condition.device)

        def velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return model.flow(
                value,
                time,
                positions,
                depths,
                condition_tokens,
                mask,
            )

        scores = euler_integrate(velocity, noise, steps=ode_steps).squeeze(0)
        predicted = scores > threshold
        target = level.target.to(condition.device).bool()
        true_positive = int((predicted & target).sum())
        predicted_positive = int(predicted.sum())
        target_positive = int(target.sum())
        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(target_positive, 1)
        results.append(
            {
                "depth": level.depth,
                "parent_count": len(parents),
                "predicted_positive_children": predicted_positive,
                "target_positive_children": target_positive,
                "precision": precision,
                "recall": recall,
                "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
                "exact_parent_fraction": float((predicted == target).all(dim=1).float().mean()),
            }
        )
    return results


@torch.no_grad()
def recover_topology_safely(
    edge_embedding: torch.Tensor,
    face_embedding: torch.Tensor,
    *,
    max_edges: int,
    max_cycles: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    vertex_count = len(edge_embedding)
    pairs = all_vertex_pairs(vertex_count, edge_embedding.device)
    scores = first_order_interval(
        edge_embedding[pairs[:, 0]], edge_embedding[pairs[:, 1]]
    )
    edges = pairs[scores > 0.0]
    audit: dict[str, Any] = {
        "candidate_pairs": len(pairs),
        "positive_edges": len(edges),
        "status": "ok",
    }
    if len(edges) > max_edges:
        audit.update(
            status="blocked_dense_edge_graph",
            limit=max_edges,
            explanation="Face recovery skipped instead of changing the paper threshold.",
        )
        empty = torch.empty((0, 3), dtype=torch.long, device=edge_embedding.device)
        return edges, empty, audit

    neighbors = [set() for _ in range(vertex_count)]
    for left, right in edges.detach().cpu().tolist():
        neighbors[left].add(right)
        neighbors[right].add(left)
    candidates: list[tuple[int, int, int]] = []
    for left in range(vertex_count):
        for middle in (neighbor for neighbor in neighbors[left] if neighbor > left):
            for right in neighbors[left].intersection(neighbors[middle]):
                if right > middle:
                    candidates.append((left, middle, right))
                    if len(candidates) > max_cycles:
                        audit.update(
                            status="blocked_excessive_cycles",
                            observed_at_least=len(candidates),
                            limit=max_cycles,
                            explanation="Face recovery skipped instead of truncating 3-cycles.",
                        )
                        empty = torch.empty(
                            (0, 3), dtype=torch.long, device=edge_embedding.device
                        )
                        return edges, empty, audit
    audit["candidate_cycles"] = len(candidates)
    if not candidates:
        return edges, torch.empty(
            (0, 3), dtype=torch.long, device=edge_embedding.device
        ), audit
    triplets = torch.tensor(candidates, dtype=torch.long, device=edge_embedding.device)
    face_scores = second_order_interval(
        face_embedding[triplets[:, 0]],
        face_embedding[triplets[:, 1]],
        face_embedding[triplets[:, 2]],
    )
    faces = triplets[face_scores > 0.0]
    audit["positive_faces"] = len(faces)
    return edges, faces, audit


def canonical_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor.reshape(0, tensor.shape[-1]).to(torch.long)
    return torch.unique(torch.sort(tensor.to(torch.long), dim=-1).values, dim=0)


def topology_metrics(
    predicted_edges: torch.Tensor,
    predicted_faces: torch.Tensor,
    target_faces: torch.Tensor,
) -> dict[str, Any]:
    return {
        "edges": set_metrics(
            canonical_rows(predicted_edges), canonical_rows(mesh_edges(target_faces))
        ),
        "faces": set_metrics(
            canonical_rows(predicted_faces), canonical_rows(target_faces)
        ),
    }


def edge_score_diagnostics(
    edge_embedding: torch.Tensor, target_faces: torch.Tensor
) -> dict[str, Any]:
    """Measure ranking separately from the paper's fixed zero threshold."""

    pairs = all_vertex_pairs(len(edge_embedding), edge_embedding.device)
    scores = first_order_interval(
        edge_embedding[pairs[:, 0]], edge_embedding[pairs[:, 1]]
    )
    target_set = row_set(canonical_rows(mesh_edges(target_faces)))
    labels = torch.tensor(
        [tuple(pair) in target_set for pair in pairs.detach().cpu().tolist()],
        dtype=torch.bool,
        device=edge_embedding.device,
    )
    positive = scores[labels]
    negative = scores[~labels]
    top_k = scores.topk(len(positive)).indices
    top_k_true_positive = int(labels[top_k].sum())
    return {
        "purpose": "diagnostic_only; top-k uses the ground-truth edge count and is not generation",
        "class_imbalance_negative_per_positive": len(negative) / max(len(positive), 1),
        "positive_score": {
            "minimum": float(positive.min()),
            "mean": float(positive.mean()),
            "maximum": float(positive.max()),
        },
        "negative_score": {
            "minimum": float(negative.min()),
            "mean": float(negative.mean()),
            "maximum": float(negative.max()),
        },
        "positive_edges_above_zero": int((positive > 0.0).sum()),
        "negative_edges_above_zero": int((negative > 0.0).sum()),
        "top_k_equal_true_edge_count_f1": top_k_true_positive / max(len(positive), 1),
    }


def export_mesh(path: Path, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=faces.detach().cpu().numpy(),
        process=False,
    )
    mesh.export(path)


def draw_geometry(
    axis: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str,
) -> None:
    if len(faces):
        axis.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            linewidth=0.08,
            color="#5aa9e6",
            edgecolor="#17324d",
            alpha=0.9,
        )
    else:
        axis.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=2)
    axis.set_title(title)
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-1.05, 1.05)
    axis.set_zlim(-1.05, 1.05)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()


def render_comparison(
    path: Path,
    target_vertices: torch.Tensor,
    target_faces: torch.Tensor,
    predicted_vertices: torch.Tensor,
    ae_faces: torch.Tensor,
    flow_faces: torch.Tensor,
) -> None:
    vertices_np = target_vertices.detach().cpu().numpy()
    target_faces_np = target_faces.detach().cpu().numpy()
    figure = plt.figure(figsize=(16, 4.5), dpi=160)
    axes = [figure.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]
    draw_geometry(axes[0], vertices_np, target_faces_np, "Target mesh")
    draw_geometry(
        axes[1],
        predicted_vertices.detach().cpu().numpy(),
        np.empty((0, 3), dtype=np.int64),
        f"Vertex flow output: {len(predicted_vertices)} points\n(unordered leaf centers)",
    )
    draw_geometry(
        axes[2],
        vertices_np,
        ae_faces.detach().cpu().numpy(),
        f"Topology AE on GT vertices\n({len(ae_faces)} recovered faces)",
    )
    draw_geometry(
        axes[3],
        vertices_np,
        flow_faces.detach().cpu().numpy(),
        f"Topology flow on GT vertices\n({len(flow_faces)} recovered faces)",
    )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def final_loss(path: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "first": records[0]["loss"],
        "last": records[-1]["loss"],
        "step": records[-1]["step"],
    }


def main() -> int:
    args = parse_args()
    if args.ode_steps <= 0:
        raise ValueError("ode-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vertex_checkpoint = load_checkpoint(args.vertex_checkpoint, device)
    ae_checkpoint = load_checkpoint(args.topology_ae_checkpoint, device)
    flow_checkpoint = load_checkpoint(args.topology_flow_checkpoint, device)
    vertex_model, ae_model, flow_model, dimensions = build_models(
        vertex_checkpoint, ae_checkpoint, flow_checkpoint, device
    )

    sample = load_stage2_sample(args.sample)
    target_cells, target_levels = prepare_vertex_stage(sample, dimensions["max_depth"])
    vertices = sample.vertices.to(device)
    faces = sample.faces.to(device)
    condition = sample.condition.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_cells, depth_metrics, vertex_interventions = generate_vertices(
            vertex_model,
            condition,
            target_cells,
            max_depth=dimensions["max_depth"],
            ode_steps=args.ode_steps,
            threshold=args.threshold,
            max_generated_parents=args.max_generated_parents,
        )
        predicted_vertices = decode_leaf_centers(
            predicted_cells.detach().cpu(), dimensions["max_depth"]
        ).to(device)
        teacher_forced_metrics = evaluate_teacher_forced_vertex(
            vertex_model,
            condition,
            target_levels,
            max_depth=dimensions["max_depth"],
            ode_steps=args.ode_steps,
            threshold=args.threshold,
        )

        ae_mu, _ = ae_model.autoencoder.encode(vertices, faces)
        ae_edge_embedding, ae_face_embedding = ae_model.autoencoder.decode(ae_mu)
        ae_edges, ae_faces, ae_recovery = recover_topology_safely(
            ae_edge_embedding,
            ae_face_embedding,
            max_edges=args.max_recovery_edges,
            max_cycles=args.max_recovery_cycles,
        )

        condition_tokens = flow_model.condition_encoder(condition)
        latent_noise = torch.randn(
            (1, len(vertices), dimensions["latent_dim"]), device=device
        )
        vertex_mask = torch.ones((1, len(vertices)), dtype=torch.bool, device=device)

        def topology_velocity(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return flow_model.flow(
                value,
                time,
                vertices.unsqueeze(0),
                condition_tokens,
                vertex_mask,
                positions=vertices.unsqueeze(0),
            )

        generated_latent = euler_integrate(
            topology_velocity, latent_noise, steps=args.ode_steps
        ).squeeze(0)
        flow_edge_embedding, flow_face_embedding = flow_model.topology_autoencoder.decode(
            generated_latent
        )
        flow_edges, flow_faces, flow_recovery = recover_topology_safely(
            flow_edge_embedding,
            flow_face_embedding,
            max_edges=args.max_recovery_edges,
            max_cycles=args.max_recovery_cycles,
        )

    target_cells_device = target_cells.to(device)
    vertex_final_metrics = set_metrics(predicted_cells, target_cells_device)
    np.savez_compressed(
        args.output / "generated_vertices.npz",
        cells=predicted_cells.detach().cpu().numpy(),
        vertices=predicted_vertices.detach().cpu().numpy(),
    )
    trimesh.PointCloud(predicted_vertices.detach().cpu().numpy()).export(
        args.output / "generated_vertices.ply"
    )
    export_mesh(args.output / "target_mesh.obj", vertices, faces)
    if len(ae_faces):
        export_mesh(args.output / "topology_ae_gt_vertices.obj", vertices, ae_faces)
    if len(flow_faces):
        export_mesh(args.output / "topology_flow_gt_vertices.obj", vertices, flow_faces)
    render_comparison(
        args.output / "comparison.png",
        vertices,
        faces,
        predicted_vertices,
        ae_faces,
        flow_faces,
    )

    report = {
        "claim": "first real-pilot quality result for independent_reimplementation_v1",
        "uid": sample.uid,
        "device": str(device),
        "seed": args.seed,
        "ode_steps": args.ode_steps,
        "composition_status": "not_verified_end_to_end",
        "composition_blocker": (
            "The octree stage emits an unordered set of leaf centers, while topology "
            "training uses canonical vertex identities. No correspondence module is "
            "implemented, so topology is evaluated on ground-truth vertices."
        ),
        "training": {
            "vertex": final_loss(args.vertex_checkpoint.parent / "train.jsonl"),
            "topology_ae": final_loss(args.topology_ae_checkpoint.parent / "train.jsonl"),
            "topology_flow": final_loss(args.topology_flow_checkpoint.parent / "train.jsonl"),
        },
        "vertex_generation": {
            "architecture": VertexStageSystem.architecture,
            "sampler": "euler_validation_baseline_not_paper_dpm_solver",
            "final": vertex_final_metrics,
            "per_depth": depth_metrics,
            "teacher_forced_per_depth": teacher_forced_metrics,
            "interventions": vertex_interventions,
        },
        "topology_ae_on_ground_truth_vertices": {
            "metrics": topology_metrics(ae_edges, ae_faces, faces),
            "recovery": ae_recovery,
            "edge_score_diagnostics": edge_score_diagnostics(ae_edge_embedding, faces),
        },
        "topology_flow_on_ground_truth_vertices": {
            "metrics": topology_metrics(flow_edges, flow_faces, faces),
            "recovery": flow_recovery,
            "edge_score_diagnostics": edge_score_diagnostics(flow_edge_embedding, faces),
        },
        "files": {
            "comparison": "comparison.png",
            "target_mesh": "target_mesh.obj",
            "generated_vertex_cloud": "generated_vertices.ply",
            "topology_ae_mesh": (
                "topology_ae_gt_vertices.obj" if len(ae_faces) else None
            ),
            "topology_flow_mesh": (
                "topology_flow_gt_vertices.obj" if len(flow_faces) else None
            ),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
