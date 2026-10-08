#!/usr/bin/env python3
"""Overfit all mini-Nexus components on one cube before touching the real pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.flow import euler_integrate, flow_matching_batch, flow_matching_loss
from mini_nexus.models import ConditionalFlowTransformer
from mini_nexus.octree import (
    build_octree_levels,
    decode_leaf_centers,
    expand_occupied_children,
    quantize_vertices,
)
from mini_nexus.topology import (
    TopologyAutoencoder,
    mesh_edges,
    recover_topology,
    topology_autoencoder_loss,
)
from mini_nexus.vertex import VertexConditionEncoder, VertexDiT


def cube_mesh(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    vertices = torch.tensor(
        [
            [-0.75, -0.75, -0.75],
            [-0.75, -0.75, 0.75],
            [-0.75, 0.75, -0.75],
            [-0.75, 0.75, 0.75],
            [0.75, -0.75, -0.75],
            [0.75, -0.75, 0.75],
            [0.75, 0.75, -0.75],
            [0.75, 0.75, 0.75],
        ],
        device=device,
    )
    faces = torch.tensor(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ],
        device=device,
    )
    return vertices, faces


def make_condition(vertices: torch.Tensor, repeats: int = 32) -> torch.Tensor:
    points = vertices.repeat_interleave(repeats, dim=0)
    normals = torch.nn.functional.normalize(points, dim=-1)
    return torch.cat([points, normals], dim=-1).unsqueeze(0)


def set_iou(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted_set = {tuple(row.tolist()) for row in predicted.cpu()}
    target_set = {tuple(row.tolist()) for row in target.cpu()}
    return len(predicted_set & target_set) / max(len(predicted_set | target_set), 1)


def f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted_set = {tuple(sorted(row.tolist())) for row in predicted.cpu()}
    target_set = {tuple(sorted(row.tolist())) for row in target.cpu()}
    true_positive = len(predicted_set & target_set)
    precision = true_positive / max(len(predicted_set), 1)
    recall = true_positive / max(len(target_set), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertex-steps", type=int, default=200)
    parser.add_argument("--topology-steps", type=int, default=400)
    parser.add_argument("--latent-flow-steps", type=int, default=200)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "toy")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    vertices, faces = cube_mesh(device)
    condition = make_condition(vertices)
    cells = quantize_vertices(vertices, args.depth).to(device)
    levels = build_octree_levels(cells.cpu(), args.depth)

    condition_encoder = VertexConditionEncoder(
        hidden_dim=64, num_tokens=8, num_heads=4, num_layers=8
    ).to(device)
    vertex_flow = VertexDiT(
        hidden_dim=64,
        condition_dim=64,
        num_layers=2,
        num_heads=4,
        max_depth=args.depth,
    ).to(device)
    vertex_optimizer = torch.optim.AdamW(
        list(condition_encoder.parameters()) + list(vertex_flow.parameters()), lr=2e-3
    )

    vertex_first_loss = None
    for step in range(args.vertex_steps):
        level = levels[step % len(levels)]
        clean = level.target.to(device).unsqueeze(0)
        parent_codes = level.parent_codes.to(device).unsqueeze(0)
        target_depth = torch.tensor([level.depth], dtype=torch.long, device=device)
        noisy, velocity, time, _ = flow_matching_batch(clean)
        prediction = vertex_flow(
            noisy, time, parent_codes, target_depth, condition_encoder(condition)
        )
        loss = flow_matching_loss(prediction, velocity)
        vertex_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        vertex_optimizer.step()
        if vertex_first_loss is None:
            vertex_first_loss = loss.item()

    condition_encoder.eval()
    vertex_flow.eval()
    with torch.no_grad():
        condition_tokens = condition_encoder(condition)
        parents = torch.zeros((1, 3), dtype=torch.long, device=device)
        for depth in range(1, args.depth + 1):
            parent_codes = parents.unsqueeze(0)
            target_depth = torch.tensor([depth], dtype=torch.long, device=device)
            noise = torch.randn((1, len(parents), 8), device=device)

            def vertex_velocity(x, time):
                return vertex_flow(x, time, parent_codes, target_depth, condition_tokens)

            scores = euler_integrate(vertex_velocity, noise, steps=args.ode_steps).squeeze(0)
            occupied = scores > 0.5
            missing = ~occupied.any(dim=1)
            if missing.any():
                occupied[missing, scores[missing].argmax(dim=1)] = True
            parents = torch.unique(expand_occupied_children(parents, occupied), dim=0)
        predicted_cells = parents
        predicted_vertices = decode_leaf_centers(predicted_cells.cpu(), args.depth).to(device)

    topology_autoencoder = TopologyAutoencoder(
        hidden_dim=64, latent_dim=16, spacetime_dim=16, num_heads=4, num_layers=2
    ).to(device)
    topology_optimizer = torch.optim.AdamW(topology_autoencoder.parameters(), lr=3e-3)
    topology_first_loss = None
    for _ in range(args.topology_steps):
        _, mu, log_variance, edge_embedding, face_embedding = topology_autoencoder(vertices, faces)
        topology_loss, _ = topology_autoencoder_loss(
            vertices, faces, mu, log_variance, edge_embedding, face_embedding
        )
        topology_optimizer.zero_grad(set_to_none=True)
        topology_loss.backward()
        topology_optimizer.step()
        if topology_first_loss is None:
            topology_first_loss = topology_loss.item()

    topology_autoencoder.eval()
    with torch.no_grad():
        target_latent, _ = topology_autoencoder.encode(vertices, faces)
        ae_edge_embedding, ae_face_embedding = topology_autoencoder.decode(target_latent)
        ae_edges, ae_faces = recover_topology(ae_edge_embedding, ae_face_embedding)

    topology_flow = ConditionalFlowTransformer(
        data_dim=16,
        metadata_dim=3,
        hidden_dim=64,
        condition_dim=64,
        num_layers=2,
        num_heads=4,
    ).to(device)
    latent_optimizer = torch.optim.AdamW(topology_flow.parameters(), lr=2e-3)
    topology_flow.train()
    latent_first_loss = None
    target_batch = target_latent.detach().unsqueeze(0)
    metadata = vertices.unsqueeze(0)
    condition_tokens = condition_encoder(condition).detach()
    for _ in range(args.latent_flow_steps):
        noisy, velocity, time, _ = flow_matching_batch(target_batch)
        prediction = topology_flow(noisy, time, metadata, condition_tokens)
        latent_loss = flow_matching_loss(prediction, velocity)
        latent_optimizer.zero_grad(set_to_none=True)
        latent_loss.backward()
        latent_optimizer.step()
        if latent_first_loss is None:
            latent_first_loss = latent_loss.item()

    topology_flow.eval()
    with torch.no_grad():
        latent_noise = torch.randn_like(target_batch)

        def latent_velocity(x, time):
            return topology_flow(x, time, metadata, condition_tokens)

        generated_latent = euler_integrate(
            latent_velocity, latent_noise, steps=args.ode_steps
        ).squeeze(0)
        generated_edge_embedding, generated_face_embedding = topology_autoencoder.decode(
            generated_latent
        )
        generated_edges, generated_faces = recover_topology(
            generated_edge_embedding, generated_face_embedding
        )

    export_faces = generated_faces if len(generated_faces) else ae_faces
    export_source = "topology_flow" if len(generated_faces) else "topology_autoencoder_fallback"
    mesh = trimesh.Trimesh(
        vertices=vertices.detach().cpu().numpy(),
        faces=export_faces.detach().cpu().numpy(),
        process=False,
    )
    mesh.export(args.output / "generated_cube.obj")

    report = {
        "vertex_architecture": "point_cloud_vertex_v1",
        "vertex_sampler": "euler_demo_not_paper_dpm_solver",
        "vertex_occupancy_postprocess": "threshold_0.5_and_at_least_one_child_per_parent_demo",
        "claim": "scaled toy pipeline; not a paper-scale reproduction or generation-quality result",
        "device": str(device),
        "seed": args.seed,
        "depth": args.depth,
        "vertex_loss_first": vertex_first_loss,
        "vertex_loss_last": loss.item(),
        "vertex_cell_iou": set_iou(predicted_cells, cells),
        "predicted_vertex_count": len(predicted_vertices),
        "target_vertex_count": len(vertices),
        "topology_ae_loss_first": topology_first_loss,
        "topology_ae_loss_last": topology_loss.item(),
        "topology_ae_edge_f1": f1(ae_edges, mesh_edges(faces)),
        "topology_ae_face_f1": f1(ae_faces, faces),
        "topology_flow_loss_first": latent_first_loss,
        "topology_flow_loss_last": latent_loss.item(),
        "topology_flow_edge_f1": f1(generated_edges, mesh_edges(faces)),
        "topology_flow_face_f1": f1(generated_faces, faces),
        "export_source": export_source,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(
        {
            "condition_encoder": condition_encoder.state_dict(),
            "vertex_flow": vertex_flow.state_dict(),
            "topology_autoencoder": topology_autoencoder.state_dict(),
            "topology_flow": topology_flow.state_dict(),
            "report": report,
        },
        args.output / "checkpoint.pt",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
