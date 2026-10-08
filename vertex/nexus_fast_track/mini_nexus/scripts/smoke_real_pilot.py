#!/usr/bin/env python3
"""Run one real pilot sample through every trainable mini-Nexus component."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mini_nexus.data import load_stage2_sample, prepare_vertex_stage
from mini_nexus.flow import flow_matching_batch, flow_matching_loss
from mini_nexus.models import ConditionalFlowTransformer
from mini_nexus.topology import TopologyAutoencoder, topology_autoencoder_loss
from mini_nexus.vertex import VertexConditionEncoder, VertexDiT


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
    parser.add_argument("--uid", default="nexus_pilot_0032")
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=PROJECT_ROOT.parent / "data_pilot32" / "stage2_outputs",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "real_pilot_smoke.json"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    sample = load_stage2_sample(args.stage2_root / args.uid)
    cells, levels = prepare_vertex_stage(sample, args.depth)
    vertices = sample.vertices.to(device)
    faces = sample.faces.to(device)
    condition = sample.condition.to(device).unsqueeze(0)

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
    level = levels[-1]
    clean_occupancy = level.target.to(device).unsqueeze(0)
    parent_codes = level.parent_codes.to(device).unsqueeze(0)
    target_depth = torch.tensor([level.depth], dtype=torch.long, device=device)
    noisy, target_velocity, time, _ = flow_matching_batch(clean_occupancy)
    condition_tokens = condition_encoder(condition)
    predicted_velocity = vertex_flow(
        noisy, time, parent_codes, target_depth, condition_tokens
    )
    vertex_loss = flow_matching_loss(predicted_velocity, target_velocity)
    vertex_loss.backward()

    topology_autoencoder = TopologyAutoencoder(
        hidden_dim=64, latent_dim=16, spacetime_dim=16, num_heads=4, num_layers=2
    ).to(device)
    _, mu, log_variance, edge_embedding, face_embedding = topology_autoencoder(
        vertices, faces
    )
    topology_ae_loss, topology_parts = topology_autoencoder_loss(
        vertices,
        faces,
        mu,
        log_variance,
        edge_embedding,
        face_embedding,
        generator=torch.Generator().manual_seed(args.seed),
    )
    topology_ae_loss.backward()

    topology_flow = ConditionalFlowTransformer(
        data_dim=16,
        metadata_dim=3,
        hidden_dim=64,
        condition_dim=64,
        num_layers=2,
        num_heads=4,
    ).to(device)
    latent_target = mu.detach().unsqueeze(0)
    latent_noisy, latent_velocity, latent_time, _ = flow_matching_batch(latent_target)
    latent_prediction = topology_flow(
        latent_noisy,
        latent_time,
        vertices.unsqueeze(0),
        condition_tokens.detach(),
    )
    topology_flow_loss = flow_matching_loss(latent_prediction, latent_velocity)
    topology_flow_loss.backward()

    report = {
        "vertex_architecture": "point_cloud_vertex_v1",
        "vertex_sampler": "not_run_forward_backward_only",
        "uid": sample.uid,
        "split": sample.quality["split"],
        "device": str(device),
        "depth": args.depth,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "condition_shape": list(condition.shape),
        "leaf_count": len(cells),
        "parents_per_depth": [len(level.parent_codes) for level in levels],
        "vertex_flow_loss": float(vertex_loss.detach().cpu()),
        "topology_ae_loss": float(topology_ae_loss.detach().cpu()),
        "topology_ae_parts": {
            key: float(value.detach().cpu()) for key, value in topology_parts.items()
        },
        "topology_flow_loss": float(topology_flow_loss.detach().cpu()),
        "backward": "ok",
        "claim": "one-step real-data smoke test; not an overfit or quality result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
