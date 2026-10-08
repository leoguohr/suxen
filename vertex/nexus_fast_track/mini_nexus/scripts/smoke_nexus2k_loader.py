#!/usr/bin/env python3
"""Gate A/B smoke for the frozen Nexus2K manifest using scaled networks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mini_nexus.data_2k import Nexus2KBatch, Nexus2KManifestDataset, collate_nexus2k_samples
from mini_nexus.topology import (
    TopologyAutoencoder,
    all_vertex_pairs,
    first_order_interval,
    sample_face_triplets,
    second_order_interval,
)
from mini_nexus.training import TopologyFlowSystem, VertexStageSystem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-uid", required=True)
    parser.add_argument("--val-uid", required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_shapes(batch: Nexus2KBatch) -> dict[str, object]:
    return {
        "condition": list(batch.condition.shape),
        "vertices_padded": list(batch.vertices.shape),
        "vertex_mask": list(batch.vertex_mask.shape),
        "vertex_counts": [int(mask.sum()) for mask in batch.vertex_mask],
        "face_counts": [len(value) for value in batch.faces],
        "edge_counts": [value.shape[1] for value in batch.edge_index],
        "incidence_counts": [value.shape[1] for value in batch.incidence_index],
        "octree_parent_counts": [
            [int(level.mask[index].sum()) for level in batch.octree_levels]
            for index in range(len(batch.uids))
        ],
    }


def gradients_are_finite(module: torch.nn.Module) -> tuple[bool, int]:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(torch.isfinite(value).all() for value in gradients), len(gradients)


def stored_topology_loss(
    autoencoder: TopologyAutoencoder,
    batch: Nexus2KBatch,
    *,
    seed: int,
) -> torch.Tensor:
    """Use frozen dataset edge/face positives without changing the network."""

    losses = []
    for sample_index, (vertices_padded, mask, faces, true_edges, true_faces) in enumerate(
        zip(batch.vertices, batch.vertex_mask, batch.faces, batch.edge_index, batch.face_set)
    ):
        vertices = vertices_padded[mask]
        _, mu, log_variance, edge_embedding, face_embedding = autoencoder(vertices, faces)

        pairs = all_vertex_pairs(len(vertices), vertices.device)
        adjacency = torch.zeros(
            (len(vertices), len(vertices)), dtype=torch.bool, device=vertices.device
        )
        adjacency[true_edges[0], true_edges[1]] = True
        adjacency[true_edges[1], true_edges[0]] = True
        edge_labels = adjacency[pairs[:, 0], pairs[:, 1]].to(vertices.dtype)
        edge_logits = first_order_interval(
            edge_embedding[pairs[:, 0]], edge_embedding[pairs[:, 1]]
        )
        edge_loss = F.binary_cross_entropy_with_logits(edge_logits, edge_labels)

        generator = torch.Generator(device="cpu").manual_seed(seed + sample_index)
        triplets, face_labels = sample_face_triplets(
            true_faces,
            len(vertices),
            negative_ratio=1.0,
            generator=generator,
        )
        face_logits = second_order_interval(
            face_embedding[triplets[:, 0]],
            face_embedding[triplets[:, 1]],
            face_embedding[triplets[:, 2]],
        )
        face_loss = F.binary_cross_entropy_with_logits(
            face_logits, face_labels.to(vertices.dtype)
        )
        kl_loss = -0.5 * (
            1.0 + log_variance - mu.square() - log_variance.exp()
        ).mean()
        losses.append(edge_loss + face_loss + 1e-4 * kl_loss)
    return torch.stack(losses).mean()


def run_stage(name: str, module: torch.nn.Module, loss_fn, device: torch.device) -> dict[str, object]:
    module.to(device).train()
    module.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss = loss_fn()
    if not torch.isfinite(loss):
        raise RuntimeError(f"{name} loss is not finite")
    loss.backward()
    finite, gradient_tensors = gradients_are_finite(module)
    if not finite:
        raise RuntimeError(f"{name} gradients are missing or non-finite")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "loss": float(loss.detach().cpu()),
        "gradient_finite": finite,
        "gradient_tensor_count": gradient_tensors,
        "peak_memory_bytes": int(peak),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but CUDA is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = Nexus2KManifestDataset(args.manifest, "train")
    val_dataset = Nexus2KManifestDataset(args.manifest, "val")
    if len(train_dataset) != 1060 or len(val_dataset) != 118:
        raise RuntimeError("final manifest must contain train=1060 and val=118")
    train_sample = train_dataset[train_dataset.index_for_uid(args.train_uid)]
    val_sample = val_dataset[val_dataset.index_for_uid(args.val_uid)]
    batch_cpu = collate_nexus2k_samples([train_sample, val_sample])

    data_paths = [args.manifest]
    for sample in (train_sample, val_sample):
        data_paths.extend(Path(value) for _, value in sample.paths)
    hashes_before = {str(path): sha256(path) for path in data_paths}

    repository = Path(__file__).resolve().parents[1]
    network_paths = [
        repository / "mini_nexus" / "models.py",
        repository / "mini_nexus" / "vertex.py",
        repository / "mini_nexus" / "training.py",
        repository / "mini_nexus" / "topology.py",
        repository / "mini_nexus" / "flow.py",
        repository / "scripts" / "train_pilot.py",
    ]
    network_hashes_before = {str(path): sha256(path) for path in network_paths}
    batch = batch_cpu.to(device)

    vertex = VertexStageSystem(
        hidden_dim=24, condition_dim=24, condition_tokens=2, num_layers=1,
        num_heads=3, condition_heads=3, condition_layers=8, max_depth=9,
    )
    vertex_result = run_stage(
        "vertex",
        vertex,
        lambda: vertex(batch.condition, batch.octree_levels[-1]),
        device,
    )

    topology_autoencoder = TopologyAutoencoder(
        hidden_dim=24, latent_dim=8, spacetime_dim=8, num_heads=3, num_layers=1
    )
    topology_ae_result = run_stage(
        "topology_ae",
        topology_autoencoder,
        lambda: stored_topology_loss(topology_autoencoder, batch, seed=args.seed),
        device,
    )

    frozen_autoencoder = TopologyAutoencoder(
        hidden_dim=24, latent_dim=8, spacetime_dim=8, num_heads=3, num_layers=1
    )
    topology_flow = TopologyFlowSystem(
        frozen_autoencoder,
        hidden_dim=24,
        latent_dim=8,
        condition_tokens=2,
        num_layers=1,
        num_heads=3,
    )
    topology_flow_result = run_stage(
        "topology_flow", topology_flow, lambda: topology_flow(batch), device
    )

    hashes_after = {str(path): sha256(path) for path in data_paths}
    network_hashes_after = {str(path): sha256(path) for path in network_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("one or more input data files changed during smoke")
    if network_hashes_before != network_hashes_after:
        raise RuntimeError("one or more existing network files changed during smoke")

    report = {
        "status": "passed",
        "claim_boundary": "forward_backward_smoke_only_not_generation_quality",
        "independent_reimplementation": True,
        "vertex_architecture": "point_cloud_vertex_v1",
        "vertex_model_dimensions": "scaled_for_forward_backward_smoke",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "manifest": str(args.manifest),
        "manifest_sha256": train_dataset.manifest_sha256,
        "uids": list(batch.uids),
        "splits": list(batch.splits),
        "tensor_shapes": tensor_shapes(batch_cpu),
        "stages": {
            "vertex": vertex_result,
            "topology_ae": topology_ae_result,
            "topology_flow": topology_flow_result,
        },
        "data_hashes_before": hashes_before,
        "data_hashes_after": hashes_after,
        "data_hashes_unchanged": True,
        "network_hashes_before": network_hashes_before,
        "network_hashes_after": network_hashes_after,
        "existing_network_files_unchanged": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
