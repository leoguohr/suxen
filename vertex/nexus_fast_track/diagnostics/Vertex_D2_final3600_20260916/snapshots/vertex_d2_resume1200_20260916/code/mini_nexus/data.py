"""Bridge from the existing Stage-2 pilot outputs to mini-Nexus tensors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .octree import OctreeLevel, build_octree_levels, quantize_vertex_cells


@dataclass(frozen=True)
class PilotSample:
    uid: str
    vertices: Tensor
    faces: Tensor
    condition: Tensor
    quality: dict


@dataclass(frozen=True)
class PilotBatch:
    uids: tuple[str, ...]
    vertices: Tensor
    vertex_mask: Tensor
    faces: tuple[Tensor, ...]
    condition: Tensor
    quality: tuple[dict, ...]

    def to(self, device: torch.device | str) -> "PilotBatch":
        return PilotBatch(
            uids=self.uids,
            vertices=self.vertices.to(device),
            vertex_mask=self.vertex_mask.to(device),
            faces=tuple(face.to(device) for face in self.faces),
            condition=self.condition.to(device),
            quality=self.quality,
        )


@dataclass(frozen=True)
class VertexLevelBatch:
    target: Tensor
    metadata: Tensor
    positions: Tensor
    depths: Tensor
    mask: Tensor

    def to(self, device: torch.device | str) -> "VertexLevelBatch":
        return VertexLevelBatch(
            target=self.target.to(device),
            metadata=self.metadata.to(device),
            positions=self.positions.to(device),
            depths=self.depths.to(device),
            mask=self.mask.to(device),
        )


def load_stage2_sample(sample_directory: Path) -> PilotSample:
    """Load one accepted Stage-2 sample without changing vertex identity."""

    quality_path = sample_directory / "quality.json"
    if not quality_path.exists():
        raise ValueError(f"missing quality.json: {sample_directory}")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if quality.get("status") != "accepted":
        reason_codes = quality.get("reason_codes", [])
        raise ValueError(f"upstream sample is not accepted: {reason_codes}")
    if quality.get("split") != "pilot":
        raise ValueError("algorithm pilot requires split: pilot")

    mesh_path = sample_directory / "mesh_normalized.npz"
    condition_path = sample_directory / "condition_point.npz"
    if not mesh_path.exists() or not condition_path.exists():
        raise ValueError("accepted sample is missing Stage-2 NPZ files")
    with np.load(mesh_path) as mesh:
        vertices = torch.from_numpy(mesh["vertices_norm"].astype(np.float32, copy=False))
        faces = torch.from_numpy(mesh["faces"].astype(np.int64, copy=False))
    with np.load(condition_path) as condition_data:
        points = torch.from_numpy(condition_data["points"].astype(np.float32, copy=False))
        normals = torch.from_numpy(condition_data["normals"].astype(np.float32, copy=False))

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_norm must have shape [V,3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F,3]")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("face index is outside vertices")
    if points.shape != (8192, 3) or normals.shape != (8192, 3):
        raise ValueError("point condition must contain 8192 points and normals")
    if not torch.isfinite(vertices).all() or not torch.isfinite(points).all():
        raise ValueError("non-finite geometry")

    return PilotSample(
        uid=quality["uid"],
        vertices=vertices,
        faces=faces,
        condition=torch.cat([points, normals], dim=-1),
        quality=quality,
    )


def prepare_vertex_stage(
    sample: PilotSample, depth: int = 9
) -> tuple[Tensor, list[OctreeLevel]]:
    """Quantize only when one canonical vertex maps to one depth-D cell."""

    cells_per_vertex = quantize_vertex_cells(sample.vertices, depth)
    unique_cells = torch.unique(cells_per_vertex, dim=0)
    collision_count = len(cells_per_vertex) - len(unique_cells)
    if collision_count:
        raise ValueError(
            f"quantization_collision: {collision_count} vertices share depth-{depth} cells"
        )
    return unique_cells, build_octree_levels(unique_cells, depth)


def load_ready_uids(audit_path: Path) -> list[str]:
    """Read the explicit algorithm audit; never infer readiness silently."""

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("algorithm audit is missing records")
    ready = [record["uid"] for record in records if record.get("status") == "ready"]
    if len(ready) != payload.get("ready_count"):
        raise ValueError("algorithm audit ready_count disagrees with records")
    return ready


class PilotStage2Dataset(Dataset[PilotSample]):
    """Lazy dataset over an explicitly audited list of Stage-2 UIDs."""

    def __init__(self, stage2_root: Path, uids: list[str]):
        if not uids:
            raise ValueError("training dataset has no algorithm-ready UID")
        if len(uids) != len(set(uids)):
            raise ValueError("training UID list contains duplicates")
        self.stage2_root = stage2_root
        self.uids = tuple(uids)

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, index: int) -> PilotSample:
        return load_stage2_sample(self.stage2_root / self.uids[index])


def collate_pilot_samples(samples: list[PilotSample]) -> PilotBatch:
    """Pad only vertices; faces retain per-object identity and condition stacks."""

    if not samples:
        raise ValueError("cannot collate an empty pilot batch")
    maximum_vertices = max(len(sample.vertices) for sample in samples)
    vertices = torch.zeros((len(samples), maximum_vertices, 3), dtype=torch.float32)
    vertex_mask = torch.zeros((len(samples), maximum_vertices), dtype=torch.bool)
    for batch_index, sample in enumerate(samples):
        count = len(sample.vertices)
        vertices[batch_index, :count] = sample.vertices
        vertex_mask[batch_index, :count] = True
    return PilotBatch(
        uids=tuple(sample.uid for sample in samples),
        vertices=vertices,
        vertex_mask=vertex_mask,
        faces=tuple(sample.faces for sample in samples),
        condition=torch.stack([sample.condition for sample in samples]),
        quality=tuple(sample.quality for sample in samples),
    )


def collate_octree_level(
    batch: PilotBatch, current_depth: int, max_depth: int = 9
) -> VertexLevelBatch:
    """Pad one common octree depth across objects for masked flow training."""

    if current_depth < 1 or current_depth > max_depth:
        raise ValueError("current_depth must be within 1..max_depth")
    levels: list[OctreeLevel] = []
    for vertices, mask in zip(batch.vertices, batch.vertex_mask):
        cells = quantize_vertex_cells(vertices[mask], max_depth)
        if len(torch.unique(cells, dim=0)) != len(cells):
            raise ValueError("algorithm-ready batch unexpectedly has a quantization collision")
        levels.append(build_octree_levels(cells, max_depth)[current_depth - 1])

    maximum_parents = max(len(level.parent_codes) for level in levels)
    target = torch.zeros((len(levels), maximum_parents, 8), dtype=torch.float32)
    metadata = torch.zeros((len(levels), maximum_parents, 4), dtype=torch.float32)
    positions = torch.zeros((len(levels), maximum_parents, 3), dtype=torch.float32)
    depths = torch.full(
        (len(levels), maximum_parents), current_depth, dtype=torch.long
    )
    level_mask = torch.zeros((len(levels), maximum_parents), dtype=torch.bool)
    for batch_index, level in enumerate(levels):
        count = len(level.parent_codes)
        target[batch_index, :count] = level.target
        metadata[batch_index, :count] = level.metadata
        positions[batch_index, :count] = level.parent_codes.to(torch.float32)
        level_mask[batch_index, :count] = True
    return VertexLevelBatch(target, metadata, positions, depths, level_mask)
