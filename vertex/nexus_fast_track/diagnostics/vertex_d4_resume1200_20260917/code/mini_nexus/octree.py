"""Octree targets for Nexus vertex generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class OctreeLevel:
    """Training data for p(O_depth | O_{depth-1}, condition)."""

    depth: int
    parent_codes: Tensor  # [N, 3], integer coordinates at depth-1
    target: Tensor  # [N, 8], multi-hot occupied child labels
    metadata: Tensor  # [N, 4], parent xyz center plus normalized depth


def quantize_vertex_cells(vertices: Tensor, depth: int) -> Tensor:
    """Map every normalized vertex in [-1,1] to a depth-D cell index."""

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V,3]")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not torch.isfinite(vertices).all():
        raise ValueError("vertices contain NaN or Inf")
    resolution = 1 << depth
    scaled = (vertices + 1.0) * 0.5 * resolution
    return torch.floor(scaled).clamp(0, resolution - 1).to(torch.long)


def quantize_vertices(vertices: Tensor, depth: int) -> Tensor:
    """Map vertices to unique depth-D cells after collision auditing."""

    return torch.unique(quantize_vertex_cells(vertices, depth), dim=0)


def decode_leaf_centers(cells: Tensor, depth: int) -> Tensor:
    """Convert depth-D integer cells back to their centers in [-1,1]."""

    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have shape [V,3]")
    resolution = 1 << depth
    if (cells < 0).any() or (cells >= resolution).any():
        raise ValueError("cell index outside the requested octree depth")
    return -1.0 + (cells.to(torch.float32) + 0.5) * (2.0 / resolution)


def _parent_metadata(parent_codes: Tensor, depth: int, max_depth: int) -> Tensor:
    parent_depth = depth - 1
    resolution = 1 << parent_depth
    centers = -1.0 + (parent_codes.to(torch.float32) + 0.5) * (2.0 / resolution)
    depth_column = torch.full(
        (len(parent_codes), 1),
        depth / max_depth,
        dtype=torch.float32,
        device=parent_codes.device,
    )
    return torch.cat([centers, depth_column], dim=-1)


def build_octree_levels(cells: Tensor, depth: int) -> list[OctreeLevel]:
    """Build the 8-way multi-hot label for every occupied parent at every depth."""

    if cells.ndim != 2 or cells.shape[1] != 3 or cells.dtype != torch.long:
        raise ValueError("cells must be an int64 tensor with shape [V,3]")
    resolution = 1 << depth
    if (cells < 0).any() or (cells >= resolution).any():
        raise ValueError("cell index outside the requested octree depth")

    levels: list[OctreeLevel] = []
    for current_depth in range(1, depth + 1):
        child_shift = depth - current_depth
        parent_shift = child_shift + 1
        parents_for_each_leaf = cells >> parent_shift
        parent_codes, inverse = torch.unique(
            parents_for_each_leaf, dim=0, return_inverse=True
        )

        child_bits = (cells >> child_shift) & 1
        child_ids = 4 * child_bits[:, 0] + 2 * child_bits[:, 1] + child_bits[:, 2]
        target = torch.zeros((len(parent_codes), 8), dtype=torch.float32)
        target[inverse, child_ids] = 1.0
        levels.append(
            OctreeLevel(
                depth=current_depth,
                parent_codes=parent_codes,
                target=target,
                metadata=_parent_metadata(parent_codes, current_depth, depth),
            )
        )
    return levels


def expand_occupied_children(parent_codes: Tensor, occupancy: Tensor) -> Tensor:
    """Expand selected 8-way children into integer codes at the next depth."""

    if occupancy.ndim != 2 or occupancy.shape != (len(parent_codes), 8):
        raise ValueError("occupancy must have shape [number_of_parents,8]")
    parent_ids, child_ids = torch.where(occupancy)
    if len(parent_ids) == 0:
        return torch.empty((0, 3), dtype=torch.long, device=parent_codes.device)
    bits = torch.stack(
        [(child_ids >> 2) & 1, (child_ids >> 1) & 1, child_ids & 1], dim=-1
    )
    return parent_codes[parent_ids] * 2 + bits


def metadata_for_parents(parent_codes: Tensor, depth: int, max_depth: int) -> Tensor:
    """Public wrapper used during coarse-to-fine inference."""

    return _parent_metadata(parent_codes, depth, max_depth).to(parent_codes.device)
