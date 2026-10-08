import torch

from mini_nexus.octree import (
    build_octree_levels,
    decode_leaf_centers,
    expand_occupied_children,
    quantize_vertices,
)


def test_octree_targets_and_round_trip():
    vertices = torch.tensor(
        [
            [-0.75, -0.75, -0.75],
            [0.75, -0.75, -0.75],
            [0.75, 0.75, 0.75],
        ]
    )
    depth = 3
    cells = quantize_vertices(vertices, depth)
    levels = build_octree_levels(cells, depth)
    assert len(levels) == depth
    assert levels[0].target.shape == (1, 8)
    assert int(levels[0].target.sum()) == 3

    parents = torch.zeros((1, 3), dtype=torch.long)
    for level in levels:
        assert torch.equal(parents, level.parent_codes)
        parents = expand_occupied_children(parents, level.target.bool())
        parents = torch.unique(parents, dim=0)
    assert torch.equal(parents, cells)

    centers = decode_leaf_centers(cells, depth)
    assert centers.shape == cells.shape
    assert torch.all((centers >= -1.0) & (centers <= 1.0))

