#!/usr/bin/env python3
"""GT oracle regression check. Never report these outputs as model generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_nexus.flow import flow_matching_batch
from mini_nexus.octree import decode_leaf_centers, expand_occupied_children, quantize_vertex_cells
from mini_nexus.vertex_evaluation import cell_metrics, generate_cells, sample_level


# Explicit XYZ order, independent of the production bit-shift implementation.
CHILD_OFFSETS = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                 (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


def reference_levels(cells, depth=9):
    """Derive parents and occupancies using integer division, not octree helpers."""
    result = []
    for d in range(1, depth + 1):
        table = {}
        for cell in cells.tolist():
            child = tuple(int(x) // (2 ** (depth - d)) for x in cell)
            parent = tuple(x // 2 for x in child)
            bits = tuple(x % 2 for x in child)
            table.setdefault(parent, [0.] * 8)[CHILD_OFFSETS.index(bits)] = 1.
        parents = sorted(table)
        result.append((torch.tensor(parents, dtype=torch.long),
                       torch.tensor([table[p] for p in parents], dtype=torch.float32)))
    return result


class GTOracle:
    """Test double implementing the network interface, with direct access to GT."""
    def __init__(self, cells, depth=9):
        self.max_depth = depth
        self.flow = self
        self.tables = [dict(zip(map(tuple, p.tolist()), y.tolist()))
                       for p, y in reference_levels(cells, depth)]

    def __call__(self, value, times, codes, depths, context):
        if not ((times >= 0) & (times < 1)).all():
            raise ValueError("oracle requires 0 <= t < 1")
        table = self.tables[int(depths[0]) - 1]
        target = torch.tensor([table.get(tuple(p), [0.] * 8) for p in codes[0].tolist()],
                              device=value.device, dtype=value.dtype).unsqueeze(0)
        return (target - value) / (1 - times[:, None, None])


def exact_cells(predicted, target):
    return predicted is not None and set(map(tuple, predicted.tolist())) == set(map(tuple, target.tolist()))


@torch.no_grad()
def check_cells(cells, *, name, depth=9, seeds=(17, 29, 43), steps_list=(1, 4, 20, 40)):
    cells = cells.cpu().long()
    if cells.ndim != 2 or cells.shape[1] != 3 or not len(cells):
        raise ValueError("nonempty [V,3] integer GT required")
    if (cells < 0).any() or (cells >= 2**depth).any() or len(torch.unique(cells, dim=0)) != len(cells):
        raise ValueError("GT must be unique and within the depth grid")
    oracle, context = GTOracle(cells, depth), torch.zeros(1, 1, 1)
    refs = reference_levels(cells, depth)
    error_max = 0.
    checks = 0
    # Exercise the actual training path and verify its derivative analytically.
    for d, (parents, target) in enumerate(refs, 1):
        for seed in seeds:
            noise = torch.randn((1, len(parents), 8), generator=torch.Generator().manual_seed(seed))
            for time in (0., .1, .5, .9):
                noisy, velocity, times, _ = flow_matching_batch(target[None], noise=noise, time=torch.tensor([time]))
                observed = oracle(noisy, times, parents[None], torch.tensor([d]), context)
                torch.testing.assert_close(observed, velocity, atol=3e-6, rtol=3e-6)
            for steps in steps_list:
                value = sample_level(oracle, context, parents, d, noise, steps=steps)
                error = float((value - target[None]).abs().max())
                error_max = max(error_max, error)
                assert error < 1e-5, (name, d, seed, steps, error)
                assert torch.equal(value[0] >= .5, target.bool())
                expanded = expand_occupied_children(parents, value[0] >= .5)
                expected = torch.unique(cells // (2 ** (depth - d)), dim=0)
                assert exact_cells(expanded, expected), (name, d, "child expansion")
                checks += 1
    generation = []
    for seed in seeds:
        for steps in steps_list:
            predicted, report = generate_cells(oracle, context, steps=steps, seed=seed,
                                                max_parents=max(10000, len(cells)))
            assert report["status"] == "complete" and exact_cells(predicted, cells), (name, seed, steps, report)
            centers = decode_leaf_centers(predicted, depth)
            expected_centers = -1. + (predicted.double() + .5) * (2. / 2**depth)
            torch.testing.assert_close(centers.double(), expected_centers, atol=0, rtol=0)
            assert torch.equal(quantize_vertex_cells(centers, depth), predicted)
            generation.append({"seed": seed, "steps": steps, "exact_coordinate_set": True,
                               **cell_metrics(predicted, cells), "levels": report["levels"]})
    return {"name": name, "passed": True, "oracle_only_not_model_score": True,
            "vertices": len(cells), "depth": depth, "level_sampler_checks": checks,
            "continuous_max_abs_error": error_max, "generation": generation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-directory", type=Path, required=True,
                        help="Directory of the saved generation NPZ files (reads target only).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    # This asymmetric shape distinguishes axis permutations even if a real mesh is symmetric.
    cases = [check_cells(torch.tensor([[0, 0, 0], [511, 511, 511], [1, 2, 4],
                                      [7, 33, 129], [401, 23, 257]]), name="asymmetric_axis_boundary_fixture")]
    for path in sorted(args.gt_directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            cases.append(check_cells(torch.from_numpy(data["target"].copy()), name=path.stem))
    if len(cases) == 1:
        raise ValueError("no saved GT files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"passed": True, "oracle_only_not_model_score": True,
        "path": "x_t=(1-t)*noise+t*GT; v_target=GT-noise; v_oracle=(GT-x_t)/(1-t)",
        "threshold": .5, "child_order_xyz": CHILD_OFFSETS, "cases": cases}, indent=2) + "\n")
    print(json.dumps({"passed": True, "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
