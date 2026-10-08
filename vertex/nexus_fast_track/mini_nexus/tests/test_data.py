from pathlib import Path

import torch

from mini_nexus.data import (
    PilotSample,
    collate_octree_level,
    collate_pilot_samples,
    load_stage2_sample,
    prepare_vertex_stage,
)


PILOT_ROOT = Path(__file__).resolve().parents[2] / "data_pilot32" / "stage2_outputs"


def test_real_accepted_pilot_0032_reaches_depth9_octree():
    sample = load_stage2_sample(PILOT_ROOT / "nexus_pilot_0032")
    cells, levels = prepare_vertex_stage(sample, depth=9)
    assert sample.condition.shape == (8192, 6)
    assert len(cells) == len(sample.vertices) == 572
    assert len(levels) == 9


def test_variable_vertex_samples_collate_with_masks():
    first = PilotSample(
        uid="first",
        vertices=torch.tensor([[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]]),
        faces=torch.empty((0, 3), dtype=torch.long),
        condition=torch.zeros((8192, 6)),
        quality={},
    )
    second = PilotSample(
        uid="second",
        vertices=torch.tensor(
            [[-0.75, 0.75, -0.75], [0.75, -0.75, 0.75], [0.75, 0.75, -0.75]]
        ),
        faces=torch.tensor([[0, 1, 2]]),
        condition=torch.ones((8192, 6)),
        quality={},
    )
    batch = collate_pilot_samples([first, second])
    assert batch.vertices.shape == (2, 3, 3)
    assert batch.vertex_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.condition.shape == (2, 8192, 6)

    level = collate_octree_level(batch, current_depth=2, max_depth=3)
    assert level.target.shape[0] == 2
    assert level.target.shape[-1] == 8
    assert level.mask.sum(dim=1).tolist() == [2, 3]
    assert torch.equal(level.depths[level.mask], torch.full((5,), 2))
