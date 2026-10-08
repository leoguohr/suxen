"""Fixed input FPS reuse must preserve the joint training trajectory."""

import copy
from unittest.mock import patch

import pytest
import torch

from mini_nexus.data import VertexLevelBatch
from mini_nexus.training import VertexStageSystem
from mini_nexus.vertex import farthest_point_sample


@pytest.mark.parametrize("use_checkpoint", [False, True])
def test_fixed_fps_preserves_outputs_gradients_adam_and_rng(use_checkpoint):
    torch.manual_seed(37)
    baseline = VertexStageSystem(
        hidden_dim=24, condition_dim=32, condition_tokens=4,
        condition_heads=4, condition_layers=2, num_layers=2,
        num_heads=3, max_depth=3, use_checkpoint=use_checkpoint,
    ).train()
    with torch.no_grad():
        baseline.flow.output.weight.normal_(std=.05)
    cached = copy.deepcopy(baseline)
    optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-5, weight_decay=0., foreach=False)
                  for m in (baseline, cached)]
    points = torch.randn(2, 12, 6)
    point_mask = torch.ones(2, 12, dtype=torch.bool)
    point_mask[0, -3:] = False
    points[~point_mask] = float('nan')
    rng = torch.get_rng_state().clone()
    indices = farthest_point_sample(points[..., :3], 4, point_mask)
    assert torch.equal(rng, torch.get_rng_state())
    level = VertexLevelBatch(
        torch.randint(0, 2, (2, 3, 8)).float(), torch.zeros(2, 3, 4),
        torch.tensor([[[0, 0, 0], [1, 0, 1], [1, 1, 1]]] * 2),
        torch.full((2, 3), 2), torch.ones(2, 3, dtype=torch.bool),
    )
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for _ in range(8):
            noise, time = torch.randn_like(level.target), torch.rand(2)
            reference = baseline(points, level, noise=noise, time=time, condition_mask=point_mask)
            before = torch.get_rng_state().clone()
            with patch('mini_nexus.vertex.farthest_point_sample', side_effect=AssertionError('FPS repeated')):
                actual = cached(points, level, noise=noise, time=time,
                                condition_mask=point_mask, condition_fps_indices=indices)
                (actual / 8).backward()
            assert torch.equal(before, torch.get_rng_state())
            assert torch.equal(reference, actual)
            (reference / 8).backward()
        assert cached.condition_encoder.point_embedding.weight.grad.norm() > 0
        for left, right in zip(baseline.parameters(), cached.parameters()):
            assert torch.equal(left.grad, right.grad)
        for optimizer in optimizers:
            optimizer.step()
        for name, value in baseline.state_dict().items():
            assert torch.equal(value, cached.state_dict()[name])
        for key, state in optimizers[0].state_dict()['state'].items():
            for name, value in state.items():
                assert torch.equal(value, optimizers[1].state_dict()['state'][key][name])
