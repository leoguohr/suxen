"""Behavioral checks for the point-conditioned octree Vertex model."""

import pytest
import torch

from mini_nexus.data import VertexLevelBatch
from mini_nexus.training import VertexStageSystem
from mini_nexus.vertex import (
    VertexConditionEncoder,
    VertexDiT,
    apply_vertex_rope,
    farthest_point_sample,
    parent_centers,
)


def small_encoder():
    return VertexConditionEncoder(
        hidden_dim=32, num_tokens=4, num_heads=4, num_layers=8
    )


def small_dit():
    return VertexDiT(
        hidden_dim=24, condition_dim=32, num_layers=2, num_heads=3, max_depth=3
    )


def activate_zero_initialized_branches(model):
    """Probe input dependencies after the intentionally zero initial output."""
    with torch.no_grad():
        torch.nn.init.normal_(model.output.weight, std=0.05)
        for block in model.blocks:
            torch.nn.init.normal_(block.modulation[-1].weight, std=0.05)
            torch.nn.init.normal_(block.modulation[-1].bias, std=0.05)


def dit_inputs():
    return (
        torch.randn(2, 3, 8),
        torch.tensor([0.2, 0.7]),
        torch.tensor([[[0, 0, 0], [1, 0, 1], [1, 1, 1]]] * 2),
        torch.tensor([2, 2]),
        torch.randn(2, 4, 32),
    )


def test_fps_uses_each_valid_index_once_even_for_duplicate_points():
    points = torch.zeros(2, 7, 3)
    points[:, -2:] = float("nan")
    mask = torch.tensor([[True] * 5 + [False] * 2] * 2)
    selected = farthest_point_sample(points, 5, mask)
    assert selected.shape == (2, 5)
    for indices in selected:
        assert set(indices.tolist()) == set(range(5))
    with pytest.raises(ValueError):
        farthest_point_sample(points, 1, torch.zeros_like(mask))


def test_condition_encoder_ignores_padding_and_point_order():
    torch.manual_seed(10)
    encoder = small_encoder().eval()
    points = torch.randn(2, 12, 6)
    expected = encoder(points)
    padded = torch.cat([points, torch.full((2, 3, 6), float("nan"))], dim=1)
    mask = torch.tensor([[True] * 12 + [False] * 3] * 2)
    torch.testing.assert_close(encoder(padded, mask), expected, atol=2e-6, rtol=2e-5)
    permutation = torch.randperm(12)
    torch.testing.assert_close(
        encoder(points[:, permutation]), expected, atol=2e-6, rtol=2e-5
    )
    changed_normals = points.clone()
    changed_normals[..., 3:] *= -1
    assert not torch.allclose(encoder(changed_normals), expected)
    with pytest.raises(ValueError):
        encoder(points, torch.zeros(2, 12, dtype=torch.bool))


def test_zero_output_initialization_then_joint_gradient_reaches_vecset():
    torch.manual_seed(11)
    encoder, model = small_encoder(), small_dit()
    noisy, time, codes, depths, _ = dit_inputs()
    points = torch.randn(2, 12, 6)
    target = torch.randn_like(noisy)
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(model.parameters()), lr=0.02
    )
    prediction = model(noisy, time, codes, depths, encoder(points))
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=0, rtol=0)
    (prediction - target).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(noisy, time, codes, depths, encoder(points))
    loss = (prediction - target).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum().item() for gradient in gradients) > 0


def test_dit_padding_and_batching_cannot_mix_objects():
    torch.manual_seed(12)
    model = small_dit().eval()
    activate_zero_initialized_branches(model)
    noisy, time, codes, depths, context = dit_inputs()
    mask = torch.tensor([[True, True, False], [True, True, True]])
    context_mask = torch.tensor([[True, True, False, False], [True] * 4])
    noisy[~mask] = float("nan")
    context[~context_mask] = float("nan")
    batched = model(noisy, time, codes, depths, context, mask, context_mask)
    assert torch.isfinite(batched).all()
    assert torch.count_nonzero(batched[~mask]) == 0
    for index in range(2):
        count, context_count = int(mask[index].sum()), int(context_mask[index].sum())
        single = model(
            noisy[index : index + 1, :count],
            time[index : index + 1],
            codes[index : index + 1, :count],
            depths[index : index + 1],
            context[index : index + 1, :context_count],
        )
        torch.testing.assert_close(
            batched[index : index + 1, :count], single, atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("changed_input", ["position", "time", "depth", "condition"])
def test_single_parent_uses_absolute_position_time_depth_and_condition(changed_input):
    torch.manual_seed(13)
    model = small_dit().eval()
    activate_zero_initialized_branches(model)
    inputs = [
        tensor[:1, :1] if index in (0, 2) else tensor[:1]
        for index, tensor in enumerate(dit_inputs())
    ]
    baseline = model(*inputs)
    if changed_input == "position":
        inputs[2] = inputs[2] + torch.tensor([1, 0, 0])
    elif changed_input == "time":
        inputs[1] = torch.tensor([0.8])
    elif changed_input == "depth":
        inputs[3] = torch.tensor([3])
    else:
        inputs[4] = torch.randn_like(inputs[4])
    assert not torch.allclose(model(*inputs), baseline, atol=1e-6, rtol=1e-5)


def test_parent_centers_use_target_depth_and_common_physical_coordinates():
    codes = torch.tensor([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [1, 1, 1]]])
    depths = torch.tensor([1, 2])
    expected = torch.tensor([[[0.0] * 3, [0.0] * 3], [[-0.5] * 3, [0.5] * 3]])
    torch.testing.assert_close(parent_centers(codes, depths), expected)
    torch.testing.assert_close(
        parent_centers(codes, depths[:, None].expand(-1, 2)), expected
    )


def test_rope_preserves_norm_tail_and_relative_attention():
    torch.manual_seed(14)
    query, key = torch.randn(2, 3, 5, 128), torch.randn(2, 3, 5, 128)
    positions = torch.randn(2, 5, 3) * 8
    rotated_query, rotated_key = apply_vertex_rope(query, key, positions)
    torch.testing.assert_close(rotated_query.norm(dim=-1), query.norm(dim=-1))
    torch.testing.assert_close(rotated_key.norm(dim=-1), key.norm(dim=-1))
    torch.testing.assert_close(rotated_query[..., -2:], query[..., -2:], atol=0, rtol=0)
    torch.testing.assert_close(rotated_key[..., -2:], key[..., -2:], atol=0, rtol=0)
    origin_query, origin_key = apply_vertex_rope(query, key, torch.zeros_like(positions))
    torch.testing.assert_close(origin_query, query, atol=0, rtol=0)
    torch.testing.assert_close(origin_key, key, atol=0, rtol=0)
    shifted_query, shifted_key = apply_vertex_rope(
        query, key, positions + torch.tensor([4.0, -6.0, 2.0])
    )
    torch.testing.assert_close(
        rotated_query @ rotated_key.transpose(-2, -1),
        shifted_query @ shifted_key.transpose(-2, -1),
        atol=3e-5,
        rtol=3e-5,
    )


def test_vertex_stage_loss_weights_objects_equally_and_ignores_nan_padding():
    torch.manual_seed(15)
    system = VertexStageSystem(
        hidden_dim=24,
        condition_dim=32,
        condition_tokens=4,
        condition_heads=4,
        condition_layers=8,
        num_layers=2,
        num_heads=3,
        max_depth=3,
    ).eval()
    activate_zero_initialized_branches(system.flow)
    condition = torch.randn(2, 12, 6)
    target = torch.randn(2, 3, 8)
    noise = torch.randn_like(target)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    target[~mask] = noise[~mask] = float("nan")
    _, time, codes, _, _ = dit_inputs()
    depths = torch.full((2, 3), 2)
    metadata = torch.zeros(2, 3, 4)
    level = VertexLevelBatch(target, metadata, codes, depths, mask)
    batched = system(condition, level, noise=noise, time=time)
    serial = []
    for index in range(2):
        count = int(mask[index].sum())
        single_level = VertexLevelBatch(
            target[index : index + 1, :count],
            metadata[index : index + 1, :count],
            codes[index : index + 1, :count],
            depths[index : index + 1, :count],
            mask[index : index + 1, :count],
        )
        serial.append(
            system(
                condition[index : index + 1],
                single_level,
                noise=noise[index : index + 1, :count],
                time=time[index : index + 1],
            )
        )
    assert torch.isfinite(batched)
    torch.testing.assert_close(batched, torch.stack(serial).mean())


def test_full_configuration_size_without_allocating_weights():
    with torch.device("meta"):
        encoder = VertexConditionEncoder()
        model = VertexDiT()
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert 400_000_000 < encoder_parameters < 410_000_000
    assert 1_925_000_000 < model_parameters < 1_950_000_000
    assert len(encoder.blocks) == 8
    assert len(model.blocks) == 36


def test_checkpointed_vertex_stage_matches_loss_and_gradients():
    torch.manual_seed(16)
    arguments = dict(
        hidden_dim=24,
        condition_dim=32,
        condition_tokens=4,
        condition_heads=4,
        condition_layers=8,
        num_layers=2,
        num_heads=3,
        max_depth=3,
    )
    ordinary = VertexStageSystem(**arguments, use_checkpoint=False)
    checkpointed = VertexStageSystem(**arguments, use_checkpoint=True)
    activate_zero_initialized_branches(ordinary.flow)
    checkpointed.load_state_dict(ordinary.state_dict())
    condition = torch.randn(2, 12, 6)
    target, time, codes, _, _ = dit_inputs()
    level = VertexLevelBatch(
        target,
        torch.zeros(2, 3, 4),
        codes,
        torch.full((2, 3), 2),
        torch.ones(2, 3, dtype=torch.bool),
    )
    noise = torch.randn_like(target)
    loss = ordinary(condition, level, noise=noise, time=time)
    checkpointed_loss = checkpointed(condition, level, noise=noise, time=time)
    loss.backward()
    checkpointed_loss.backward()
    torch.testing.assert_close(checkpointed_loss, loss)
    checkpointed_parameters = dict(checkpointed.named_parameters())
    for name, parameter in ordinary.named_parameters():
        other_gradient = checkpointed_parameters[name].grad
        if parameter.grad is None:
            assert other_gradient is None, name
        else:
            assert other_gradient is not None, name
            torch.testing.assert_close(
                other_gradient, parameter.grad, atol=2e-6, rtol=2e-5, msg=name
            )
