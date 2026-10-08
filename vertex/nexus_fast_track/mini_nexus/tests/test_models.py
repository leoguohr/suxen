import torch

from mini_nexus.flow import flow_matching_batch, flow_matching_loss
from mini_nexus.models import (
    ConditionalFlowTransformer,
    PointConditionEncoder,
    RotarySelfAttention,
    VecSetConditionEncoder,
    apply_3d_rope,
    farthest_point_sample,
)


def test_condition_and_flow_model_shapes_and_backward():
    torch.manual_seed(0)
    condition_encoder = PointConditionEncoder(hidden_dim=32, num_tokens=4, num_heads=4)
    model = ConditionalFlowTransformer(
        data_dim=8,
        metadata_dim=4,
        hidden_dim=32,
        condition_dim=32,
        num_layers=1,
        num_heads=4,
    )
    condition = torch.randn(2, 24, 6)
    clean = torch.randint(0, 2, (2, 5, 8)).float()
    metadata = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    noisy, target, time, _ = flow_matching_batch(clean)
    prediction = model(noisy, time, metadata, condition_encoder(condition), mask)
    loss = flow_matching_loss(prediction, target, mask)
    loss.backward()
    assert prediction.shape == clean.shape
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_vecset_uses_fps_anchors_and_condition_mask():
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [99.0, 0.0, 0.0]]]
    )
    mask = torch.tensor([[True, True, True, False]])
    indices = farthest_point_sample(points, 3, mask)
    assert set(indices[0].tolist()) == {0, 1, 2}

    encoder = VecSetConditionEncoder(
        input_dim=6, hidden_dim=24, num_tokens=3, num_heads=3
    )
    condition = torch.cat([points, torch.zeros_like(points)], dim=-1)
    tokens = encoder(condition, mask)
    assert tokens.shape == (1, 3, 24)


def test_3d_rope_preserves_norm_and_depends_on_position():
    torch.manual_seed(4)
    query = torch.randn(2, 3, 5, 8)
    key = torch.randn_like(query)
    zero_positions = torch.zeros(2, 5, 3)
    shifted_positions = torch.randn(2, 5, 3)
    zero_query, zero_key = apply_3d_rope(query, key, zero_positions)
    shifted_query, shifted_key = apply_3d_rope(query, key, shifted_positions)
    assert torch.allclose(zero_query, query)
    assert torch.allclose(zero_key, key)
    assert torch.allclose(shifted_query.norm(dim=-1), query.norm(dim=-1), atol=1e-5)
    assert torch.allclose(shifted_key.norm(dim=-1), key.norm(dim=-1), atol=1e-5)
    assert not torch.allclose(shifted_query, query)


def test_rotary_sdpa_ignores_padded_keys():
    torch.manual_seed(8)
    attention = RotarySelfAttention(hidden_dim=24, num_heads=3).eval()
    values = torch.randn(1, 4, 24)
    changed = values.clone()
    changed[:, -1] = 10_000.0
    positions = torch.randn(1, 4, 3)
    padding_mask = torch.tensor([[False, False, False, True]])

    original = attention(values, positions, padding_mask)
    updated = attention(changed, positions, padding_mask)
    assert torch.allclose(original[:, :3], updated[:, :3], atol=1e-5, rtol=1e-5)
