"""Small CPU candidate tests: no optimizer, checkpoint loading or GPU use."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from unittest.mock import patch

import torch
from torch import nn

import topology_flow
from topology_flow import (TopologyFlowConfig, PointCloudTopologyFlow,
                           _teacher_fourier_xyz, equal_mesh_velocity_loss)


def config(variant='c0'):
    return TopologyFlowConfig(hidden_dim=24, num_layers=2, num_heads=3,
        condition_dim=32, condition_layers=2, condition_heads=4,
        condition_tokens=4, rope_scale=256., variant=variant)


def seeded_model(variant):
    torch.manual_seed(0)
    model = PointCloudTopologyFlow(config(variant))
    return model, torch.get_rng_state().clone()


def activate(model):
    # Avoid vacuous comparisons caused by zero output and residual projections.
    with torch.no_grad():
        model.flow.output.weight.normal_(std=.04)
        for block in model.flow.blocks:
            block.modulation[-1].weight.normal_(std=.03)
            block.modulation[-1].bias.normal_(std=.03)
            block.cross_attention.output.weight.normal_(std=.03)


def inputs():
    torch.manual_seed(81)
    noisy = torch.randn(2, 7, 512)
    time = torch.tensor([.2, .8])
    vertices = torch.rand(2, 7, 3) * 2 - 1
    points = torch.randn(2, 12, 6)
    mask = torch.tensor([[True]*7, [True]*4+[False]*3])
    point_mask = torch.tensor([[True]*12, [True]*9+[False]*3])
    return noisy, time, vertices, points, mask, point_mask


def test_initialization_and_structure():
    c0, rng0 = seeded_model('c0')
    c1, rng1 = seeded_model('c1_fourier')
    c2, _ = seeded_model('c2_teacher')
    assert torch.equal(rng0, rng1)
    state0, state1 = c0.state_dict(), c1.state_dict()
    assert set(state1) - set(state0) == {'flow.position_embedding.weight', 'flow.position_embedding.bias'}
    for name, value in state0.items():
        torch.testing.assert_close(value, state1[name], atol=0, rtol=0)
    assert not c1.flow.position_embedding.weight.count_nonzero()
    assert not c1.flow.position_embedding.bias.count_nonzero()
    assert c2.flow.position_embedding.weight.count_nonzero()
    assert not c2.flow.position_embedding.bias.count_nonzero()
    for name, value in c0.condition_encoder.state_dict().items():
        torch.testing.assert_close(value, c2.condition_encoder.state_dict()[name], atol=0, rtol=0)
    for block in c2.flow.blocks:
        for norm in (block.self_norm, block.cross_norm, block.ffn_norm):
            assert norm.eps == 1e-5
        assert not block.self_norm.elementwise_affine and not block.ffn_norm.elementwise_affine
        assert block.cross_norm.elementwise_affine
        for attn in (block.self_attention, block.cross_attention):
            assert isinstance(attn.query_norm, nn.Identity) and isinstance(attn.key_norm, nn.Identity)
            assert all(layer.bias is not None for layer in attn.modules() if isinstance(layer, nn.Linear))
        assert block.ffn[1].approximate == 'none'
        assert block.ffn[0].out_features == 4*c2.cfg.hidden_dim
        assert block.modulation[-1].out_features == 6*c2.cfg.hidden_dim
        assert not block.modulation[-1].weight.count_nonzero()
        assert not block.cross_attention.output.weight.count_nonzero()
    assert c2.flow.output_norm.elementwise_affine and c2.flow.output_norm.eps == 1e-5
    assert c2.flow.time_embedding[0].in_features == 256
    assert c2.flow.output.out_features == 512
    counts = [sum(p.numel() for p in m.parameters()) for m in (c0, c1, c2)]
    width, depth = c0.cfg.hidden_dim, c0.cfg.num_layers
    assert counts[1]-counts[0] == 40*width
    assert counts[2]-counts[0] == 42*width-4*width*depth
    # Verify the exact teacher feature layout, including unscaled normalized XYZ.
    xyz = torch.tensor([[[-.8, .15, .6]]])
    angles = xyz[..., None] * 2.**torch.arange(6) * torch.pi
    expected = torch.cat((xyz, angles.sin().flatten(-2), angles.cos().flatten(-2)), -1)
    torch.testing.assert_close(_teacher_fourier_xyz(xyz), expected, atol=0, rtol=0)
    batch = inputs()
    for model in (c0, c1, c2):
        assert model(*batch).count_nonzero() == 0
    activate(c0)
    c1.load_state_dict(c0.state_dict(), strict=False)
    with torch.no_grad():
        baseline, candidate = c0(*batch), c1(*batch)
    assert baseline.abs().max() > .01
    torch.testing.assert_close(candidate, baseline, atol=0, rtol=0)
    # Position input is live once its new projection has learned nonzero values.
    with torch.no_grad():
        c1.flow.position_embedding.weight.normal_(std=.03)
        assert (c1(*batch)-baseline).abs().max() > 1e-5
    return counts


def test_permutation_and_mask(variant):
    model, _ = seeded_model(variant)
    activate(model)
    if variant == 'c1_fourier':
        with torch.no_grad():
            model.flow.position_embedding.weight.normal_(std=.03)
    model.eval()
    noisy, time, vertices, points, mask, point_mask = inputs()
    with torch.no_grad():
        out = model(noisy, time, vertices, points, mask, point_mask)
        assert out.abs().max() > .01 and not out[~mask].count_nonzero()
        perm = torch.tensor([6, 1, 3, 0, 5, 4, 2])
        actual = model(noisy[:, perm], time, vertices[:, perm], points, mask[:, perm], point_mask)
        torch.testing.assert_close(actual, out[:, perm], atol=2e-5, rtol=2e-5)
        noisy[~mask] = float('nan')
        vertices[~mask] = float('nan')
        points[~point_mask] = float('nan')
        actual = model(noisy, time, vertices, points, mask, point_mask)
        torch.testing.assert_close(actual, out, atol=0, rtol=0)


def test_execution_policy(variant):
    model, _ = seeded_model(variant)
    activate(model)
    batch = inputs()
    target = torch.randn_like(batch[0])
    original_config = asdict(model.cfg)
    original_keys = tuple(model.state_dict())
    policies = (([], []), ([0, 1], [0, 1]), ([1], [0]))
    expected_checkpoint_calls = (4, 0, 2)
    outputs, gradients = [], []
    for (flow, condition), expected_calls in zip(policies, expected_checkpoint_calls):
        candidate = deepcopy(model).train()
        candidate.set_execution_policy(flow_direct_blocks=flow, condition_direct_blocks=condition)
        assert asdict(candidate.cfg) == original_config
        assert tuple(candidate.state_dict()) == original_keys
        rng_before = torch.get_rng_state().clone()
        with patch.object(topology_flow, 'checkpoint', wraps=topology_flow.checkpoint) as calls:
            prediction = candidate(*batch)
            equal_mesh_velocity_loss(prediction, target, batch[4]).backward()
            assert calls.call_count == expected_calls
        assert torch.equal(rng_before, torch.get_rng_state())
        outputs.append(prediction.detach())
        gradients.append({name: p.grad for name, p in candidate.named_parameters()})
        assert all(g is not None and torch.isfinite(g).all() for g in gradients[-1].values())
        assert sum(p.grad.square().sum() for p in candidate.condition_encoder.parameters()) > 0
    for i in (1, 2):
        torch.testing.assert_close(outputs[0], outputs[i], atol=0, rtol=0)
        for name, grad in gradients[0].items():
            torch.testing.assert_close(grad, gradients[i][name], atol=2e-6, rtol=2e-6)
    model.set_execution_policy([1], [0])
    for invalid_flow, invalid_condition in (([-1], []), ([2], []), ([0, 0], []),
            ([True], []), ([.5], []), ([], [2]), ([], [False])):
        try:
            model.set_execution_policy(invalid_flow, invalid_condition)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid policy accepted')
        assert model.flow._direct_blocks == frozenset([1])
        assert model._condition_direct_blocks == frozenset([0])
    model.set_execution_policy()
    assert not model.flow._direct_blocks and not model._condition_direct_blocks


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    counts = test_initialization_and_structure()
    for variant in ('c0', 'c1_fourier', 'c2_teacher'):
        test_permutation_and_mask(variant)
        test_execution_policy(variant)
    try:
        replace(config(), variant='unsupported')
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid variant accepted')
    print(json.dumps(dict(passed=True, device='cpu', torch_version=torch.__version__,
        optimizer_updates=0, production_scale_tested=False, small_model_parameters=counts,
        production_parameter_delta_vs_c0={'c1_fourier': 40*1536, 'c2_teacher': (42-4*36)*1536},
        tests=['C0/C1 shared seed-0 tensors and RNG identical', 'C1 nonzero-function equivalence',
            'teacher Fourier feature layout', 'C2 structure and parameter deltas',
            'vertex permutation and invalid-mask isolation for all variants',
            'full/mixed/no recomputation output and gradient equivalence',
            'execution-only policy validation and checkpoint call counts']), indent=2))


if __name__ == '__main__':
    main()
