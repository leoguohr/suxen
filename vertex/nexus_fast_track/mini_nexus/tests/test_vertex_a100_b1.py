import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import numpy as np

from mini_nexus.data_2k import collate_nexus2k_samples, load_nexus2k_sample
from scripts.train_vertex_a100_b1 import (
    a_passes, fixed_seed, make_model, response_metrics, response_passes, token_energy, assert_b1_scope,
)
from scripts.vertex_b1_evidence import empty_baseline
from test_data_2k import _write_manifest, _write_sample


def test_signed_response_rejects_correct_magnitude_wrong_direction():
    dx = torch.randn(1, 8, 8)
    correct = response_metrics(-2 * dx, dx)
    wrong = response_metrics(2 * dx, dx)
    assert correct['signed_gain'] == pytest.approx(-2)
    assert correct['relative_response_error'] == 0
    assert correct['magnitude_gain'] == wrong['magnitude_gain'] == 2
    assert response_passes(correct) and not response_passes(wrong)


def test_energy_statistics_detect_single_channel_dominance_per_token():
    x = torch.ones(1, 2, 16)
    x[:, 1] = 0
    x[:, 1, 3] = 8
    out = token_energy(x)
    assert out['token_top1_energy_fraction'][0] == [1 / 16, 1.]
    assert out['token_top8_energy_fraction'][0] == [.5, 1.]


def test_every_fixed_problem_must_pass_not_just_average():
    good = dict(velocity_mse=1e-5, relative_mse=1e-5, exact_occupancy=True, exact_coordinate_set=True)
    assert a_passes([good] * 100)
    assert not a_passes([good] * 99 + [{**good, 'exact_coordinate_set': False}])
    assert not a_passes([good] * 99 + [{**good, 'velocity_mse': .001}])
    assert fixed_seed('first', 1) == fixed_seed('first', 1) != fixed_seed('second', 1)


def test_stable_initialization_changes_only_requested_tensors():
    a, b = make_model('R0', 17, True), make_model('R1', 17, True)
    for name, original in a.state_dict().items():
        changed = name == 'flow.depth_embedding.weight' or name.startswith(('flow.time_embedding.0.', 'flow.time_embedding.2.'))
        changed |= 'cross_attention.output.' in name
        if not changed:
            torch.testing.assert_close(original, b.state_dict()[name], atol=0, rtol=0)
    assert b.flow.depth_embedding.weight.std() < .03
    assert b.flow.time_embedding[0].weight.std() < .03
    assert all(torch.count_nonzero(block.cross_attention.output.weight) == 0 for block in b.flow.blocks)


def test_zero_cross_outputs_delay_but_do_not_disconnect_vecset_gradients(tmp_path):
    row = _write_sample(tmp_path, 'sample', 'train', torch.tensor([[0,0,0],[0,0,511],[0,511,0],[511,0,0]]))
    batch = collate_nexus2k_samples([load_nexus2k_sample(row)])
    model = make_model('R1', 17, True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    noise = torch.randn_like(batch.octree_levels[8].target)
    gradients = []
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        model(batch.condition, batch.octree_levels[8], noise=noise, time=torch.tensor([.5])).backward()
        gradients.append(sum(float(p.grad.abs().sum()) for p in model.condition_encoder.parameters() if p.grad is not None))
        optimizer.step()
    assert gradients[0] == gradients[1] == 0
    assert max(gradients[2:]) > 0


@pytest.mark.parametrize('phase', ['pipeline', 'B1'])
def test_cli_pipeline_gate_and_independent_b1_eight_noise_accumulation(tmp_path, phase):
    cells = torch.tensor([[0,0,0],[0,0,511],[0,511,0],[511,0,0]])
    count = 2 if phase == 'pipeline' else 1
    rows = [_write_sample(tmp_path, f'fixture_{i}', 'train', cells) for i in range(count)]
    manifest = tmp_path / 'manifest.csv'; _write_manifest(manifest, rows)
    script = Path(__file__).resolve().parents[1] / 'scripts/train_vertex_a100_b1.py'
    output = tmp_path / 'run'
    result = subprocess.run([sys.executable, str(script), '--manifest', str(manifest), '--output', str(output),
        '--phase', phase, '--expected-samples', str(count), '--smoke-model', '--device', 'cpu', '--precision', 'fp32',
        '--a-updates', '2', '--a-eval-every', '2', '--b-updates', '1', '--b-max-updates', '1'],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    if phase == 'pipeline':
        assert not (output / 'B1_R0').exists()
        records = [json.loads(r) for r in (output / 'A100/train.jsonl').read_text().splitlines()]
        assert {r['uid'] for r in records} == {'fixture_0', 'fixture_1'}
    else:
        assert not (output / 'A100').exists()
        a = json.loads((output / 'B1_R0/train.jsonl').read_text())
        b = json.loads((output / 'B1_R1/train.jsonl').read_text())
        assert a['noise_sha256'] == b['noise_sha256']
        assert len(set(a['noise_sha256'])) == a['accumulation'] == 8
        assert a['time'] == b['time'] == .5
        assert a['lr'] == pytest.approx(1e-7)
        for variant in ('R0', 'R1'):
            c = json.loads((output / f'B1_{variant}/config.json').read_text())
            assert c['fresh_initialization'] and c['loaded_checkpoint'] is None
            assert not (output / f'B1_{variant}/heldout_once.json').exists()
            folder = output / f'B1_{variant}'
            report = json.loads((folder / 'evaluation-000001.json').read_text())
            assert report['scope']['actually_consumed_uids'] == ['fixture_0']
            assert report['scope']['consumed_updates'] == 1
            cache = report['training_cache']
            assert cache['tensor_sha256']['noise'] == a['noise_sha256'][0]
            assert report['training_cache_probe']['noisy_sha256'] == cache['tensor_sha256']['noisy']
            assert report['training_cache_probe']['target_velocity_sha256'] == cache['tensor_sha256']['target_velocity']
            arrays = np.load(folder / 'training_cache.npz')
            np.testing.assert_array_equal(arrays['noisy'], .5 * arrays['noise'] + .5 * arrays['target_occupancy'])
            assert all(r['empty_baseline']['generated_vertices'] == 0 for r in report['probes'])
            record = json.loads((folder / 'train.jsonl').read_text())
            assert record['projection_gradients_before_clip']['output.weight'] > 0
            assert record['projection_updates']['output.weight']['update_norm'] > 0
            assert report['responses_fp32'][0]['layers']['block_00']['rms'] > 0


def test_b1_rejects_multiple_or_wrong_uids():
    from types import SimpleNamespace
    anchor = SimpleNamespace(uid='nexus_2k_000105')
    assert_b1_scope([anchor])
    with pytest.raises(AssertionError):
        assert_b1_scope([anchor, anchor])
    with pytest.raises(AssertionError):
        assert_b1_scope([SimpleNamespace(uid='wrong')])


def test_empty_velocity_control_matches_four_p_and_recovers_no_shape():
    y = torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0.]]])
    noise = torch.randn_like(y)
    result = empty_baseline(.5 * (noise + y), y - noise, y)
    assert result['velocity_mse'] == pytest.approx(.5)
    assert result['theoretical_mse_4p'] == .5
    assert result['generated_vertices'] == 0
    assert result['occupancy']['tp'] == 0 and result['occupancy']['fn'] == 1
