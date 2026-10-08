import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from mini_nexus.data_2k import load_nexus2k_sample
from scripts.evaluate_vertex_b1_frozen import separated_gates
from scripts.train_vertex_a100_b1 import make_model
from test_data_2k import _write_manifest, _write_sample


def test_geometry_and_response_are_separate_without_changing_strict_gate():
    probes = [{'exact_coordinate_set': True, 'velocity_mse': .001}] * 64
    response = [{'signed_gain': -2., 'relative_response_error': .12}] * 64
    assert separated_gates(probes, response) == dict(geometry_regression_passed=True,
        local_response_diagnostic_passed=False, strict_passed_on_this_followup=False)
    assert not separated_gates(probes[:-1], response)['geometry_regression_passed']


def test_frozen_cli_preserves_checkpoint_and_exports_recomputable_responses(tmp_path):
    uid = 'nexus_2k_000105'
    cells = torch.tensor([[x, y, z] for x in [0, 511] for y in [0, 511] for z in [0, 511]])
    row = _write_sample(tmp_path, uid, 'train', cells)
    sample = load_nexus2k_sample(row)
    manifest = tmp_path / 'manifest.csv'; _write_manifest(manifest, [row])
    model = make_model('R1', 17, True)
    run = tmp_path / 'run'; run.mkdir()
    config = {'phase': 'B1', 'variant': 'R1', 'uids': [uid], 'manifest': str(manifest),
              'condition_sha256': {uid: hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest()},
              'smoke_model': True, 'seed': 17, 'precision': 'fp32',
              'parameter_count': sum(p.numel() for p in model.parameters())}
    checkpoint = run / 'checkpoint-last.pt'
    torch.save({'step': 1000, 'config': config, 'model': model.state_dict()}, checkpoint)
    original = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'evaluation'
    command = [sys.executable, str(root / 'scripts/evaluate_vertex_b1_frozen.py'),
               '--code-root', str(root), '--run-dir', str(run), '--output', str(output), '--device', 'cpu']
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original
    report = json.loads((output / 'report.json').read_text())
    assert len(report['probes']) == len(report['responses_fp32']) == 64
    assert len(report['original_four_responses_fp32']) == 4
    for response in report['responses_fp32'] + report['original_four_responses_fp32']:
        z = np.load(output / 'arrays' / f"response-{response['seed']}.npz")
        dx = z['delta_x'].astype('float64')
        dv = (z['velocity_after'] - z['velocity_before']).astype('float64')
        assert np.isclose(response['signed_gain'], (dv * dx).sum() / (dx * dx).sum())
        assert np.isclose(response['relative_response_error'], np.linalg.norm(dv + 2 * dx) / (2 * np.linalg.norm(dx)))
    # An existing evaluation directory cannot silently be overwritten or consumed twice.
    repeat = subprocess.run(command, text=True, capture_output=True)
    assert repeat.returncode != 0
