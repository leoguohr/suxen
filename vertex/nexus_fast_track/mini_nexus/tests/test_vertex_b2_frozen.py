import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from mini_nexus.data_2k import load_nexus2k_sample
from scripts.train_vertex_a100_b1 import make_model
from scripts.evaluate_vertex_b2_frozen import occupancy_metrics, sampling_gate
from test_data_2k import _write_manifest, _write_sample


def test_geometry_gate_ignores_diagnostic_errors():
    rows = [{'exact_coordinate_set': True, 'occupancy_mse': 999, 'minimum_signed_threshold_margin': .00001}] * 64
    assert sampling_gate(rows)
    assert not sampling_gate(rows[:63])
    assert not sampling_gate(rows[:63] + [{'exact_coordinate_set': False}])
    gt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    predicted = torch.tensor([[1, 2, 3], [7, 8, 9]])
    row = occupancy_metrics(torch.tensor([[[.49, .8]]]), torch.tensor([[[1., 0.]]]), predicted, gt)
    assert row['missing_count'] == row['extra_count'] == 1
    assert row['minimum_signed_threshold_margin'] < 0


def make_checkpoint(tmp_path):
    uid = 'nexus_2k_000105'
    cells = torch.tensor([[x, y, z] for x in [0, 511] for y in [0, 511] for z in [0, 511]])
    row = _write_sample(tmp_path, uid, 'train', cells)
    sample = load_nexus2k_sample(row)
    manifest = tmp_path / 'manifest.csv'; _write_manifest(manifest, [row])
    model = make_model('R1', 17, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False)
    sum(p.square().mean() for p in model.parameters()).backward(); opt.step()
    source = {'phase': 'B1', 'variant': 'R1', 'uids': [uid], 'manifest': str(manifest),
              'condition_sha256': {uid: hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest()},
              'smoke_model': True, 'seed': 17, 'precision': 'fp32'}
    config = {'phase': 'B2', 'variant': 'R1', 'uids': [uid], 'source_config': source,
              'parameter_count': sum(p.numel() for p in model.parameters())}
    cp = tmp_path / 'source.pt'
    torch.save({'step': 2000, 'b2_update': 1000, 'config': config, 'model': model.state_dict(),
                'optimizer': opt.state_dict(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': None}, cp)
    return cp, manifest


def test_frozen_b2_cli_uses_step2000_and_actual_saved_noise(tmp_path):
    cp, manifest = make_checkpoint(tmp_path)
    paired = tmp_path / 'paired.npz'
    noise = np.arange(64, dtype=np.float32).reshape(1, 8, 8) / 64
    np.savez_compressed(paired, noise=noise)
    sha = hashlib.sha256(cp.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]; out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(root / 'scripts/evaluate_vertex_b2_frozen.py'),
        '--code-root', str(root), '--checkpoint', str(cp), '--manifest', str(manifest), '--output', str(out),
        '--paired-noise', str(paired), '--expected-sha', sha, '--device', 'cpu'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((out / 'report.json').read_text())
    assert report['source_step'] == 2000 and report['source_b2_update'] == 1000
    assert len(report['samples']) == 64 and not report['parameters_updated']
    assert report['seeds'] == list(range(14000000, 14000064))
    expected = hashlib.sha256(noise.tobytes()).hexdigest()
    assert {r['noise_sha256'] for r in report['paired_13000002']} == {expected}
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == sha
