import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mini_nexus.data_2k import load_nexus2k_sample
from scripts.train_vertex_a100_b1 import make_model
from scripts.train_vertex_b2 import time_response, verify_restored_optimizer, persist_checkpoint
from test_data_2k import _write_manifest, _write_sample


def test_response_reference_changes_with_time():
    dx = torch.randn(1, 8, 8)
    for t in [0, .1, .5, .95]:
        row = time_response(-dx / (1 - t), dx, t)
        assert row['signed_gain'] == pytest.approx(-1 / (1 - t), rel=1e-6)
        assert row['relative_response_error'] < 1e-6
    assert time_response(-2 * dx, dx, .9)['relative_response_error'] > .7


def test_resume_verification_detects_changed_adam_state():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False)
    model(torch.ones(1, 2)).sum().backward();opt.step()
    import copy
    saved = copy.deepcopy(opt.state_dict())
    assert verify_restored_optimizer(opt, saved) == 6
    opt.state[model.weight]['exp_avg'].add_(1)
    with pytest.raises(AssertionError):verify_restored_optimizer(opt, saved)


def test_checkpoint_backup_preserves_contents_and_rng(tmp_path):
    source = tmp_path / 'source.pt'
    torch.save({'model': torch.randn(4, 4), 'optimizer': {'step': 1100}}, source)
    rng = torch.get_rng_state()
    result = persist_checkpoint(source, tmp_path / 'persistent', 100)
    assert result['verified_readback'] and result['b2_update'] == 100
    assert Path(result['path']).read_bytes() == source.read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['sha256']
    assert torch.equal(rng, torch.get_rng_state())


def test_b2_cli_randomizes_each_microbatch_and_keeps_adam_lr(tmp_path):
    uid = 'nexus_2k_000105'
    cells = torch.tensor([[x, y, z] for x in [0, 511] for y in [0, 511] for z in [0, 511]])
    row = _write_sample(tmp_path, uid, 'train', cells)
    sample = load_nexus2k_sample(row)
    manifest = tmp_path / 'manifest.csv';_write_manifest(manifest, [row])
    model = make_model('R1', 17, True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False)
    # Small synthetic optimizer state for the execution test, not a training result.
    sum(p.square().mean() for p in model.parameters()).backward();opt.step()
    config = {'phase': 'B1', 'variant': 'R1', 'uids': [uid], 'manifest': str(manifest),
              'condition_sha256': {uid: hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest()},
              'smoke_model': True, 'seed': 17, 'precision': 'fp32'}
    cp = tmp_path / 'source.pt'
    torch.save({'step': 1000, 'config': config, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': None}, cp)
    sha = hashlib.sha256(cp.read_bytes()).hexdigest()
    gate = tmp_path / 'gate.json';gate.write_text(json.dumps({'geometry_regression_passed': True, 'checkpoint_sha256': sha}))
    root = Path(__file__).resolve().parents[1];out = tmp_path / 'out'
    result = subprocess.run([sys.executable, str(root / 'scripts/train_vertex_b2.py'), '--code-root', str(root),
        '--checkpoint', str(cp), '--frozen-report', str(gate), '--output', str(out),
        '--updates', '1', '--eval-every', '1', '--device', 'cpu',
        '--backup-dir', str(tmp_path / 'persistent')], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    record = json.loads((out / 'train.jsonl').read_text())
    assert record['uid'] == uid and record['lr'] == 1e-5
    assert len(set(record['times'])) == len(set(record['noise_sha256'])) == 8
    assert all(0 <= t < 1 and t != .5 for t in record['times'])
    final = torch.load(out / 'checkpoint-last.pt', weights_only=False)
    assert final['step'] == 1001
    assert (tmp_path / 'persistent/checkpoint-last.pt').read_bytes() == (out / 'checkpoint-last.pt').read_bytes()
    assert all(s['step'] == 2 for s in final['optimizer']['state'].values())
    assert hashlib.sha256(cp.read_bytes()).hexdigest() == sha
    report = json.loads((out / 'evaluation-000001.json').read_text())
    assert len(report['by_time']) == 7 and len(report['sampling']) == 16
    assert {r['steps'] for r in report['sampling']} == {20, 40}
    fresh = json.loads((out / 'fresh_sampling_once.json').read_text())
    assert {r['seed'] for r in fresh['samples']}.isdisjoint({r['seed'] for r in report['sampling']})
