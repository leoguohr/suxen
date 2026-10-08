"""Archive an interrupted attempt and migrate only deployment GPU metadata on CPU."""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_support import read, write, sha, torch, tensor_hash, code_hashes, save_torch
from runtime_fixed100 import lock_mainline
import numpy as np

GPU = 'GPU-c305d79a-5668-a5cd-17b8-af8c89a95cfb'
EXPECTED = '3028c73e7bc244c5eaa73585f4439fb85d47d743fd0094795f0cedca07331c41'


def equal(a, b):
    assert type(a) is type(b), (type(a), type(b))
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else:
        assert a == b


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    config = read(ROOT/'run_config.json')
    lock = lock_mainline(config['mainline_lock'])
    run = Path(config['run_directory'])
    archive = Path(__file__).parent/'prior_attempt'
    assert not archive.exists(), 'Preparation is one-shot; inspect evidence before retrying'
    assert read(run/'config.json') == config
    assert not any((run/f).exists() for f in ('STOP', 'FINISH_TRAINING', 'complete.json', 'failure.json'))
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert GPU not in raw, 'Target GPU is occupied'
    entry = read(run/'recovery-latest.json')
    assert entry['completed_updates'] == 13280 and entry['sha256'] == EXPECTED
    assert sha(entry['path']) == EXPECTED
    cp = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert cp['completed_updates'] == 13280
    assert cp['source_sha256'] == code_hashes()
    assert cp['config'] == config
    assert cp['data'] == read(run/'data_manifest.json')
    assert len(cp['participation']) == 100 and set(cp['participation'].values()) == {664}
    assert cp['cursor']['epoch'] == 664 and cp['cursor']['next_mesh_position'] == 0
    assert cp['scheduler']['completed_updates'] == 13280 and cp['scheduler']['last_lr'] == 1e-4
    steps = {int(s['step']) for s in cp['optimizer']['state'].values()}
    assert steps == {13280}
    assert len(cp['optimizer']['param_groups']) == 1
    group = cp['optimizer']['param_groups'][0]
    assert group['lr'] == 1e-4 and group['betas'] == (.9, .999)
    assert group['eps'] == 1e-8 and group['weight_decay'] == .01
    assert tensor_hash(cp['model']) == entry['model_state_sha256']
    lines = (run/'updates.jsonl').read_bytes().splitlines(keepends=True)
    assert [json.loads(x)['update'] for x in lines] == list(range(1, 13281))
    assert all(json.loads(x)['parameters_finite'] for x in lines)
    archive.mkdir()
    for name in ('config.json', 'budget.json', 'status.json', 'runtime.json', 'recovery-latest.json', 'updates.jsonl'):
        shutil.copy2(run/name, archive/name)
    shutil.copy2(ROOT/'run_config.json', archive/'root-run_config.json')
    preserved = archive/'checkpoint-step13280-original.pt'
    os.link(entry['path'], preserved)
    assert sha(preserved) == EXPECTED
    before_config = copy.deepcopy(cp['config'])
    new_config = copy.deepcopy(config)
    new_config['gpu_uuid'] = GPU
    cp['config'] = new_config
    migrated = save_torch(Path(__file__).parent/'checkpoint-step13280-device-migration.pt', cp)
    reloaded = torch.load(migrated['path'], map_location='cpu', weights_only=False)
    equal(cp, reloaded)
    assert {k for k in before_config if before_config[k] != reloaded['config'][k]} == {'gpu_uuid'}
    migrated.update(completed_updates=13280, model_state_sha256=entry['model_state_sha256'],
                    reason='exact_state_with_explicit_deployment_GPU_metadata_migration')
    write(Path(migrated['path']).with_suffix('.json'), migrated)
    budget = read(run/'budget.json')
    closed = []
    for row in budget['sessions']:
        if row['ended'] is None:
            row['ended'] = row['heartbeat']
            row['interrupted_observed_lower_bound_only'] = True
            row['note'] = 'Actual external interruption time unknown; close at last observed heartbeat. Unobserved downtime is not counted as GPU execution.'
            closed.append(dict(started=row['started'], last_observed_heartbeat=row['heartbeat']))
    budget['charged_seconds'] = sum(r['ended']-r['started'] for r in budget['sessions'])
    budget['historical_time_is_observed_lower_bound'] = True
    evidence = dict(prepared_at=time.time(), original_checkpoint=entry, preserved_original=str(preserved),
        migrated_checkpoint=migrated, changed_checkpoint_fields=['config.gpu_uuid'],
        old_gpu_uuid=config['gpu_uuid'], new_gpu_uuid=GPU,
        model_optimizer_rng_cursor_scheduler_participation_exactly_preserved=True,
        serialized_roundtrip_exact=True, source_hashes=code_hashes(), optimizer_steps=sorted(steps),
        retained_durable_updates=13280, prior_logged_updates=13280,
        unsaved_attempts_archived=[], replay_note='No updates lost or replayed: last logged update equals durable step13280.',
        prior_log_sha256=sha(archive/'updates.jsonl'), observed_prior_seconds=budget['charged_seconds'],
        interrupted_sessions=closed, no_optimizer_updates_in_preparation=True,
        authorization='User reported the old server stopped and requested continuation on the supplied new server. Resume full step13280 state, continuous checkpoint-only training, no monitoring.')
    write(Path(__file__).parent/'PREPARED.json', evidence)
    # Commit only after preserving and validating all source and migrated state.
    write(ROOT/'run_config.json', new_config)
    write(run/'config.json', new_config)
    write(run/'budget.json', budget)
    write(run/'recovery-latest.json', migrated)
    write(run/'status.json', dict(state='prepared_for_exact_resume', completed_updates=13280,
        prior_logged_update=13280, evidence=str(Path(__file__).parent/'PREPARED.json')))
    assert code_hashes() == cp['source_sha256']
    lock.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__': main()
