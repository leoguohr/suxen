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

GPU = 'GPU-0fd7fb91-93f2-e726-7898-21d85a3acf36'
EXPECTED = 'ce5972fbd9a194e5aafdd767638544e2f2de96febefa7ec11ee891f040fceb10'


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
    assert not any((run/f).exists() for f in ('STOP', 'FINISH_TRAINING', 'complete.json'))
    failure = read(run/'failure.json')
    assert failure['completed_updates'] == 16780 and not failure['update_in_progress']
    assert 'file write failed' in failure['traceback']
    assert failure['recover_from']['sha256'] == EXPECTED
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    assert GPU not in raw, 'Target GPU is occupied'
    entry = read(run/'recovery-latest.json')
    assert entry['completed_updates'] == 16760 and entry['sha256'] == EXPECTED
    assert sha(entry['path']) == EXPECTED
    cp = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert cp['completed_updates'] == 16760
    assert cp['source_sha256'] == code_hashes()
    assert cp['config'] == config
    assert cp['data'] == read(run/'data_manifest.json')
    assert len(cp['participation']) == 100 and set(cp['participation'].values()) == {838}
    assert cp['cursor']['epoch'] == 838 and cp['cursor']['next_mesh_position'] == 0
    assert cp['scheduler']['completed_updates'] == 16760 and cp['scheduler']['last_lr'] == 1e-4
    steps = {int(s['step']) for s in cp['optimizer']['state'].values()}
    assert steps == {16760}
    assert len(cp['optimizer']['param_groups']) == 1
    group = cp['optimizer']['param_groups'][0]
    assert group['lr'] == 1e-4 and group['betas'] == (.9, .999)
    assert group['eps'] == 1e-8 and group['weight_decay'] == .01
    assert tensor_hash(cp['model']) == entry['model_state_sha256']
    lines = (run/'updates.jsonl').read_bytes().splitlines(keepends=True)
    assert [json.loads(x)['update'] for x in lines] == list(range(1, 16781))
    assert all(json.loads(x)['parameters_finite'] for x in lines)
    archive.mkdir()
    for name in ('config.json', 'budget.json', 'status.json', 'runtime.json', 'recovery-latest.json', 'updates.jsonl', 'failure.json'):
        shutil.copy2(run/name, archive/name)
    shutil.copy2(ROOT/'run_config.json', archive/'root-run_config.json')
    preserved = archive/'checkpoint-step16760-original.pt'
    os.link(entry['path'], preserved)
    assert sha(preserved) == EXPECTED
    (archive/'unsaved-updates16761-16780.jsonl').write_bytes(b''.join(lines[16760:]))
    before_config = copy.deepcopy(cp['config'])
    new_config = copy.deepcopy(config)
    new_config['gpu_uuid'] = GPU
    cp['config'] = new_config
    migrated = save_torch(Path(__file__).parent/'checkpoint-step16760-device-migration.pt', cp)
    with open(migrated['path'], 'rb') as persisted:
        os.fsync(persisted.fileno())
    reloaded = torch.load(migrated['path'], map_location='cpu', weights_only=False)
    equal(cp, reloaded)
    assert {k for k in before_config if before_config[k] != reloaded['config'][k]} == {'gpu_uuid'}
    migrated.update(completed_updates=16760, model_state_sha256=entry['model_state_sha256'],
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
        serialized_roundtrip_exact=True, new_checkpoint_fsync_and_readback_passed=True, previous_stop_reason='checkpoint write failure, no nonfinite training records', storage_failure_root_cause='not established; filesystem free space and inodes available, current full checkpoint write and readback tested', source_hashes=code_hashes(), optimizer_steps=sorted(steps),
        retained_durable_updates=16760, prior_logged_updates=16780,
        unsaved_attempts_archived=[16761,16780], replay_note='Resume from durable 16760; replay 16761-16780. Old attempt records preserved, not counted twice as model progress.',
        prior_log_sha256=sha(archive/'updates.jsonl'), observed_prior_seconds=budget['charged_seconds'],
        interrupted_sessions=closed, no_optimizer_updates_in_preparation=True,
        authorization='User supplied new server credentials and requested continuation. Prior checkpoint write failure and twenty unsaved updates disclosed. Resume full16760 state only after successful storage write/readback; no evaluation or monitoring.')
    write(Path(__file__).parent/'PREPARED.json', evidence)
    # Commit only after preserving and validating all source and migrated state.
    temporary = run/'updates.resumed-prefix.tmp'
    temporary.write_bytes(b''.join(lines[:16760])); temporary.replace(run/'updates.jsonl')
    write(ROOT/'run_config.json', new_config)
    write(run/'config.json', new_config)
    write(run/'budget.json', budget)
    write(run/'recovery-latest.json', migrated)
    write(run/'status.json', dict(state='prepared_for_exact_resume', completed_updates=16760,
        prior_unsaved_logged_update=16780, evidence=str(Path(__file__).parent/'PREPARED.json')))
    assert sha(run/'failure.json') == sha(archive/'failure.json')
    (run/'failure.json').unlink()
    for name in ('recovery-a.tmp', 'failure-scene-not-resumable.tmp'):
        path = run/name
        if path.exists(): path.replace(archive/name)
    assert code_hashes() == cp['source_sha256']
    lock.close()
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__': main()
