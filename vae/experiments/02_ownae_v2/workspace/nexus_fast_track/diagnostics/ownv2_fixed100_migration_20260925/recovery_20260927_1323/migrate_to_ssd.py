"""Resume the same run on writable persistent storage, without scientific changes."""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OLD = Path('/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_migration_20260925')
NEW = Path('/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_resume16760_20260927')
GPU = 'GPU-0fd7fb91-93f2-e726-7898-21d85a3acf36'
EXPECTED = 'ce5972fbd9a194e5aafdd767638544e2f2de96febefa7ec11ee891f040fceb10'
sys.path.insert(0, str(OLD))
from run_support import read, write, sha, torch, tensor_hash, save_torch, code_hashes
from runtime_fixed100 import lock_mainline, next_state
from data_objective import load_dataset
import numpy as np


def equal(a, b):
    assert type(a) is type(b)
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
    oldrun = OLD/'run'; run = NEW/'run'
    config = read(OLD/'run_config.json')
    lock = lock_mainline(config['mainline_lock'])
    lock.seek(0); assert lock.read().strip() == str(oldrun)
    assert not NEW.exists(), 'One-shot migration: inspect any existing attempt before retrying'
    assert not any((oldrun/x).exists() for x in ('STOP', 'FINISH_TRAINING', 'complete.json'))
    compute = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'], text=True)
    assert GPU not in compute
    entry = read(oldrun/'recovery-latest.json')
    assert entry['completed_updates'] == 16760 and entry['sha256'] == EXPECTED
    assert sha(entry['path']) == EXPECTED
    failure = read(oldrun/'failure.json')
    assert failure['completed_updates'] == 16780 and not failure['update_in_progress']
    assert 'file write failed' in failure['traceback']
    cp = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert cp['source_sha256'] == code_hashes()
    assert cp['config'] == config == read(oldrun/'config.json')
    assert cp['completed_updates'] == 16760 and set(cp['participation'].values()) == {838}
    assert len(cp['participation']) == 100
    assert {int(s['step']) for s in cp['optimizer']['state'].values()} == {16760}
    group = cp['optimizer']['param_groups'][0]
    assert len(cp['optimizer']['param_groups']) == 1
    assert group['lr'] == 1e-4 and group['betas'] == (.9,.999)
    assert group['eps'] == 1e-8 and group['weight_decay'] == .01
    assert tensor_hash(cp['model']) == entry['model_state_sha256']
    items, data = load_dataset(config['data_source'])
    assert data == cp['data'] == read(oldrun/'data_manifest.json')
    assert cp['cursor'] == next_state(items, data['uids'], 16760, config['seed'])
    lines = (oldrun/'updates.jsonl').read_bytes().splitlines(keepends=True)
    assert [json.loads(x)['update'] for x in lines] == list(range(1,16781))
    run.mkdir(parents=True)
    evidence_dir = NEW/'migration'; evidence_dir.mkdir()
    for filename, digest in cp['source_sha256'].items():
        shutil.copy2(OLD/filename, NEW/filename)
        assert sha(NEW/filename) == digest
    shutil.copy2(OLD/'launch.sh', NEW/'launch.sh')
    shutil.copy2(__file__, evidence_dir/'migrate_to_ssd.py')
    for filename in ('failure.json','status.json','runtime.json','config.json','budget.json','recovery-latest.json'):
        shutil.copy2(oldrun/filename, evidence_dir/('prior-'+filename))
    (evidence_dir/'unsaved-updates16761-16780.jsonl').write_bytes(b''.join(lines[16760:]))
    (run/'updates.jsonl').write_bytes(b''.join(lines[:16760]))
    new_config = copy.deepcopy(config)
    new_config['gpu_uuid'] = GPU
    new_config['run_directory'] = str(run)
    cp['config'] = new_config
    migrated = save_torch(evidence_dir/'checkpoint-step16760-storage-migration.pt', cp)
    with open(migrated['path'], 'rb') as f: os.fsync(f.fileno())
    reloaded = torch.load(migrated['path'], map_location='cpu', weights_only=False)
    equal(cp, reloaded)
    assert {k for k in config if config[k] != new_config[k]} == {'gpu_uuid','run_directory'}
    migrated.update(completed_updates=16760,model_state_sha256=entry['model_state_sha256'],reason='exact_state_with_storage_and_GPU_deployment_migration')
    write(Path(migrated['path']).with_suffix('.json'), migrated)
    write(NEW/'run_config.json',new_config);write(run/'config.json',new_config)
    write(run/'data_manifest.json',data);write(run/'budget.json',read(oldrun/'budget.json'))
    write(run/'recovery-latest.json',migrated)
    write(run/'status.json',dict(state='prepared_for_exact_resume',completed_updates=16760,prior_logged_updates=16780))
    # Preserve failed partial outputs on SSD before freeing their unusable bytes.
    moved = []
    for filename in ('recovery-a.tmp','failure-scene-not-resumable.tmp'):
        src=oldrun/filename
        if src.exists():
            dest=evidence_dir/filename
            shutil.copy2(src,dest)
            digest=sha(src);assert sha(dest)==digest
            moved.append(dict(old_path=str(src),new_path=str(dest),sha256=digest,bytes=dest.stat().st_size))
            src.unlink()
    evidence = dict(prepared_at=time.time(),source_root=str(OLD),active_root=str(NEW),
        original_checkpoint=entry,migrated_checkpoint=migrated,
        changed_checkpoint_fields=['config.gpu_uuid','config.run_directory'],
        exact_model_optimizer_rng_cursor_scheduler_participation_preserved=True,
        serialized_roundtrip_exact=True,full_checkpoint_fsync_and_readback_passed=True,
        original_storage_error='errno122 Disk quota exceeded confirmed by 1MiB write probe',
        destination_filesystem='ssdwork GPFS persistent mount; 1MiB probe and full checkpoint write passed',
        source_hashes=cp['source_sha256'],prior_logged_updates=16780,resume_from=16760,
        unsaved_updates_to_replay=[16761,16780],prior_log_path=str(oldrun/'updates.jsonl'),prior_log_sha256=sha(oldrun/'updates.jsonl'),
        archived_failed_partial_files=moved,original_valid_checkpoints_untouched=True,
        no_optimizer_updates_in_migration=True,expected_next_state=cp['cursor'],
        authorization='User requested continuation. Migrate only writable output location and device; keep continuous checkpoint-only training, original data and scientific settings, no monitoring.')
    write(evidence_dir/'PREPARED.json',evidence)
    # Same cross-instance lock, now explicitly registering the relocated mainline.
    lock.seek(0);lock.truncate();lock.write(str(run));lock.flush();os.fsync(lock.fileno())
    write(OLD/'repro_outputs'/'ACTIVE_LOCATION.json',dict(active_root=str(NEW),active_run=str(run),migration_evidence=str(evidence_dir/'PREPARED.json'),reason='old filesystem quota exhausted; no parallel branch'))
    assert code_hashes() == cp['source_sha256']
    lock.close()
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('expected_next_state','source_hashes')},indent=2),flush=True)


if __name__ == '__main__': main()
