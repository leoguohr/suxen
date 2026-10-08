"""Explicit authorization, one-mainline lock, independent accounting and resume state."""
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path
from run_support import read, write, sha, torch, rng_state, restore_rng, save_torch, code_hashes
from data_objective import resume_cursor, negative_faces


def validated_config(path):
    c = read(path)
    assert c['gpu_authorized'] is True, 'GPU use is not authorized'
    assert c['gpu_uuid'].startswith('GPU-')
    assert isinstance(c['evaluation_every'], int) and c['evaluation_every'] > 0
    if not c.get('continuous_authorized', False):
        assert isinstance(c['max_new_updates'], int) and c['max_new_updates'] > 0
        assert isinstance(c['gpu_hours'], (int, float)) and c['gpu_hours'] > 0
    else:
        assert c.get('authorization_note'), 'Record the explicit continuous-training instruction'
    assert c['seed'] == 0 and c['model_variant'] == 'B_v2_teacher_blocks'
    assert c['mesh_per_update'] == 5 and c['warmup_updates'] == 100
    assert c['lr'] == 1e-4 and c['clip'] == 1 and c['weight_decay'] == .01
    assert c['betas'] == [.9, .999] and c['eps'] == 1e-8
    return c


def lock_mainline(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return lock


def claim_run(lock, run):
    lock.seek(0); previous = lock.read().strip()
    assert not previous or previous == str(Path(run).resolve()), 'A different fixed100 mainline is already registered'
    lock.seek(0); lock.truncate(); lock.write(str(Path(run).resolve())); lock.flush(); os.fsync(lock.fileno())


def verify_gpu(uuid):
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'], text=True)
    found = [x for x in raw.splitlines() if x.split(',')[0].strip() == uuid]
    assert len(found) == 1 and 'A100' in found[0] and '80GB' in found[0], 'Authorized GPU is absent'
    raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'], text=True)
    assert not any(uuid in x for x in raw.splitlines()), 'Authorized GPU already has a compute process'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == uuid, 'Select the exact authorized UUID before importing torch'
    return found[0]


class Budget:
    def __init__(self, run, config):
        self.run = Path(run); self.path = self.run/'budget.json'; self.config = config
        self.state = read(self.path) if self.path.exists() else dict(charged_seconds=0., sessions=[])
        # An interrupted process is conservatively charged through this resume.
        now = time.time()
        for row in self.state['sessions']:
            if row.get('ended') is None:
                row['ended'] = now; row['interrupted_conservative_accounting'] = True
        self.started = now
        self.state['sessions'].append(dict(started=now, ended=None, pid=os.getpid(), gpu_uuid=config['gpu_uuid']))
        self.update()

    def update(self, finish=False):
        now = time.time(); self.state['sessions'][-1]['heartbeat'] = now
        if finish: self.state['sessions'][-1]['ended'] = now
        total = sum((r['ended'] if r['ended'] is not None else now)-r['started'] for r in self.state['sessions'])
        self.state.update(charged_seconds=total, gpu_hours_limit=self.config.get('gpu_hours'),
                          scope='assigned process wall time including model initialization, training, saves and evaluation')
        write(self.path, self.state)
        return total

    def remaining(self):
        spent = self.update(); hours = self.config.get('gpu_hours')
        return float('inf') if hours is None else max(0., hours*3600-spent)

    def stop(self, reserve=0):
        return (self.run/'STOP').exists() or self.remaining() <= reserve


def next_state(items, uids, completed, seed=0):
    cursor = resume_cursor(uids, completed, seed)
    cursor['next_negative_hashes'] = {
        u: negative_faces(items[u], cursor['epoch'], seed)[1] for u in cursor['next_uids']}
    return cursor


def build_checkpoint(model, opt, completed, participation, config, data, items, frozen_hash):
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    return dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng_state(),
        completed_updates=completed, participation=participation, config=config, data=data,
        optimizer_parameter_names=names, model_config=model.cfg.to_dict(),
        cursor=next_state(items, data['uids'], completed, config['seed']),
        frozen_logvar_hash=frozen_hash, source_sha256=code_hashes(),
        scheduler=dict(kind='linear_100_update_warmup_then_constant', completed_updates=completed,
                       last_lr=opt.param_groups[0]['lr']))


def restore_checkpoint(cp, model, opt, items, data, config):
    assert cp['data'] == data and cp['config'] == config
    assert cp['source_sha256'] == code_hashes(), 'Source changed since checkpoint'
    assert cp['model_config'] == model.cfg.to_dict()
    assert cp['optimizer_parameter_names'] == [n for n,p in model.named_parameters() if p.requires_grad]
    model.load_state_dict(cp['model'], strict=True); opt.load_state_dict(cp['optimizer'])
    completed = cp['completed_updates']
    assert cp['cursor'] == next_state(items, data['uids'], completed, config['seed'])
    assert sum(cp['participation'].values()) == completed*5
    expected = {u:completed//20 for u in data['uids']}
    for u in cp['cursor']['order'][:cp['cursor']['next_mesh_position']]: expected[u] += 1
    assert cp['participation'] == expected
    assert cp['scheduler']['completed_updates'] == completed
    assert cp['scheduler']['last_lr'] == opt.param_groups[0]['lr']
    expected_lr = config['lr']*min(max(completed, 1)/config['warmup_updates'], 1)
    assert opt.param_groups[0]['lr'] == expected_lr
    for state in opt.state.values():
        if 'step' in state: assert int(state['step']) == completed
    restore_rng(cp['rng'])
    return completed, cp['participation'].copy()
