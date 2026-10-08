"""One exclusive, persistent GPU budget shared by export, training and evaluation."""
import fcntl
import json
import math
import os
import time
from pathlib import Path
from _faces_reference import atomic_json


class GPUBudget:
    """Charge wall reservation time on one explicitly authorized GPU, including setup.

    A CUDA ledger must be created explicitly with schema_version=1,
    authorized_gpu_uuid, max_gpu_seconds and max_optimizer_updates. An unclosed
    attempt fails closed for manual crash accounting; it never silently resets.
    """
    def __init__(self, path, phase, device):
        self.path = Path(path).resolve() if path else None
        self.phase, self.device = phase, str(device)
        self.cuda = self.device.startswith('cuda')
        self.lock = None
        self.state = None
        self.started = None
        self.last_write = 0.

    def __enter__(self):
        if not self.cuda:
            if self.path: raise ValueError('GPU budget ledgers cannot be charged by CPU runs')
            return self
        if not self.path: raise ValueError('CUDA work requires the shared --budget-file')
        self.lock = self.path.with_suffix(self.path.suffix+'.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.state = json.loads(self.path.read_text())
            s = self.state
            if s.get('schema_version') != 1: raise ValueError('Unsupported GPU ledger schema')
            gpu = s['authorized_gpu_uuid']
            if not isinstance(gpu, str) or not gpu.startswith('GPU-'):
                raise ValueError('The authorized GPU UUID must be explicitly recorded')
            if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or self.device not in ('cuda', 'cuda:0'):
                raise ValueError('Require CUDA_VISIBLE_DEVICES=<authorized UUID> and --device cuda:0')
            if not math.isfinite(s['max_gpu_seconds']) or s['max_gpu_seconds'] <= 0:
                raise ValueError('Invalid total GPU time allowance')
            if type(s['max_optimizer_updates']) is not int or s['max_optimizer_updates'] <= 0:
                raise ValueError('Invalid total optimizer allowance')
            if s.get('active') is not None:
                raise RuntimeError('Unclosed GPU attempt: reconcile crash time before any restart')
            s.setdefault('charged_gpu_seconds', 0.)
            s.setdefault('effective_optimizer_updates', 0)
            s.setdefault('last_completed_updates', 0)
            s.setdefault('attempts', [])
            if s['charged_gpu_seconds'] >= s['max_gpu_seconds']:
                raise RuntimeError('The shared GPU time budget is exhausted')
            self.base_seconds = float(s['charged_gpu_seconds'])
            self.started = time.monotonic()
            s['active'] = dict(phase=self.phase, pid=os.getpid(), started_unix=time.time(),
                               gpu_uuid=gpu, charged_before=self.base_seconds)
            self._commit(force=True)
            return self
        except BaseException:
            self.lock.close(); self.lock = None
            raise

    def _charge(self):
        if self.started is not None:
            self.state['charged_gpu_seconds'] = self.base_seconds+time.monotonic()-self.started
            self.state['active']['last_seen_unix'] = time.time()

    def _commit(self, force=False):
        if not self.cuda: return
        self._charge()
        now = time.monotonic()
        if force or now-self.last_write >= 5:
            atomic_json(self.path, self.state)
            self.last_write = now

    def remaining_seconds(self):
        if not self.cuda: return float('inf')
        self._commit()
        return max(0., self.state['max_gpu_seconds']-self.state['charged_gpu_seconds'])

    def stop(self, reserve_seconds=0):
        if reserve_seconds < 0: raise ValueError('Negative shutdown reserve')
        return self.remaining_seconds() <= reserve_seconds

    def require_training_cursor(self, completed):
        if self.cuda and completed != self.state['last_completed_updates']:
            raise ValueError('Training cursor differs from the durable GPU budget ledger')

    def updates_exhausted(self):
        return self.cuda and self.state['effective_optimizer_updates'] >= self.state['max_optimizer_updates']

    def record_update(self, completed):
        if not self.cuda: return
        if self.updates_exhausted(): raise RuntimeError('Total optimizer budget exhausted')
        if completed != self.state['last_completed_updates']+1:
            raise ValueError('Refuse repeated or skipped optimizer-update accounting')
        self.state['last_completed_updates'] = completed
        self.state['effective_optimizer_updates'] += 1
        self._commit(force=True)

    def snapshot(self):
        if not self.cuda: return dict(gpu_used=False, charged_gpu_seconds=0.)
        self._commit()
        return {k:self.state[k] for k in ('authorized_gpu_uuid', 'max_gpu_seconds',
            'charged_gpu_seconds', 'max_optimizer_updates', 'effective_optimizer_updates',
            'last_completed_updates')}

    def __exit__(self, exc_type, exc, traceback):
        if not self.cuda: return False
        try:
            self._charge()
            attempt = dict(self.state['active'], ended_unix=time.time(),
                seconds=self.state['charged_gpu_seconds']-self.base_seconds,
                outcome='finished' if exc is None else type(exc).__name__,
                exceeded_time_budget=self.state['charged_gpu_seconds'] > self.state['max_gpu_seconds'])
            self.state['attempts'].append(attempt)
            self.state['active'] = None
            atomic_json(self.path, self.state)
        finally:
            self.lock.close(); self.lock = None
        return False
