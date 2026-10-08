"""Shared numerical policy, durable records and conservative GPU accounting."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
import fcntl
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'repro_outputs'
CHECKS = [0,500] + list(range(1000,20001,1000))
MAX_GPU_SECONDS = 24*3600
TRAIN_GPU_SECONDS = 23*3600  # Reserve one GPU-hour for final saves/evaluation.


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def configure(seed=0):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision('highest')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['torch'])
    if value['cuda']: torch.cuda.set_rng_state_all(value['cuda'])


def tensor_hash(state):
    digest = hashlib.sha256()
    for name,value in state.items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_torch(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def code_hashes():
    return {p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}


class ResourceAccount:
    """Charge assigned-process wall time, including waits, as GPU time."""
    def __init__(self, name):
        self.name = name
        self.update('start')

    def update(self, action='heartbeat'):
        OUT.mkdir(exist_ok=True)
        with (OUT/'resource_account.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = OUT/'RESOURCE_ACCOUNT.json'
            state = read(path) if path.exists() else dict(limit_gpu_seconds=MAX_GPU_SECONDS, jobs={})
            now = time.time()
            if action == 'start':
                assert self.name not in state['jobs'], 'Duplicate GPU job'
                state['jobs'][self.name] = dict(started=now, status='active', pid=os.getpid(),
                    gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), node=os.uname().nodename)
            row = state['jobs'][self.name]
            row['heartbeat'] = now
            if action == 'finish': row.update(status='finished', finished=now)
            for job in state['jobs'].values():
                job['charged_seconds'] = max(0,job.get('finished',now)-job['started'])
            state['total_charged_gpu_seconds'] = sum(j['charged_seconds'] for j in state['jobs'].values())
            state['updated'] = now
            write(path, state)
            return state['total_charged_gpu_seconds']


def move_item(item, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in item.items()}
