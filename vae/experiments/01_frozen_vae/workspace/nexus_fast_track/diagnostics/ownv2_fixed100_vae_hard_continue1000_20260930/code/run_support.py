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


def configure(seed=0, cuda=True):
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
    if cuda: torch.cuda.manual_seed_all(seed)


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



def move_item(item, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in item.items()}
