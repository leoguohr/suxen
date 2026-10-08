"""Fixed continuation contract and CPU-only boundary/launch checks."""
import fcntl
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = '/ssdwork/guohaoran/nexus_fast_track/diagnostics'
START = 2000
LIMIT = 10000  # Astra-confirmed total budget per branch; no automatic extension.
MILESTONES = (2000, 3000, 4000, 5000, 7500, 9000, 10000)
PAIRS = (('Fourier_LN_post', 'Fourier_LN_pre'), ('XYZ_LN_post', 'XYZ_LN_pre'))
PARENTS = {
    'Fourier_LN_post': dict(directory='cad50_graph_activation_ablation_20261003', arm='V2_control',
        sha256='c7399685d3f22e4ee93df4c69d7509dc25d171d22a63a313ba55d05d11d76806'),
    'Fourier_LN_pre': dict(directory='cad50_graph_activation_ablation_20261003', arm='Graph_LN_pre',
        sha256='d8bbfdd622d7dd7b3f2ec2ea20139a8de0157071130a8d23f087c9b9722548c4'),
    'XYZ_LN_post': dict(directory='cad50_fourier_graph_factorial_20261003', arm='XYZ_LN_post',
        sha256='c10d6db589072e34fda03fd476b138bb7dcc5afec626ee1b9d7bb3332703773d'),
    'XYZ_LN_pre': dict(directory='cad50_fourier_graph_factorial_20261003', arm='XYZ_LN_pre',
        sha256='91f655f3d9059f28923af4c4869dbc7baeddb5c0a3f64a38fcd5e97be9a6f40a'),
}
ADAM = dict(lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=.01,
    amsgrad=False, foreach=True, maximize=False, capturable=False, differentiable=False, fused=None)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def info(path, expected=None):
    path = Path(path).resolve()
    value = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    require(expected is None or value['sha256'] == expected, f'Checkpoint SHA mismatch: {path}')
    return value


def parent_paths(arm, base):
    parent = PARENTS[arm]
    run = Path(base)/parent['directory']/'runs'/parent['arm']
    return run/'checkpoint-02000.pt', run/'evaluations/step-02000/evaluation.json'


def schedule(budget):
    require(budget == LIMIT, 'Authorized budget is exactly 10000 per branch')
    return list(MILESTONES[1:])


def check_cursor(cp, uids, batches, budget):
    done = cp['completed_updates']
    require(type(done) is int and START <= done <= budget, 'Checkpoint outside continuation budget')
    epoch, cursor = divmod(done, 10)
    require((cp['next_epoch'], cp['next_batch']) == (epoch, cursor), 'Epoch cursor mismatch')
    require(cp['negative_seed'] == 0, 'Negative sampling seed mismatch')
    expected = {u: epoch for u in uids}
    for group in batches(uids, epoch)[:cursor]:
        for uid in group:
            expected[uid] += 1
    require(cp['participation'] == expected, 'Per-UID participation mismatch')
    require(sum(expected.values()) == 5*done, 'Participation total mismatch')


def check_log(path, done):
    expected = START+1
    if Path(path).exists():
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                require(row['update_after'] == expected, 'Noncontiguous/duplicate update log')
                require(row['update_before'] == expected-1, 'Invalid before/after update boundary')
                expected += 1
    require(expected-1 == done, 'Log/checkpoint boundary differs; explicit recovery required, no replay/reset')


def output_folder(path):
    path = Path(path).resolve()
    require(ROOT in path.parents, 'Output must stay inside this new experiment directory')
    return path


def acquire(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f'Already running/locked: {path}') from None
    return stream


def verify_code():
    expected = read(ROOT/'CODE_SHA256.json')
    actual = {p.name: sha(p) for p in sorted(ROOT.glob('*.py'))}
    require(actual == expected, 'Runtime code differs from reviewed CODE_SHA256.json')
    return actual


def completed_run(folder, budget, arm):
    path = Path(folder)/'complete.json'
    if not path.exists():
        return False
    complete = read(path)
    cfg = read(Path(folder)/'config.json')
    require(cfg['arm'] == arm and cfg['max_updates'] == budget, 'Completed run has a different contract')
    require(complete['completed_updates'] == budget and complete['state'] == 'complete', 'Invalid completion marker')
    info(complete['final_checkpoint']['path'], complete['final_checkpoint']['sha256'])
    result = read(complete['final_evaluation_path'])
    require(result['complete'] and len(result['meshes']) == 50, 'Final complete CAD50 evaluation missing')
    require(result['checkpoint']['sha256'] == complete['final_checkpoint']['sha256'], 'Final evaluation/checkpoint mismatch')
    require(complete['next_epoch'] == budget//10 and complete['next_batch'] == 0, 'Final cursor mismatch')
    return True


def launch_action(folder, resume, budget, arm):
    if completed_run(folder, budget, arm):
        require(resume is None, 'Cannot resume an already completed branch')
        return 'skip_complete'
    existing = (Path(folder)/'config.json').exists()
    require(not existing or resume is not None, f'{arm}: existing run requires explicit --resume-map')
    require(existing or resume is None, f'{arm}: resume requires an existing independent run config')
    if resume:
        require(set(resume) == {'path', 'sha256'}, 'Each resume entry requires exactly path and sha256')
    return 'resume' if existing else 'start_from_parent'
