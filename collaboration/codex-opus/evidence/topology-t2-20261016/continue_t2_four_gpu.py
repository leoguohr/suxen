"""Continue the existing T2 cooldown on four GPUs within its original deadline."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from run_t2 import OUT, CODE, PY, CACHE, VAE, VEVAL, VDATA, DEADLINE, write

RUN = OUT / 'B_cooldown_4gpu'
lock = threading.Lock()
active = {}
identities = {}
results = json.loads((OUT / 'stage_status.json').read_text())


def event(value):
    value['time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with lock:
        with (OUT / 'events.jsonl').open('a') as f:
            f.write(json.dumps(value) + '\n')
            f.flush()


def process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return int(fields[1]), fields[19]  # parent PID and start-time ticks
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_descendants(roots):
    table = {int(p.name): process_identity(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    owned = set(roots)
    while True:
        children = {pid for pid, value in table.items() if value and value[0] in owned}
        if children <= owned:
            break
        owned.update(children)
    return {pid: table[pid][1] for pid in owned if table.get(pid)}


def deadline_guard():
    reserved = False
    while True:
        remaining = DEADLINE - time.time()
        if remaining <= 600 and not reserved:
            (RUN / 'STOP').touch()
            event(dict(event='four_gpu_budget_checkpoint_reserve', remaining_seconds=max(0, remaining)))
            reserved = True
        with lock:
            owned = list(active.values())
        identities.update(owned_descendants([p.pid for p in owned]))
        if remaining <= 10:
            for pid, start in identities.items():
                current = process_identity(pid)
                if current and current[1] == start:
                    try:
                        os.kill(pid, signal.SIGTERM if remaining > 0 else signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        if remaining <= 0:
            event(dict(event='hard_deadline'))
            return
        time.sleep(min(2, remaining))


def run(name, gpus, argv):
    if time.time() >= DEADLINE - 600:
        raise RuntimeError('Insufficient authorized time to start ' + name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
               PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', OMP_NUM_THREADS='8')
    started = time.time()
    with (OUT / 'logs' / (name + '.log')).open('x') as log:
        proc = subprocess.Popen([PY, '-u', *argv], cwd=CODE, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        with lock:
            active[name] = proc
        event(dict(event='started', stage=name, gpus=gpus, pid=proc.pid, argv=[PY, '-u', *argv]))
        rc = proc.wait()
    seconds = time.time() - started
    row = dict(stage=name, gpus=gpus, returncode=rc, seconds=seconds,
               gpu_hours_upper_bound=len(gpus) * seconds / 3600)
    with lock:
        active.pop(name, None)
        results[name] = row
        write(OUT / 'stage_status.json', results)
    event(dict(event='finished', **row))
    if rc:
        raise RuntimeError(f'{name} exited {rc}; preserved state in {RUN}')


def cascade(weights, checkpoint):
    run('B_cooldown_' + weights, [0 if weights == 'raw' else 1],
        ['cascade.py', '--mode', 'cascade', '--cache', CACHE, '--vae', VAE,
         '--checkpoint', str(checkpoint), '--weights', weights, '--flow-seeds', '0,1',
         '--vertex-data', VDATA, '--vertex-eval', VEVAL, '--vertex-seeds', '97029000,97029001',
         '--output', str(OUT / ('B_cooldown_' + weights))])


def main():
    guard = (OUT / 'runner.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'BATCH_COMPLETE.json').exists()
    assert not (OUT / 'BATCH_ERROR.json').exists()
    assert json.loads((OUT / 'four_gpu_validation.json').read_text())['passed']
    validation = json.loads((OUT / 'four_gpu_parent_validation.json').read_text())
    assert validation['passed'] and validation['adam_step'] == 4000 + validation['update']
    assert json.loads((OUT / 'B_cooldown/status.json').read_text())['state'] == 'interrupted'
    args = json.loads((OUT / 'four_gpu_launch_args.json').read_text())
    assert args[args.index('--output') + 1] == str(RUN)
    assert args[args.index('--max-updates') + 1] == '1000'
    gpu_processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                            '--format=csv,noheader'], text=True).strip()
    assert not gpu_processes, 'GPU processes present; refusing to interfere'
    threading.Thread(target=deadline_guard, daemon=True).start()
    run('B_cooldown_4gpu', [0, 1, 2, 3],
        ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', *args])
    pointer = json.loads((RUN / 'final.json').read_text())
    checkpoint = (RUN / pointer['file']).resolve()
    assert checkpoint.parent == RUN.resolve() and pointer['update'] == 1000 and pointer['final_stage']
    digest = hashlib.sha256()
    with checkpoint.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    assert checkpoint.stat().st_size == pointer['bytes'] and digest.hexdigest() == pointer['sha256']
    write(RUN / 'final.sha256.json', dict(path=str(checkpoint), bytes=pointer['bytes'],
                                      sha256=digest.hexdigest(), update=pointer['update']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(cascade, weights, checkpoint) for weights in ('raw', 'ema')]
        for task in tasks:
            task.result()
    write(OUT / 'BATCH_COMPLETE.json', dict(complete=True, stages=results, time_unix=time.time(),
                                          deadline_unix=DEADLINE, cooldown_run=str(RUN)))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        identities.update(owned_descendants([p.pid for p in active.values()]))
        for pid, start in list(identities.items()):
            current = process_identity(pid)
            if current and current[1] == start:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        event(dict(event='four_gpu_batch_error', error=repr(exc)))
        write(OUT / 'BATCH_ERROR.json', dict(error=repr(exc), stages=results,
                                           cooldown_run=str(RUN), time_unix=time.time()))
        raise
