"""Independent four-hour process-group deadline for the approved continuation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def bounded_wait(process, seconds, grace=10):
    try:
        return process.wait(timeout=seconds), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return process.returncode, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    ledger_path = root/'gpu_budget.json'
    status_path = root/'continuation_20261006_deadline.json'
    with (root/'CONTINUATION_20261006.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status_path.exists():
            raise FileExistsError('Refuse duplicate continuation launch')
        initial = json.loads(ledger_path.read_text())
        if (initial['active'] is not None or initial['charged_gpu_seconds'] != 28800
                or initial['max_gpu_seconds'] != 43200
                or initial['last_completed_updates'] != 500):
            raise ValueError('Expected approved reconciled budget and update500')
        started, wall = time.monotonic(), time.time()
        with (root/'continuation_20261006_runner.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(root/'continue_round_20261006.py'),
                '--root', str(root)], cwd=root, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status = dict(status='started', pid=os.getpid(), child_pid=child.pid,
                started_unix=wall, hard_timeout_seconds=14380, kill_grace_seconds=10,
                authorized_additional_gpu_seconds=14400, max_optimizer_updates=1000)
            atomic_json(status_path, status)
            code, expired = bounded_wait(child, 14380)
        # The runner owns a separate process group; remove any surviving child on exit.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        elapsed = time.monotonic()-started
        status.update(status='deadline_stopped' if expired else 'runner_exited',
            returncode=code, elapsed_seconds=elapsed, ended_unix=time.time())
        # Reconcile only this launch, after its children have exited and released the ledger.
        with ledger_path.with_suffix('.json.lock').open('a') as budget_lock:
            fcntl.flock(budget_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = json.loads(ledger_path.read_text())
            active = ledger.get('active')
            if active and active['started_unix'] < wall:
                raise RuntimeError('Refuse to reconcile an unrelated GPU attempt')
            ledger['charged_gpu_seconds'] = max(ledger['charged_gpu_seconds'], 28800+elapsed)
            if active:
                active.update(ended_unix=time.time(),
                    seconds=ledger['charged_gpu_seconds']-active['charged_before'],
                    outcome='deadline_or_child_exit_without_normal_cleanup',
                    accounting='Conservative wall reservation including orchestration overhead')
                ledger['attempts'].append(active)
                ledger['active'] = None
            ledger['continuation_20261006_guard'] = status
            atomic_json(ledger_path, ledger)
        atomic_json(status_path, status)


if __name__ == '__main__':
    main()
