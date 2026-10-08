"""One finite continuation; no automatic retry, parameter change or budget extension."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def status(**fields):
    path = ROOT/'launcher_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(time=time.time(), pid=os.getpid(), **fields), indent=2)+'\n')
    temporary.replace(path)


def main():
    with (ROOT/'continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((ROOT/'config.json').read_text())
        run = Path(config['run_directory'])
        if (ROOT/'STOP').exists() or (run/'STOP').exists():
            status(state='stopped'); return
        if not (run/'complete.json').exists():
            mode = 'resume' if (run/'recovery-latest.json').exists() else 'start'
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=config['gpu_uuid'], PYTHONDONTWRITEBYTECODE='1')
            command = [sys.executable, '-B', '-u', str(ROOT/'code/train_continue.py'),
                       '--config', str(ROOT/'config.json'), '--mode', mode]
            status(state='running', mode=mode, command=command, target_step=36220)
            with (ROOT/'train.stdout.log').open('a', buffering=1) as log:
                code = subprocess.call(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            if code or not (run/'complete.json').exists():
                status(state='failed' if code else 'stopped', returncode=code)
                sys.exit(code or 1)
        done = json.loads((run/'complete.json').read_text())
        assert done['completed_updates'] == 36220 and done['new_updates'] == 1000 and done['evaluations_complete']
        status(state='reporting', target_step=36220)
        code = subprocess.call([sys.executable, '-B', str(ROOT/'summarize_results.py')])
        status(state='complete' if code == 0 else 'report_failed', returncode=code, target_step=36220)
        sys.exit(code)


if __name__ == '__main__':
    main()
