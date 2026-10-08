"""One GPU, two independent bounded runs; no auto-extension or retries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(source, old_code, gpu_uuid):
    root = Path(__file__).resolve().parent
    lock = (root/'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root/'LAUNCH.json').exists(), 'Already submitted; inspect status before any resume'
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    assert gpu_uuid not in apps, 'Authorized GPU already has a compute process; do not share it'
    (root/'LAUNCH.json').write_text(json.dumps(dict(started=time.time(), gpu_uuid=gpu_uuid,
          arms=['XYZ_LN_post', 'XYZ_LN_pre'], updates_each=2000, source=source, old_code=old_code), indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid)
    tasks = [
        ('cpu', ['test_factorial_cpu.py', '--old-code', old_code, '--source', source],
         dict(env, CUDA_VISIBLE_DEVICES='')),
        ('control_gate', ['verify_controls.py', '--old-code', old_code, '--source', source], env),
        *[(arm, ['train.py', '--arm', arm, '--source', source], env)
          for arm in ('XYZ_LN_post', 'XYZ_LN_pre')],
        ('comparison', ['compare_results.py', '--old-code', old_code], dict(env, CUDA_VISIBLE_DEVICES='')),
    ]
    for name, arguments, task_env in tasks:
        print('START', name, time.time(), flush=True)
        with (root/f'{name}.console.log').open('x') as log:
            result = subprocess.run([sys.executable, '-B', '-u', *arguments], cwd=root,
                                    env=task_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (root/'STOPPED.json').write_text(json.dumps(dict(task=name, returncode=result.returncode,
                 reason='Task failed; no retry, no protocol change', stopped=time.time()), indent=2))
            raise SystemExit(result.returncode)
        print('DONE', name, time.time(), flush=True)
    (root/'DONE.json').write_text(json.dumps(dict(completed=time.time(), new_updates=4000,
          note='Each model has 2000 updates; no additional training authorized by this runner'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--old-code', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    main(args.source, args.old_code, args.gpu_uuid)
