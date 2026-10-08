"""Resume paused arm and start pending arm concurrently on the same authorized GPU."""
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
    assert json.loads((root/'performance/EQUIVALENCE.json').read_text())['all_gradients_bitwise_equal']
    boundary = json.loads((root/'performance/BOUNDARY.json').read_text())
    pending = root/'runs/XYZ_LN_pre/config.json'
    assert not pending.exists(), 'Pending arm already started; do not duplicate it'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid)
    children = {}
    for arm in ('XYZ_LN_post', 'XYZ_LN_pre'):
        args = [sys.executable, '-B', '-u', 'train_fast.py', '--arm', arm, '--source', source]
        if arm == 'XYZ_LN_post':
            args += ['--resume', boundary['checkpoint']['path']]
        with (root/f'{arm}.fast.log').open('x') as log:
            children[arm] = subprocess.Popen(args, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
    (root/'performance/PARALLEL_LAUNCH.json').write_text(json.dumps(dict(started=time.time(),
          pids={a:p.pid for a,p in children.items()}, gpu_uuid=gpu_uuid,
          update_limit_per_arm=2000, resumed_post=boundary['completed_updates'],
          optimizer_and_rng_restored=True, allocator_fraction_each=.44), indent=2))
    with (root/'performance/gpu_usage.jsonl').open('x', buffering=1) as log:
        while any(p.poll() is None for p in children.values()):
            metrics = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True).strip()
            log.write(json.dumps(dict(time=time.time(), gpu=metrics,
                      returncodes={a:p.poll() for a,p in children.items()}))+'\n')
            time.sleep(10)
    codes = {a:p.wait() for a,p in children.items()}
    (root/'performance/EXIT.json').write_text(json.dumps(codes))
    if any(codes.values()):
        raise SystemExit('An arm failed; inspect logs. No automatic retry.')
    subprocess.run([sys.executable, '-B', 'compare_results.py', '--old-code', old_code],
                   cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True)
    (root/'DONE.json').write_text(json.dumps(dict(completed=time.time(), updates_per_arm=2000,
            note='No extension; both optimizer trajectories stayed within original budgets.'), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--old-code', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    main(args.source, args.old_code, args.gpu_uuid)
