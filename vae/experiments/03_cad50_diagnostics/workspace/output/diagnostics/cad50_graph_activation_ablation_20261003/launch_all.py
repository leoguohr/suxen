"""Run four independent arms on the assigned A100; no gradient synchronization."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from support import MODES, read, write

GPU = 'GPU-4db14efb-13b4-ec8b-bcba-670816c5837f'


def main(source):
    root = Path(__file__).resolve().parent
    lock = (root/'experiment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root/'LAUNCH.json').exists(), 'Refuse duplicate launcher'
    assert (root/'INITIALIZATION.json').exists()
    devices = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True)
    active = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    assert GPU in devices and not active.strip(), 'Assigned card has a process; do not interfere'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    jobs = {}
    for arm in MODES:
        log = (root/f'{arm}.console.log').open('x')
        command = [sys.executable,'-B','-u',str(root/'train.py'),'--arm',arm,'--source',source]
        proc = subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs[arm] = dict(process=proc, log=log, command=command)
    write(root/'LAUNCH.json',dict(gpu_uuid=GPU, devices=devices, started=time.time(),
        jobs={arm:dict(pid=j['process'].pid,command=j['command']) for arm,j in jobs.items()},
        independent_processes=True, shared_gradient_synchronization=False, updates_per_arm=2000))
    while any(j['process'].poll() is None for j in jobs.values()):
        write(root/'repro_outputs/status.json',dict(state='running',
            arms={arm:dict(pid=j['process'].pid,exit_code=j['process'].poll(),
                progress=read(root/'runs'/arm/'status.json') if (root/'runs'/arm/'status.json').exists() else None)
                for arm,j in jobs.items()}))
        time.sleep(15)
    exits={arm:j['process'].returncode for arm,j in jobs.items()}
    for j in jobs.values(): j['log'].close()
    for arm,exit_code in list(exits.items()):
        if exit_code==0:
            with (root/f'{arm}.cold.log').open('x') as log:
                cold=subprocess.run([sys.executable,'-B','-u',str(root/'verify_final.py'),
                    '--arm',arm,'--source',source],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            exits[arm]=cold.returncode
    subprocess.run([sys.executable,str(root/'package_results.py')],cwd=root,check=True)
    write(root/'repro_outputs/status.json',dict(state='complete' if not any(exits.values()) else 'failed',
        exit_codes=exits, finished=time.time()))
    print('EXPERIMENT_FINISHED',exits,flush=True)
    if any(exits.values()): raise SystemExit(1)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    main(p.parse_args().source)
