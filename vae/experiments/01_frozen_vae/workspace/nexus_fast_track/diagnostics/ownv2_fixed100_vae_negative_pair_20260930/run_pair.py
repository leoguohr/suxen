"""Run A then B on the assigned GPU; finite budgets and fail-closed process handling."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def write_status(value):
    path=ROOT/'pair_status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(time=time.time(),pid=os.getpid(),**value),indent=2)+'\n');temp.replace(path)


def main():
    with (ROOT/'pair.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for branch in ('A_uniform','B_hard'):
            if (ROOT/'STOP').exists():
                write_status(dict(state='stopped',branch=branch));return
            config_path=ROOT/f'{branch}.json'
            config=json.loads(config_path.read_text());run=Path(config['run_directory'])
            if (run/'complete.json').exists():
                complete=json.loads((run/'complete.json').read_text())
                assert complete['completed_updates']==35220 and complete['new_updates']==500 and complete['evaluations_complete']
                assert json.loads((run/'config.json').read_text())==config
                continue
            mode='resume' if (run/'recovery-latest.json').exists() else 'start'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=config['gpu_uuid'],PYTHONDONTWRITEBYTECODE='1')
            command=[sys.executable,'-B','-u',str(ROOT/'code/train_pair.py'),'--config',str(config_path),'--mode',mode]
            write_status(dict(state='running',branch=branch,mode=mode,command=command))
            with (ROOT/f'{branch}.stdout.log').open('a',buffering=1) as log:
                code=subprocess.call(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            if code or not (run/'complete.json').exists():
                write_status(dict(state='failed' if code else 'stopped',branch=branch,returncode=code))
                sys.exit(code or 1)
        write_status(dict(state='comparing'))
        code=subprocess.call([sys.executable,'-B',str(ROOT/'compare_pair.py')])
        write_status(dict(state='complete' if code==0 else 'comparison_failed',returncode=code))
        sys.exit(code)


if __name__=='__main__':main()
