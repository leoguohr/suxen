"""CPU coordinator: each queued checkpoint gets a new third-GPU process."""
import fcntl
import os
from pathlib import Path
import subprocess
import time
import json

ROOT=Path(__file__).resolve().parent
VARIANTS=('A_v1_recipe_control','B_v2_teacher_blocks')


def main():
    queue=ROOT/'queue';queue.mkdir(exist_ok=True)
    lock=(queue/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        pending=[p for p in sorted(queue.glob('*.json')) if not p.with_suffix('.done').exists()]
        if not pending:
            if all((ROOT/v/'training_complete.json').exists() or (ROOT/v/'failure.json').exists() for v in VARIANTS):break
            time.sleep(2);continue
        request=pending[0];record=json.loads(request.read_text())
        log=queue/(request.stem+'.log')
        with log.open('x') as stream:
            result=subprocess.run(['/opt/conda/bin/python','-u',str(ROOT/'evaluate_checkpoint.py'),str(request)],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        request.with_suffix('.done').write_text(json.dumps(dict(returncode=result.returncode,finished=time.time()))+'\n')
        print('EVALUATION_JOB',request.name,result.returncode,flush=True)
    (queue/'worker_complete.json').write_text(json.dumps(dict(state='complete',finished=time.time()))+'\n')


if __name__=='__main__':main()
