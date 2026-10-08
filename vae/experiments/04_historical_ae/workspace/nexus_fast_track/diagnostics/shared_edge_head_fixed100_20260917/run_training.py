import subprocess,sys,fcntl
from pathlib import Path
r=Path(__file__).resolve().parent
lock=(r/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (r/'benchmark.json').exists() and (r/'training_plan.json').exists()
for name in ['train_head.py','verify.py']:
    subprocess.run([sys.executable,'-u',str(r/name)],cwd=r,check=True)
