"""Three fixed independent probes, bounded at 2000 updates each, then verify/package."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
UIDS=['nexus_2k_000446','nexus_2k_001093','nexus_2k_000898']

def write(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')

def main():
    with (ROOT/'job.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        exported=json.loads((ROOT/'export_complete.json').read_text())
        assert exported['model_unchanged'] and exported['checkpoint_unchanged']
        assert [r['uid'] for r in exported['records']]==UIDS
        assert not (ROOT/'runs').exists(),'Do not replay or overwrite previous updates'
        (ROOT/'runs').mkdir()
        for uid in UIDS:
            write('status.json',dict(stage='optimizing_free_embedding',uid=uid,time=time.time()))
            snap=ROOT/'snapshots'/uid;out=ROOT/'runs'/uid
            with (ROOT/'runs'/f'{uid}.log').open('xb') as log:
                subprocess.run([sys.executable,'-u',str(ROOT/'run_probe.py'),'--snapshot',str(snap),'--output',str(out)],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'-u',str(ROOT/'verify_saved.py'),'--snapshot',str(snap),'--run',str(out)],
                               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write('status.json',dict(stage='packaging',time=time.time()))
        subprocess.run([sys.executable,'-u',str(ROOT/'package_results.py')],cwd=ROOT,check=True)
        write('status.json',dict(stage='complete',updates_per_uid=2000,network_updates=0,time=time.time()))

if __name__=='__main__':
    try:main()
    except Exception as e:
        write('failure.json',dict(error=repr(e),time=time.time()))
        raise
