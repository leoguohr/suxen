"""Export, execute one bounded shared-head run, verify, and package."""
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    with (ROOT/'job.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/'run').exists() and not (ROOT/'snapshots').exists()
        for stage,script in [('export','export_hidden.py'),('training','probe.py'),('verification','verify.py'),('packaging','package_results.py')]:
            (ROOT/'status.json').write_text(json.dumps(dict(stage=stage,time=time.time()))+'\n')
            with (ROOT/(stage+'.log')).open('xb') as log:
                subprocess.run([sys.executable,'-u',str(ROOT/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        (ROOT/'status.json').write_text(json.dumps(dict(stage='complete',head_updates=2000,network_updates=0,time=time.time()))+'\n')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        (ROOT/'failure.json').write_text(json.dumps(dict(error=repr(exc),time=time.time()))+'\n')
        raise
