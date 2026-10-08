"""Run the two independent branches serially on the available A100."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
status={}
for branch in ['A','B']:
    assert not (ROOT/branch/'trace.jsonl').exists()
for branch in ['A','B']:
    status[branch]='running'
    (ROOT/'queue_status.json').write_text(json.dumps(status))
    with (ROOT/(branch+'.log')).open('w') as log:
        result=subprocess.run([sys.executable,'-u',str(ROOT/'run.py'),'--phase',branch,'--gpu','0'],stdout=log,stderr=subprocess.STDOUT)
    status[branch]='complete' if result.returncode==0 else f'failed_exit_{result.returncode}'
    (ROOT/'queue_status.json').write_text(json.dumps(status))
    if result.returncode:
        raise SystemExit(result.returncode)
