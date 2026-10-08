"""One-time postprocessing of the already authorized bounded run, never starts training."""
import json,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
pid=json.loads((r/'resume_launcher.json').read_text())['pid']
while not (r/'run/complete.json').exists():
    if (r/'failure.txt').exists() or (r/'resume_failure.txt').exists():raise SystemExit('Training failed; no packaging or extra updates.')
    try:os.kill(pid,0)
    except ProcessLookupError:raise SystemExit('Training process ended without completion.')
    time.sleep(10)
subprocess.run(['/opt/conda/bin/python','-u',str(r/'package_results.py')],cwd=r,check=True)
