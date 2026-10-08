"""One-time packaging after the already launched bounded process exits."""
import json,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
pid=json.loads((r/'launcher.json').read_text())['pid']
while not (r/'run/complete.json').exists():
    if not Path(f'/proc/{pid}').exists():raise SystemExit('Training exited without complete.json; no result package claimed.')
    time.sleep(10)
subprocess.run(['/opt/conda/bin/python','-u',str(r/'package_results.py')],cwd=r,check=True)
