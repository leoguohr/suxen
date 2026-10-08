"""Mine once, run bounded paired continuation, verify and package. Never auto-retry."""
import fcntl,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(name,data):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
lock=(ROOT/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'run/updates.jsonl').exists(),'Refuse duplicate continuation'
for script,log,record,stage in [
    ('mine.py','mining-console.log','mining_exit.json','mining_fixed_parent'),
    ('train_hardneg.py','train-console.log','runner_exit.json','baseline_check_then_training'),
    ('verify_results.py','verification-console.log','verification_exit.json','verifying_saved_results'),
    ('compare_control.py','comparison-console.log','comparison_exit.json','comparing_completed_control'),
    ('package_results.py','package-console.log','package_exit.json','packaging')]:
    write('job_status.json',dict(state=stage,start_update=12500,end_update=15000))
    with (ROOT/log).open('w') as f:
        p=subprocess.Popen([sys.executable,'-u',script],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        write('process.json',dict(stage=stage,pid=p.pid));code=p.wait()
    write(record,{'train_exit' if script=='train_hardneg.py' else 'exit_code':code})
    if code:
        write('job_status.json',dict(state='stopped_with_error',stage=stage,exit_code=code))
        sys.exit(code)
write('job_status.json',dict(state='completed',additional_updates=2500,end_epoch=600,end_update=15000,package='evaluation_package.zip'))
