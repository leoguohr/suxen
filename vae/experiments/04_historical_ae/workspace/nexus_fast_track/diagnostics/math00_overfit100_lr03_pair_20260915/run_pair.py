"""One GPU: verify B first, then train A and B independently in sequence."""
import fcntl,json,subprocess,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def status(**d):
    p=ROOT/'pair_status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
lock=(ROOT/'pair.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not any((ROOT/b/'run/updates.jsonl').exists() for b in ['A_hold','B_lr03']),'Refuse duplicate pair launch'
stages=[('B_lr03',True),('A_hold',False),('B_lr03',False)]
for branch,verify in stages:
    folder=ROOT/branch
    status(state='baseline_check' if verify else 'training',branch=branch,queued=['A_hold','B_lr03'] if verify else ['B_lr03'] if branch=='A_hold' else [])
    logfile=folder/('precheck-console.log' if verify else 'train-console.log')
    cmd=[sys.executable,'-u','train_pair.py']+(['--verify-only'] if verify else [])
    with logfile.open('w') as f:
        proc=subprocess.Popen(cmd,cwd=folder,stdout=f,stderr=subprocess.STDOUT)
        (folder/('precheck_pid' if verify else 'train_pid')).write_text(str(proc.pid)+'\n')
        code=proc.wait()
    (folder/('precheck_exit.json' if verify else 'runner_exit.json')).write_text(json.dumps({'train_exit':code}))
    if code:
        status(state='stopped_with_error',branch=branch,stage='precheck' if verify else 'training',exit_code=code)
        sys.exit(code)
    if verify:
        saved=folder/'precheck';saved.mkdir(exist_ok=True)
        for name in ['resume_verification.json','baseline_comparison.json','eval-summary-epoch400.json','eval-epoch400.jsonl']:
            shutil.copy2(folder/'run'/name,saved/name)
    else:
        with (folder/'package-console.log').open('w') as f:
            result=subprocess.run([sys.executable,'package_results.py'],cwd=folder,stdout=f,stderr=subprocess.STDOUT)
        (folder/'package_exit.json').write_text(json.dumps({'exit_code':result.returncode}))
status(state='training_completed_packaging',branches=['A_hold','B_lr03'],updates_each=2500,total_update_each=12500)
with (ROOT/'package-console.log').open('w') as f:
    result=subprocess.run([sys.executable,'package_pair.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
status(state='completed' if result.returncode==0 else 'training_completed_package_error',branches=['A_hold','B_lr03'],updates_each=2500,total_update_each=12500,package_exit=result.returncode)
