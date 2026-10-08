"""Bounded continuation of the already-started user50 run; no assistant polling."""
import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root/'code'))
    from _faces_reference import atomic_json, file_sha
    state = dict(status='waiting_for_started_train_to500', max_updates=1000,
        max_gpu_seconds=28800, flow_noise_seed=0, euler_steps=50, phases=[])
    status_path = root/'first_round_status.json'
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', PYTHONUNBUFFERED='1')

    def remaining():
        ledger = json.loads((root/'gpu_budget.json').read_text())
        if ledger.get('active') is not None:
            raise RuntimeError('Previous GPU phase did not close its shared budget ledger')
        if ledger['max_gpu_seconds'] != 28800 or ledger['max_optimizer_updates'] != 1000:
            raise ValueError('First-round authorization differs from the pinned limits')
        return max(0., ledger['max_gpu_seconds']-ledger['charged_gpu_seconds'])

    def run_phase(name, argv):
        if (root/'STOP').exists() or (root/'run/STOP').exists():
            raise InterruptedError('Explicit experiment STOP')
        if remaining() <= 120: raise InterruptedError('Shared GPU time budget exhausted')
        ledger = json.loads((root/'gpu_budget.json').read_text())
        gpu = ledger['authorized_gpu_uuid']; env['CUDA_VISIBLE_DEVICES'] = gpu
        # A just-finished child can take a moment to release its CUDA context.
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader'], text=True)
        if gpu in processes: raise RuntimeError('Authorized GPU has a compute process; refusing a duplicate phase')
        entry = dict(name=name, argv=argv, started_unix=time.time(), gpu_uuid=gpu)
        state['status'] = name; state['phases'].append(entry); atomic_json(status_path,state)
        with (root/(name+'.log')).open('x') as log:
            result = subprocess.run([sys.executable,'-B',*argv], cwd=root, env=env,
                stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, ended_unix=time.time())
        atomic_json(status_path,state)
        if result.returncode: raise RuntimeError(name+' failed; no retry, fallback or cleanup')

    def checkpoint(update):
        result = json.loads((root/'run/latest.json').read_text())
        if result['completed_updates'] != update or file_sha(result['path']) != result['sha256']:
            raise ValueError('Expected immutable complete checkpoint is missing or has the wrong SHA')
        return result

    def evaluate(cp, seed, name, max_seconds):
        run_phase(name, ['code/evaluate_flow.py','--checkpoint',cp['path'],
            '--checkpoint-sha256',cp['sha256'],'--vae-checkpoint',vae,'--cache','cache',
            '--vae-baseline','vae_baseline','--output',f'evaluations/step{cp["completed_updates"]}_seed{seed}',
            '--device','cuda:0','--seed',str(seed),'--steps','50','--max-seconds',str(max_seconds),
            '--budget-file','gpu_budget.json'])
        return json.loads((root/f'evaluations/step{cp["completed_updates"]}_seed{seed}/summary.json').read_text())

    def strict_followup(cp, result):
        if not result.get('complete') or not result.get('metrics',{}).get('strict50_under_this_noise'):
            return False
        atomic_json(Path(cp['path']).with_suffix('.protect.json'),dict(reason='first_strict50_primary_seed0',checkpoint=cp))
        atomic_json(root/'best.json',cp)
        if remaining() > 120:
            evaluate(cp,1,f'evaluate_step{cp["completed_updates"]}_seed1',max(1,math.floor(remaining())))
        state['status']='strict50_primary_saved_secondary_reported_separately'
        return True

    with (root/'FIRST_ROUND.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(status_path,state)
        try:
            # Block on the actual training lock; do not repeatedly poll training metrics.
            with (root/'run/RUNNING.lock').open('a') as training_lock:
                fcntl.flock(training_lock, fcntl.LOCK_EX)
            time.sleep(3)
            trained=json.loads((root/'run/status.json').read_text())
            if not trained['complete'] or trained['completed_updates'] != 500:
                raise InterruptedError('Initial training ended before500; durable progress retained')
            vae=json.loads((root/'cache/manifest.json').read_text())['source_checkpoint']
            cp500=checkpoint(500)
            # Reserve time for the second training/evaluation stage if the first graph is very dense.
            first=evaluate(cp500,0,'evaluate_step500_seed0',min(7200,max(1,math.floor(remaining()))))
            if strict_followup(cp500,first): return
            if remaining() <= 720: raise InterruptedError('Insufficient remaining time for continuation and checkpoint reserve')
            run_phase('train_to1000',['code/train_flow.py','--config','configs/user50_recipe.json',
                '--stage-budget','configs/user50_stage_to1000.json','--budget-file','gpu_budget.json',
                '--cache','cache','--output','run','--device','cuda:0','--resume',cp500['path'],
                '--resume-sha256',cp500['sha256']])
            trained=json.loads((root/'run/status.json').read_text())
            if not trained['complete'] or trained['completed_updates'] != 1000:
                raise InterruptedError('Continuation ended before1000; complete durable progress retained')
            cp1000=checkpoint(1000)
            final=evaluate(cp1000,0,'evaluate_step1000_seed0',max(1,math.floor(remaining())))
            if strict_followup(cp1000,final): return
            if not first.get('complete') and final.get('complete') and remaining() > 120:
                first=evaluate(cp500,0,'resume_evaluate_step500_seed0',max(1,math.floor(remaining())))
            state['status']='finished' if first.get('complete') and final.get('complete') else 'evaluation_incomplete_budget_limited'
            state['evaluation_complete']=dict(step500=first.get('complete',False),step1000=final.get('complete',False))
        except InterruptedError as error:
            state.update(status='stopped_with_saved_progress',reason=str(error))
        except Exception as error:
            state.update(status='failed_no_retry',error=repr(error))
            raise
        finally:
            state['finished_unix']=time.time()
            state['gpu_budget']=json.loads((root/'gpu_budget.json').read_text())
            atomic_json(status_path,state)


if __name__ == '__main__': main()
