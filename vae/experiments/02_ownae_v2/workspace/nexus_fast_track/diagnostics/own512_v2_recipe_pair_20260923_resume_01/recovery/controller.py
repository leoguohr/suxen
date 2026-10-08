"""Single assigned GPU: complete queued evaluations, then alternate bounded resumes."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import subprocess
import traceback
from run_support import *

VARIANTS=['A_v1_recipe_control','B_v2_teacher_blocks']
RUNNER='/guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/_rigorpilot/skills/run-train/scripts/run_training.py'
PYTHON='/opt/conda/bin/python'


def gpu_free(gpu):
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    assert gpu not in apps,'Assigned GPU now occupied; do not interfere'
    text=subprocess.check_output(['nvidia-smi','--id='+gpu,'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    parts=[p.strip() for p in text.split(',')]
    assert parts[0]==gpu and int(parts[1])==0,text
    return text


def command(argv,tag,gpu):
    before=gpu_free(gpu)
    logs=ROOT/'recovery/job_logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',CUBLAS_WORKSPACE_CONFIG=':4096:8',RIGORPILOT_LESSONS='0',PYTHONDONTWRITEBYTECODE='1')
    write(logs/(tag+'.command.json'),dict(argv=argv,cwd=str(ROOT),started=time.time(),gpu_before=before))
    with (logs/(tag+'.log')).open('x') as stream:
        result=subprocess.run(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    write(logs/(tag+'.exit.json'),dict(returncode=result.returncode,finished=time.time()))
    print('JOB_EXIT',tag,result.returncode,flush=True)
    return result.returncode


def pending():
    return [p for p in sorted((ROOT/'queue').glob('*.json')) if p.name!='worker_complete.json' and not p.with_suffix('.done').exists()]


def mark_complete(variant,reason):
    branch=ROOT/variant;current=read(branch/'recovery_current.json')
    plan=read(OUT/'RECOVERY_PLAN.json');history=plan['branches'][variant]
    write(branch/'training_complete.json',dict(state='training_complete',completed_updates=current['step'],
        mesh_participations=current['step']*5,stop_reason=reason,final_checkpoint=current['checkpoint'],
        maximum_updates_respected=True,new_recovery_updates=current['step']-history['parent_step'],
        original_recorded_updates=history['original_recorded_updates'],
        conservative_physical_updates=history['conservative_original_updates']+current['step']-history['parent_step']))


def main():
    locks=[]
    for path in [ROOT/'recovery/controller.lock',ROOT/'queue/worker.lock']:
        lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    plan=read(OUT/'RECOVERY_PLAN.json');gpu=plan['gpu_uuid'];gpu_free(gpu)
    assert code_hashes()==plan['unchanged_runtime_code']
    try:
        while True:
            for request in pending():
                q=read(request);variant=q['variant'];branch=ROOT/variant
                if (branch/'failure.json').exists() or (branch/'evaluation_failure.json').exists():
                    write(request.with_suffix('.done'),dict(status='skipped_after_branch_failure'));continue
                rc=command([PYTHON,'-u',str(ROOT/'evaluate_checkpoint.py'),str(request)],request.stem,gpu)
                write(request.with_suffix('.done'),dict(returncode=rc,finished=time.time()))
                if rc!=0 and not (branch/'evaluation_failure.json').exists():
                    write(branch/'evaluation_failure.json',dict(step=q['step'],error='Evaluation process exited without a completion record',returncode=rc))
            # New cold-repeat jobs may have been queued by a successful target.
            if pending():continue
            candidates=[]
            for variant in VARIANTS:
                branch=ROOT/variant
                if any((branch/f).exists() for f in ['training_complete.json','failure.json','evaluation_failure.json']):continue
                step=read(branch/'recovery_current.json')['step']
                if (branch/'TARGET_REACHED.json').exists():mark_complete(variant,'target_checkpoint_reached');continue
                if step>=plan['common_logical_stop']:mark_complete(variant,'original_physical_budget_conservative_common_stop');continue
                resources=read(OUT/'RESOURCE_ACCOUNT.json')
                if resources['total_charged_gpu_seconds']>=TRAIN_GPU_SECONDS:mark_complete(variant,'resource_budget');continue
                candidates.append((step,variant))
            write(OUT/'RECOVERY_STATUS.json',dict(pid=os.getpid(),heartbeat=time.time(),state='running' if candidates else 'finalizing',
                checkpoints={v:read(ROOT/v/'recovery_current.json')['step'] for v in VARIANTS},common_stop=plan['common_logical_stop']))
            if not candidates:break
            start,variant=min(candidates);end=min((start//1000+1)*1000,plan['common_logical_stop'])
            receipt=read(ROOT/variant/'recovery_current.json')['checkpoint']
            cmd=f'{PYTHON} -u {ROOT}/recovery/train_segment.py --variant {variant} --end {end}'
            argv=[PYTHON,RUNNER,'--repo',str(ROOT),'--command',cmd,'--run-mode','resume','--lane','trusted',
                '--timeout','7200','--max-steps',str(end-start),'--resume-from',receipt['path'],
                '--checkpoint-source',receipt['path'],'--dataset','unchanged CAD50 five-mesh updates',
                '--runtime-root',str(OUT/'_runtime_recovery'/variant)]
            rc=command(argv,f'train-{variant}-{start+1:05d}-{end:05d}',gpu)
            complete=ROOT/variant/'segments'/f'{start+1:05d}-{end:05d}'/'complete.json'
            if rc!=0 or not complete.exists():
                if not (ROOT/variant/'failure.json').exists():
                    write(ROOT/variant/'failure.json',dict(error='Resume process did not complete its segment',returncode=rc,start=start,end=end))
            elif read(complete)['stop_reason']=='resource_budget':mark_complete(variant,'resource_budget')
        write(ROOT/'queue/worker_complete.json',dict(state='complete',finished=time.time()))
        sys.path.insert(0,str(ROOT/'postprocess'))
        import finish_unattended as finish
        finish.summarize()
        summary=OUT/'SUMMARY.md'
        with summary.open('a') as stream:
            stream.write('\n## Recovery accounting\n\nOriginal training was interrupted at recorded A10643 / B9464. Full states at A10000 / B9000 were restored, including Adam/RNG. Unsaved updates remain charged; one possible unlogged update per branch is reserved. The common logical stop is19356, keeping conservative total physical updates <=20000 per branch. GPU accounting also conservatively includes the disconnected interval. New recovery entrypoints preserve native model/loss/optimizer operations. See RECOVERY_PLAN.json and per-segment restore/replay audits.\n')
        finish.package()
        write(OUT/'UNATTENDED_COMPLETE.json',dict(finished=time.time(),delivery=str(ROOT/'delivery'),
            optimizer_updates_added_by_packaging=0,failed_branches=[v for v in VARIANTS if any((ROOT/v/f).exists() for f in ['failure.json','evaluation_failure.json'])]))
    except BaseException as error:
        write(OUT/'RECOVERY_CONTROLLER_FAILURE.json',dict(error=str(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
