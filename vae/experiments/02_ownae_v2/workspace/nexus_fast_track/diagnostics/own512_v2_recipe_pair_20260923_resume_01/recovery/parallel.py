"""Move only orchestration to two trainers and one evaluator at saved boundaries."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'recovery'))
import argparse
import signal
import traceback
from run_support import *
from controller import command, pending, mark_complete, gpu_free, RUNNER, PYTHON, VARIANTS


def state(role,phase,**extra):
    write(OUT/f'PARALLEL_{role}_STATUS.json',dict(pid=os.getpid(),host=os.uname().nodename,
        heartbeat=time.time(),state=phase,**extra))


def branch_terminal(variant):
    return any((ROOT/variant/name).exists() for name in ['training_complete.json','failure.json','evaluation_failure.json'])


def run_branch(role):
    variant=VARIANTS[0 if role=='A' else 1];branch=ROOT/variant
    handoff=read(OUT/'PARALLEL_HANDOFF.json');plan=read(OUT/'RECOVERY_PLAN.json')
    assert os.uname().nodename==handoff['new_host']
    gpu=handoff['gpu_'+role]
    with (ROOT/'recovery'/f'parallel-{role}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if role=='A' and read(OUT/'PARALLEL_HANDOFF.json')['state']!='parallel_released':
                state(role,'waiting_for_current_A_segment_to_finish');time.sleep(3);continue
            if branch_terminal(variant):break
            current=read(branch/'recovery_current.json');start=current['step']
            evaluated=branch/'evaluations'/f'eval-{start:05d}.json'
            if not evaluated.exists():
                state(role,'waiting_for_full_evaluation',checkpoint_step=start);time.sleep(3);continue
            assert read(evaluated)['complete']
            if (branch/'TARGET_REACHED.json').exists():mark_complete(variant,'target_checkpoint_reached');break
            if start>=plan['common_logical_stop']:mark_complete(variant,'original_physical_budget_conservative_common_stop');break
            if read(OUT/'RESOURCE_ACCOUNT.json')['total_charged_gpu_seconds']>=TRAIN_GPU_SECONDS:
                mark_complete(variant,'resource_budget');break
            end=min((start//1000+1)*1000,plan['common_logical_stop'])
            receipt=current['checkpoint']
            cmd=f'{PYTHON} -u {ROOT}/recovery/train_segment.py --variant {variant} --end {end}'
            argv=[PYTHON,RUNNER,'--repo',str(ROOT),'--command',cmd,'--run-mode','resume','--lane','trusted',
                '--timeout','7200','--max-steps',str(end-start),'--resume-from',receipt['path'],
                '--checkpoint-source',receipt['path'],'--dataset','unchanged CAD50 five-mesh updates',
                '--runtime-root',str(OUT/'_runtime_recovery'/variant)]
            state(role,'training_segment',start=start,end=end,gpu_uuid=gpu)
            rc=command(argv,f'train-{variant}-{start+1:05d}-{end:05d}',gpu)
            complete=branch/'segments'/f'{start+1:05d}-{end:05d}'/'complete.json'
            if rc!=0 or not complete.exists():
                if not (branch/'failure.json').exists():write(branch/'failure.json',dict(error='Parallel resume segment failed',returncode=rc,start=start,end=end))
                break
            if read(complete)['stop_reason']=='resource_budget':mark_complete(variant,'resource_budget');break
        state(role,'terminal',checkpoint=read(branch/'recovery_current.json'))


def release_original_controller():
    handoff=read(OUT/'PARALLEL_HANDOFF.json')
    assert os.uname().nodename==handoff['old_host']
    if handoff['state']=='parallel_released':return
    complete=ROOT/VARIANTS[0]/'segments'/handoff['waiting_for_A_segment']/'complete.json'
    while True:
        runtime=[(p,read(p)) for p in (OUT/'_runtime_recovery'/VARIANTS[0]).glob('*/state.json')]
        successful=[(p,x) for p,x in runtime if x['status']=='success' and x.get('returncode')==0]
        if complete.exists() and successful:
            assert read(complete)['end']==11000 and read(complete)['stop_reason']=='segment_complete'
            assert read(ROOT/VARIANTS[0]/'recovery_current.json')['step']==11000
            gpu_free(handoff['gpu_eval'])
            break
        state('eval','waiting_for_saved_A11000_and_successful_runtime');time.sleep(3)
    pid=handoff['old_controller_pid']
    expected=[b'/opt/conda/bin/python',b'-u',str(ROOT/'recovery/controller.py').encode()]
    assert Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:3]==expected
    # Only the stopped CPU scheduler is terminated. Its training child has
    # already saved full state and completed successfully under run-train.
    os.kill(pid,signal.SIGTERM)
    try:os.kill(pid,signal.SIGCONT)
    except ProcessLookupError:pass
    deadline=time.monotonic()+15
    while Path(f'/proc/{pid}/stat').exists():
        if Path(f'/proc/{pid}/stat').read_text().split()[2]=='Z':break
        assert time.monotonic()<deadline,'Old scheduler did not exit'
        time.sleep(.2)
    handoff.update(state='parallel_released',released=time.time(),
        old_cpu_scheduler_terminated_after_successful_checkpoint=True,
        successful_A_runtime_paths=[str(p) for p,x in successful],
        A_handoff_checkpoint=read(ROOT/VARIANTS[0]/'recovery_current.json'),
        additional_optimizer_updates_for_handoff=0)
    write(OUT/'PARALLEL_HANDOFF.json',handoff)


def run_evaluator():
    release_original_controller()
    handoff=read(OUT/'PARALLEL_HANDOFF.json');gpu=handoff['gpu_eval']
    with (ROOT/'queue/worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            jobs=pending()
            if not jobs:
                if all(branch_terminal(v) for v in VARIANTS):break
                state('eval','waiting_for_checkpoint');time.sleep(3);continue
            for request in jobs:
                q=read(request);variant=q['variant'];branch=ROOT/variant
                if any((branch/f).exists() for f in ['failure.json','evaluation_failure.json']):
                    write(request.with_suffix('.done'),dict(status='skipped_after_branch_failure'));continue
                state('eval','evaluating',variant=variant,step=q['step'],gpu_uuid=gpu)
                rc=command([PYTHON,'-u',str(ROOT/'evaluate_checkpoint.py'),str(request)],request.stem,gpu)
                write(request.with_suffix('.done'),dict(returncode=rc,finished=time.time()))
                if rc!=0 and not (branch/'evaluation_failure.json').exists():
                    write(branch/'evaluation_failure.json',dict(error='Parallel evaluation failed',returncode=rc,step=q['step']))
        write(ROOT/'queue/worker_complete.json',dict(state='complete',finished=time.time()))
        state('eval','packaging')
        sys.path.insert(0,str(ROOT/'postprocess'))
        import finish_unattended as finish
        finish.summarize()
        with (OUT/'SUMMARY.md').open('a') as stream:
            stream.write('\n## Recovery and device handoff\n\nOriginal run interruption lost unsaved tails A643/B464; one extra potentially unlogged step each remains reserved. Complete model/Adam/RNG were restored at A10000/B9000. Shared effective stop19356 preserves the original physical update limits. The later two-card acceleration moved A at a successfully saved11000 checkpoint and B at10000, with zero extra/lost updates for this handoff. The original single A100 performs all full evaluations; only CPU scheduling changed. GPU accounting retains prior conservative charges and the original24 GPU-hour cap. See RECOVERY_PLAN.json, PARALLEL_HANDOFF.json and segment audits.\n')
        finish.package()
        write(OUT/'UNATTENDED_COMPLETE.json',dict(finished=time.time(),delivery=str(ROOT/'delivery'),
            failed_branches=[v for v in VARIANTS if any((ROOT/v/f).exists() for f in ['failure.json','evaluation_failure.json'])],
            optimizer_updates_added_by_packaging=0))
        state('eval','complete')


def main(role):
    try:
        if role=='eval':run_evaluator()
        else:run_branch(role)
    except BaseException as error:
        value=dict(role=role,error=str(error),traceback=traceback.format_exc())
        write(OUT/f'PARALLEL_{role}_FAILURE.json',value)
        if role in ['A','B']:
            branch=ROOT/VARIANTS[0 if role=='A' else 1]
            if not (branch/'failure.json').exists():write(branch/'failure.json',value)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['A','B','eval'],required=True)
    main(parser.parse_args().role)
