"""Launch only the two explicitly reserved, idle CAD GPUs. No other task changes."""
import fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FRESH=ROOT.parent/'teacher_cad50_fresh512_20260921'
GPUS={'A_soft4_full':('GPU-c3c69625-6ea3-026c-c731-5c7dd735e1b7','full'),
      'B_soft4_stopgrad':('GPU-0cb18edf-c41d-02e8-3b12-14d55a283546','stopgrad')}
def write(p,obj):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(p)
def main():
    lock=(ROOT/'pair.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'launch.json').exists(),'Already launched; do not duplicate'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    for uuid,mode in GPUS.values():
        assert uuid not in processes,'GPU already in use: '+uuid
        row=next(line for line in gpu.splitlines() if line.startswith(uuid))
        assert int(row.split(',')[2].strip())<1000,'Unexpected GPU allocation'
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,str(ROOT/'test_soft4.py')],env=env,check=True)
    procs={}
    for name,(uuid,mode) in GPUS.items():
        out=ROOT/name;out.mkdir(exist_ok=False)
        for filename in ['runtime.py','effective_loss_and_scoring.py','loader.py','evaluate.py','construction_args.json','helpers.py','train.py']:
            shutil.copy2(ROOT/filename,out/filename)
        (out/'data').symlink_to(FRESH/'data',target_is_directory=True)
        (out/'pools').symlink_to(FRESH/'pools',target_is_directory=True)
        shutil.copy2(FRESH/'pool_manifest.json',out/'pool_manifest.json')
        stream=(out/'execution.log').open('x',buffering=1)
        procs[name]=subprocess.Popen([sys.executable,'-u',str(out/'train.py'),'--mode',mode],cwd=out,
            env=dict(env,CUDA_VISIBLE_DEVICES=uuid),stdout=stream,stderr=subprocess.STDOUT)
    write(ROOT/'launch.json',dict(pids={k:p.pid for k,p in procs.items()},gpus=GPUS,gpu_state_before=gpu,compute_processes_before=processes))
    while not all((ROOT/k/'ready.json').exists() for k in GPUS):
        if any(p.poll() is not None for p in procs.values()):
            write(ROOT/'barrier_failure.json',dict(reason='A worker stopped before matched baseline verification'))
            raise RuntimeError('Pre-update worker failed; see branch execution.log')
        time.sleep(3)
    a,b=[json.loads((ROOT/k/'ready.json').read_text()) for k in GPUS]
    if a!=b:
        write(ROOT/'barrier_failure.json',dict(reason='Cross-device baseline or full-gradient mismatch',A=a,B=b))
        raise RuntimeError('Baseline/device discrepancy: no training released')
    write(ROOT/'release.json',dict(passed=True,baseline_prediction_arrays_match_parent=True,full_gradient_cross_gpu_bitwise_equal=True,
        device_check=a['device_check'],source_sha256=a['source_sha256']))
    results={k:p.wait() for k,p in procs.items()}
    write(ROOT/'process_exit.json',results)
    assert set(results.values())=={0},results
    write(ROOT/'pair_complete.json',dict(state='training_complete',new_updates_each=100,completed_updates_each=2600,stopped_at_budget=True))
    print('PAIR_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'launcher_failure.json',dict(traceback=traceback.format_exc()))
        raise
