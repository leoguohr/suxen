"""Start only the authorized Hard4 branch, via the installed RigorPilot runtime."""
import fcntl,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
GPU='GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351'
RUNNER=ROOT.parent/'decoder_last3_trainability_pair_A500_20260921/_rigorpilot/skills/run-train/scripts/run_training.py'
SOURCE=ROOT.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt'
lock=(ROOT/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'repro_outputs/launch.json').exists(),'Refuse duplicate launch'
assert json.loads((ROOT/'repro_outputs/CONTROL_REUSE_AUDIT.json').read_text())['passed']
assert json.loads((ROOT/'H_paper_hard4/hard4_test.json').read_text())['passed']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert GPU not in processes,'Assigned GPU currently in use; do not preempt'
row=next(x for x in gpu.splitlines() if GPU in x)
assert row.split(',')[0].strip()=='1' and int(row.split(',')[3].strip())<512
command=f'{sys.executable} -u {ROOT}/H_paper_hard4/train.py --mode paper_hard4'
argv=[sys.executable,str(RUNNER),'--repo',str(ROOT),'--command',command,
      '--run-mode','resume','--lane','trusted','--timeout','7200',
      '--dataset','fixed teacher CAD50 original FP32 meshes and immutable base Face pool',
      '--checkpoint-source',str(SOURCE),'--resume-from',str(SOURCE),'--max-steps','100',
      '--runtime-root',str(ROOT/'repro_outputs/_runtime/train')]
env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
         PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
with (ROOT/'repro_outputs/runner_supervisor.log').open('x',buffering=1) as log:
    proc=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=proc.pid,argv=argv,command=command,gpu_index=1,gpu_uuid=GPU,
            gpu_before=gpu,compute_processes_before=processes,control_reused=True,
            authorization='User confirmed both visible GPUs may be used; only GPU1 allocated to H',
            budget_new_updates_H=100,new_updates_S=0)
(ROOT/'repro_outputs/launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
