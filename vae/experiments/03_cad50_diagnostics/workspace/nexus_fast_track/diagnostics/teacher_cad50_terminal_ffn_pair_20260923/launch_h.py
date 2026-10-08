"""Launch the authorized 100-update terminal-FFN treatment, with a startup gate."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
GPU='GPU-0ae7719a-74e5-08ae-75de-0a6205381365'
SOURCE=ROOT.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt'
SOURCE_SHA='4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66'
RUNNER=ROOT.parent/'decoder_last3_trainability_pair_A500_20260921/_rigorpilot/skills/run-train/scripts/run_training.py'
lock=(ROOT/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'repro_outputs/launch.json').exists(),'Refuse duplicate launch'
audit=json.loads((ROOT/'repro_outputs/CONTROL_REUSE_AUDIT.json').read_text())
assert audit['passed'] and audit['reuse_control'] and audit['source_sha256']==SOURCE_SHA
test=json.loads((ROOT/'H_terminal_ffn/terminal_ffn_test.json').read_text())
assert test['passed'] and test['real_experiment_optimizer_updates']==0
for name,expected in test['code_sha256'].items():
    assert hashlib.sha256((ROOT/'H_terminal_ffn'/name).read_bytes()).hexdigest()==expected,name
hashes=json.loads((ROOT/'repro_outputs/CODE_HASHES.json').read_text())
for relative,expected in hashes.items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert GPU not in processes,'Allocated GPU is now occupied; do not preempt'
row=next(x for x in gpus.splitlines() if GPU in x)
assert row.split(',')[0].strip()=='0' and int(row.split(',')[3].strip())<512
command=f'{sys.executable} -u {ROOT}/H_terminal_ffn/train.py --mode terminal_ffn'
argv=[sys.executable,str(RUNNER),'--repo',str(ROOT),'--command',command,
    '--run-mode','resume','--lane','trusted','--timeout','7200',
    '--dataset','Original teacher CAD50 and fixed13946 Face pool; all50 accumulated per update',
    '--checkpoint-source',str(SOURCE),'--resume-from',str(SOURCE),'--max-steps','100',
    '--runtime-root',str(ROOT/'repro_outputs/_runtime/train')]
environment=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',RIGORPILOT_LESSONS='0')
with (ROOT/'repro_outputs/runner_supervisor.log').open('x',buffering=1) as log:
    process=subprocess.Popen(argv,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(pid=process.pid,argv=argv,command=command,gpu_index=0,gpu_uuid=GPU,
    gpu_before=gpus,processes_before=processes,source_sha256=SOURCE_SHA,
    control_reused=True,new_updates_control_this_turn=0,new_updates_treatment_budget=100,
    startup_gate='Trainer must write ready and await release after zero-update real-network verification',
    authorization='User explicitly requested this terminal FFN experiment and allocated both servers',
    runner_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest())
(ROOT/'repro_outputs/launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2),flush=True)
