"""Record exact bounded launch and run it through RigorPilot's training runner."""
import json,os,platform,shlex,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
R=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ['preflight','train']
cfg=json.loads((R/'config.json').read_text())
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout
assert cfg['gpu_uuid'] in gpu and not apps.strip(),'Assigned GPU must be idle; no process termination or resource expansion'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],PYTHONDONTWRITEBYTECODE='1',RIGORPILOT_LESSONS='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
command=shlex.join([sys.executable,'-B',str(R/'pipeline.py'),phase])
argv=[sys.executable,'-B',str(R/'_rigorpilot/skills/run-train/scripts/run_training.py'),'--repo',str(R),'--command',command,'--timeout',str(3600 if phase=='preflight' else 172800),'--lane','trusted','--run-mode','startup_verification' if phase=='preflight' else 'resume','--dataset','fixed_original_100_meshes','--checkpoint-source',cfg['source_full'],'--resume-from',cfg['source_checkpoint'],'--max-steps',str(0 if phase=='preflight' else 500),'--runtime-root',str(R/'repro_outputs/_runtime'/phase)]
record={'created_at':datetime.now(timezone.utc).isoformat(),'phase':phase,'argv':argv,'inner_command':command,'cwd':str(R),'explicit_environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','RIGORPILOT_LESSONS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']},'gpu_before':gpu,'compute_processes_before':apps,'host':platform.node(),'python':platform.python_version(),'budget_per_branch':0 if phase=='preflight' else 500,'total_budget':0 if phase=='preflight' else 1000,'runner_timeout_seconds':3600 if phase=='preflight' else 172800}
(R/'repro_outputs'/f'{phase}_launch.json').write_text(json.dumps(record,indent=2)+'\n')
with (R/'repro_outputs'/f'{phase}_runner_result.json').open('w') as output:
    code=subprocess.call(argv,cwd=R,env=env,stdout=output)
if code:raise SystemExit(code)
result=json.loads((R/'repro_outputs'/f'{phase}_runner_result.json').read_text())
assert result['runtime_status']=='success',result['runtime_status']
if phase=='train':
    subprocess.run([sys.executable,'-B',str(R/'summarize_results.py')],cwd=R,env=env,check=True)
print(phase+' supervisor completed',flush=True)
