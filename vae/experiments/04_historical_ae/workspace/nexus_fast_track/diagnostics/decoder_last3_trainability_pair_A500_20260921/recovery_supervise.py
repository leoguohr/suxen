"""Record explicit interrupted-run lineage and remaining fixed budget."""
import json,os,platform,shlex,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ['gate','train']
sys.path.insert(0,str(R/'_rigorpilot/shared/scripts'))
from runtime_runner import run_persistent_command,reconcile_run
cfg=json.loads((R/'config.json').read_text());o=R/'repro_outputs/recovery_204';parent='20260921T142436Z-4bf71f12'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert cfg['gpu_uuid'] in gpu and not apps.strip()
reconciled=reconcile_run(R/'repro_outputs/_runtime/train'/parent);assert reconciled['status']=='interrupted'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=cfg['gpu_uuid'],PYTHONDONTWRITEBYTECODE='1',RIGORPILOT_LESSONS='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
argv=[sys.executable,'-B',str(R/'recovery.py'),phase]
record={'created_at':datetime.now(timezone.utc).isoformat(),'phase':phase,'argv':argv,'host':platform.node(),'gpu_before':gpu,'compute_apps_before':apps,'original_run':parent,'resume_from_update':204,'remaining_control_updates':296,'remaining_treatment_updates':500,'explicit_environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','RIGORPILOT_LESSONS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']}}
(o/(phase+'_launch.json')).write_text(json.dumps(record,indent=2)+'\n')
result=run_persistent_command(repo=R,command=shlex.join(argv),timeout=3600 if phase=='gate' else 172800,runtime_root=R/'repro_outputs/_runtime'/('recovery_gate' if phase=='gate' else 'train'),child_env=env,monitor_gpu=True,retry_of=parent if phase=='train' else None,attempt=2 if phase=='train' else 1)
(o/(phase+'_runner_result.json')).write_text(json.dumps(result,indent=2)+'\n')
assert result['runtime_status']=='success',result['runtime_status']
if phase=='train':subprocess.run([sys.executable,'-B',str(R/'summarize_results.py')],cwd=R,env=env,check=True)
print('RECOVERY '+phase+' COMPLETE',flush=True)
