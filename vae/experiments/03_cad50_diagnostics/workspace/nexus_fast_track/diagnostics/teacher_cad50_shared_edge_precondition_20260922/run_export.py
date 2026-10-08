"""Persist the bounded export with the already installed RigorPilot runtime."""
import os,sys,json
from pathlib import Path
N=Path(__file__).resolve().parent
sys.path.insert(0,str(N.parent/'decoder_last3_trainability_pair_A500_20260921/_rigorpilot/shared/scripts'))
from runtime_runner import run_persistent_command
env=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351',
    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
result=run_persistent_command(repo=N,command=f'{sys.executable} -u {N}/export_features.py',
    timeout=600,runtime_root=N/'repro_outputs/_runtime/export',monitor_gpu=True,child_env=env)
(N/'repro_outputs/export_runner_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ['status','returncode','exit_code','runtime']}),flush=True)
