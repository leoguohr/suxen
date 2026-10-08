"""Refuse to initialize CUDA unless the recorded CAD GPU is unoccupied."""
import json,os,subprocess,time
from pathlib import Path
GPU='GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
info=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=index,name,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines()
occupied=[line for line in processes if GPU in line]
assert not occupied,('CAD GPU occupied; do not stop or share other training',occupied)
assert int(info.split(',')[3].strip())<100,('Unexpected GPU allocation',info)
with (Path(__file__).resolve().parent/'gpu_checks.jsonl').open('a') as f:
    f.write(json.dumps(dict(time=time.time(),gpu=info,compute_processes=occupied,passed=True))+'\n')
print('GPU_FREE_FOR_CAD',info,flush=True)
