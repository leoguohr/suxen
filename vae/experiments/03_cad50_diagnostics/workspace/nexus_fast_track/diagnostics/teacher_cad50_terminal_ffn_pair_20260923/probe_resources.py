"""Read-only host/resource inventory for the explicitly assigned servers."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--port',choices=['31548','32483'],required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent
parent=root.parent/'teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt'
digest=hashlib.sha256()
with parent.open('rb') as handle:
    for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
assert digest.hexdigest()=='4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66'
record=dict(port=int(args.port),host=platform.node(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,driver_version','--format=csv,noheader'],text=True),
    compute_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True),
    parent_checkpoint=str(parent),parent_sha256=digest.hexdigest(),parent_bytes=parent.stat().st_size,
    parent_inode=parent.stat().st_ino,optimizer_updates=0,gpu_model_forwards=0)
out=root/'repro_outputs'/('resources_'+args.port+'.json');out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
