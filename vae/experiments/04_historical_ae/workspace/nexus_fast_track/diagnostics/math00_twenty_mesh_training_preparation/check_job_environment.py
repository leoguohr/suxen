"""Validate this task in a newly scheduled AIStation container; no training."""
import hashlib,importlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
expected=json.loads((ROOT/'job_environment.json').read_text())
assert sys.version_info[:2]==(3,10),sys.version
actual={}
for name,version in expected['versions'].items():
    module=importlib.import_module(name);actual[name]=module.__version__
    assert actual[name]==version,(name,actual[name],version,'Use the same image as the validated container.')
import torch
assert torch.cuda.is_available(),'No visible CUDA GPU'
assert torch.cuda.get_device_properties(0).total_memory>=75*1024**3,'This task was prepared for one 80GB GPU.'
report=json.loads((ROOT/'preflight/result.json').read_text())
assert report['passed'] and report['optimizer_updates']==0
for filename,key in [('train.py','entry_sha256'),('recommended_config.json','config_sha256')]:
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==report[key],filename
for filename,digest in report['input_file_sha256'].items():
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,filename
cfg=json.loads((ROOT/'recommended_config.json').read_text())
assert Path(cfg['start_checkpoint']).is_file(),cfg['start_checkpoint']
assert len(list((ROOT/'pools').glob('*_pool.npz')))==20
# Import the actual helpers in this image to detect missing archived dependencies.
sys.path.insert(0,str(ROOT.parent/'math00_four_mesh_lr10x_20260913'))
import backend00
print(json.dumps(dict(environment_check='passed',versions=actual,gpu=torch.cuda.get_device_name(0),training_started=False)),flush=True)
