"""Read-only verification of continuation checkpoints, Adam, RNG and acceptance."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'math00_latent512_804_lr03_continue1000_20260914'
torch.set_num_threads(1)
def load(path):
    return torch.load(path, map_location='cpu', mmap=True)
def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()

p=load(PARENT/'checkpoint-update1000.pt')
s=load(ROOT/'checkpoint-update0000.pt')
e=load(ROOT/'checkpoint-update0500.pt')
assert digest(PARENT/'checkpoint-update1000.pt')=='d620530cb3654d08f98204f538e5b9bd754e260ce20a6f79a43776ac0cb3ad1b'
assert s['completed_updates']==4500 and e['completed_updates']==5000
assert e['additional_updates']==500
assert equal(s['model'],p['model']) and equal(s['optimizer'],p['optimizer'])
for key in ['python_rng','numpy_rng','torch_rng','cuda_rng']:
    assert equal(s[key],p[key]) and equal(e[key],s[key]),key
assert equal(e['optimizer']['param_groups'],s['optimizer']['param_groups'])
for g in e['optimizer']['param_groups']:
    assert g['name']!='logvar'
    for i in g['params']: assert int(e['optimizer']['state'][i]['step'])==5000
for key,t in e['model'].items():
    if isinstance(t,torch.Tensor): assert torch.isfinite(t).all(),key
    if 'log_variance.' in key: assert equal(t,p['model'][key]),key
logs=[json.loads(line) for line in (ROOT/'updates.jsonl').read_text().splitlines()]
assert [x['update'] for x in logs]==list(range(1,501))
lrs={g['name']:g['lr'] for g in s['optimizer']['param_groups']}
for x in logs:
    assert x['lr']==lrs and x['cumulative_update']==4500+x['update']
    assert x['interface']['mu_gradient_nonzero_channels']==512
    assert all(v['delta_l2']>0 for v in x['actual_updates'].values())
hashes={}
for step in [0,100,200,300,400,500]:
    row=json.loads((ROOT/f'eval-update{step:04d}.json').read_text())
    hashes[str(step)]=digest(ROOT/f'checkpoint-update{step:04d}.pt')
    assert row['checkpoint_sha256']==hashes[str(step)]
    assert row['edge']['tp']+row['edge']['fn']==2406
    assert row['face']['tp']+row['face']['fn']==1604
result=dict(parent_weights_exact=True,parent_optimizer_including_lr_exact=True,
            rng_exact_and_unchanged=True,logvar_unchanged=True,updates=500,cumulative_updates=5000,
            four_groups_updated_every_step=True,all512_channels_have_gradient=True,
            checkpoint_sha256=hashes)
(ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
