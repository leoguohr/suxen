"""CPU-only verification of paired checkpoint origins and completed budgets."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from pathlib import Path
import json
import hashlib
import torch

r=Path(__file__).resolve().parent
parent=r.parent/'math00_latent512_804_fresh_20260914/checkpoint-update3000.pt'
torch.set_num_threads(1)
def load(p):return torch.load(p,map_location='cpu',mmap=True)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
assert digest(parent)=='06dc42726c816b69f06db353d4de505e9d8bceda864294c8bb75c7907ba0bdd6'
p=load(parent);out={}
for name,mult in [('A_hold',1.),('B_lr03',.3)]:
    d=r/name
    start=load(d/'checkpoint-update0000.pt');end=load(d/'checkpoint-update0500.pt')
    assert start['completed_updates']==3000 and end['completed_updates']==3500 and end['additional_updates']==500
    assert equal(start['model'],p['model'])
    assert equal(start['optimizer']['state'],p['optimizer']['state'])
    assert equal(start['torch_rng'],p['torch_rng']) and equal(start['cuda_rng'],p['cuda_rng'])
    assert equal(start['python_rng'],p['python_rng'])
    assert start['numpy_rng'][0]==p['numpy_rng'][0]
    assert (start['numpy_rng'][1]==p['numpy_rng'][1]).all() and start['numpy_rng'][2:]==p['numpy_rng'][2:]
    assert equal(end['torch_rng'],start['torch_rng']) and equal(end['cuda_rng'],start['cuda_rng'])
    for g,pg in zip(start['optimizer']['param_groups'],p['optimizer']['param_groups']):
        assert g['lr']==pg['lr']*mult
        assert {k:v for k,v in g.items() if k!='lr'}=={k:v for k,v in pg.items() if k!='lr'}
    for group in end['optimizer']['param_groups']:
        assert group['name']!='logvar'
        for pid in group['params']:assert int(end['optimizer']['state'][pid]['step'])==3500
    for key,t in end['model'].items():
        if isinstance(t,torch.Tensor):assert torch.isfinite(t).all(),key
        if 'log_variance.' in key:assert equal(t,p['model'][key])
    logs=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
    assert [x['update'] for x in logs]==list(range(1,501))
    lrs={g['name']:g['lr'] for g in start['optimizer']['param_groups']}
    for x in logs:
        assert x['lr']==lrs
        assert x['interface']['mu_gradient_nonzero_channels']==512
        assert all(g['delta_l2']>0 for g in x['actual_updates'].values())
    hashes={}
    for step in [0,100,200,300,400,500]:
        e=json.loads((d/f'eval-update{step:04d}.json').read_text())
        sha=digest(d/f'checkpoint-update{step:04d}.pt')
        assert sha==e['checkpoint_sha256']
        assert e['edge']['tp']+e['edge']['fn']==2406
        assert e['face']['tp']+e['face']['fn']==1604
        hashes[str(step)]=sha
    out[name]=dict(parent_weights_exact=True,parent_adam_moments_and_steps_exact=True,
                   parent_rng_exact=True,rng_unchanged_during_mu_training=True,only_lr_options_changed=True,
                   logvar_weights_unchanged=True,updates=500,cumulative_updates=3500,
                   four_groups_updated_all_steps=True,all512_channels_have_gradient=True,checkpoint_sha256=hashes)
(r/'verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
