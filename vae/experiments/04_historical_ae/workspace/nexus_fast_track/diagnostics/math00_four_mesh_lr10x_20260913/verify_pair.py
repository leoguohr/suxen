"""Read-only verification of the paired LR branches."""
from pathlib import Path
import json
import torch

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'math00_four_mesh_20260913'/'checkpoint-update1000.pt'

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def read(path):return json.loads(path.read_text())

parent=torch.load(PARENT,map_location='cpu',weights_only=False,mmap=True)
expected_parent_sha=read(ROOT.parent/'math00_four_mesh_20260913'/'checkpoint_final_manifest.json')['sha256']
starts={b:torch.load(ROOT/b/'checkpoint-update0000.pt',map_location='cpu',weights_only=False,mmap=True) for b in ['control','high10x']}
ends={b:torch.load(ROOT/b/'checkpoint-update0400.pt',map_location='cpu',weights_only=False,mmap=True) for b in ['control','high10x']}
assert all(equal(parent['model'],x['model']) for x in starts.values())
assert all(equal(parent['optimizer']['state'],x['optimizer']['state']) for x in starts.values())
assert all(equal(parent['training_rng_state'],x['training_rng_state']) for x in starts.values())
assert all(x['training_noise_draws']==5200 for x in starts.values())
for branch,start in starts.items():
    for pg,base in zip(start['optimizer']['param_groups'],parent['optimizer']['param_groups']):
        assert pg['params']==base['params'] and pg['name']==base['name']
        expected=base['lr'] if branch=='control' or pg['name']=='logvar' else 10*base['lr']
        assert pg['lr']==expected
for branch,end in ends.items():
    assert end['completed_updates']==1400 and end['additional_updates']==400 and end['training_noise_draws']==6800
    steps={}
    for pg in end['optimizer']['param_groups']:
        steps[pg['name']]=sorted(set(float(end['optimizer']['state'][i]['step']) for i in pg['params']))
        assert steps[pg['name']]==([2000.] if pg['name']=='logvar' else [2200.])

rows={b:[json.loads(x) for x in (ROOT/b/'updates.jsonl').read_text().splitlines()] for b in ['control','high10x']}
assert len(rows['control'])==len(rows['high10x'])==400
assert all(a['epsilon_sha256']==b['epsilon_sha256'] and a['rng_before_sha256']==b['rng_before_sha256'] and a['rng_after_sha256']==b['rng_after_sha256'] for a,b in zip(rows['control'],rows['high10x']))
assert len({h for x in rows['control'] for h in x['epsilon_sha256']})==1600
assert all(len(x['parts_before_update'])==4 and abs(x['reconstruction_before_update']-sum(p['edge']+p['face'] for p in x['parts_before_update'])/4)<1e-5 for branch in rows.values() for x in branch)
assert all(x['actual_updates'][g]['delta_l2']>0 for branch in rows.values() for x in branch for g in x['actual_updates'])
assert all(p['actual_noise_nonzero_elements']>0 for branch in rows.values() for x in branch for p in x['posterior_before_update'])
for branch in ['control','high10x']:
    manifest=read(ROOT/branch/'manifest.json');complete=read(ROOT/branch/'complete.json')
    assert manifest['parent_sha256']==expected_parent_sha and starts[branch]['parent_checkpoint_sha']==expected_parent_sha and complete['training_epsilon_hashes']==1600
    assert complete['all_groups_updated_every_step'] and complete['nonzero_sampling_every_step']
result=dict(parent_weights_exact_both=True,parent_Adam_moments_exact_both=True,parent_training_RNG_exact_both=True,only_optimizer_LR_differs_at_step0=True,paired_training_epsilon_exact_all_400=True,unique_training_epsilon_hashes=1600,equal_mesh_reconstruction_verified_each_step=True,all_five_groups_updated_each_step=True,nonzero_sampling_each_step=True,final_Adam_steps={b:{pg['name']:sorted(set(float(ends[b]['optimizer']['state'][i]['step']) for i in pg['params'])) for pg in ends[b]['optimizer']['param_groups']} for b in ends})
(ROOT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
