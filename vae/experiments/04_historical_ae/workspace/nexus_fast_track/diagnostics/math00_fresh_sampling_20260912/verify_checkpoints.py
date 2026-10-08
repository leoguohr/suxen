"""CPU verification of preserved Adam history and genuinely updated posterior."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
load=lambda path:torch.load(path,map_location='cpu',mmap=True,weights_only=False)
parent=load(ROOT.parent/'math00_mu_continuation_20260912/checkpoint-update0200.pt')
zero=load(ROOT/'checkpoint-update0000.pt');final=load(ROOT/'checkpoint-update0200.pt')
assert zero['completed_updates']==0 and final['completed_updates']==200
assert len(zero['optimizer']['param_groups'])==len(final['optimizer']['param_groups'])==5
for i,g in enumerate(parent['optimizer']['param_groups']):
    assert g==zero['optimizer']['param_groups'][i]
    for key in g['params']:
        a=parent['optimizer']['state'][key];b=zero['optimizer']['state'][key]
        assert a.keys()==b.keys()
        assert all(torch.equal(a[k],b[k]) for k in a)
for n,v in parent['model'].items():
    if isinstance(v,torch.Tensor):assert torch.equal(v,zero['model'][n])
    else:assert v==zero['model'][n]
lv=zero['optimizer']['param_groups'][4]
assert lv['name']=='logvar' and lv['lr']==1e-4
assert all(i not in zero['optimizer']['state'] for i in lv['params'])
steps={}
for i,g in enumerate(final['optimizer']['param_groups']):
    values=[float(final['optimizer']['state'][k]['step']) for k in g['params']]
    assert all(v==(200 if i==4 else 400) for v in values)
    steps[g['name']]=sorted(set(values))
lv_changes={n:float((final['model'][n].double()-zero['model'][n].double()).norm()) for n in zero['model'] if n.startswith('autoencoder.log_variance.')}
assert any(v>0 for v in lv_changes.values())
result=dict(checkpoints_readable=True,initial_weights_exactly_math00_mu_step200=True,existing_adam_state_exactly_preserved=True,logvar_adam_initially_absent=True,final_adam_steps=steps,logvar_actual_changes=lv_changes)
(ROOT/'checkpoint_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
