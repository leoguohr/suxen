"""Read-only verification of the completed budget and saved state; CPU only."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import json
import hashlib
from pathlib import Path
import torch

r=Path(__file__).resolve().parent
torch.set_num_threads(1)
done=json.loads((r/'completion.json').read_text())
assert done['completed_updates']==3000
logs=[json.loads(l) for l in (r/'updates.jsonl').read_text().splitlines()]
assert [x['update'] for x in logs]==list(range(1,3001))
def load(path):return torch.load(path,map_location='cpu',mmap=True)
initial=load(r/'checkpoint-update0000.pt')
final=load(r/'checkpoint-update3000.pt')
assert not initial['optimizer']['state']
assert final['completed_updates']==3000
for n,p in final['model'].items():
    if isinstance(p,torch.Tensor):assert torch.isfinite(p).all(),n
    if 'log_variance.' in n:assert torch.equal(p,initial['model'][n]),n
for g in final['optimizer']['param_groups']:
    assert g['name']!='logvar'
    assert g['betas']==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
    assert abs(g['lr']-(1e-5 if g['name']=='encoder_mu' else 1e-4))<1e-15
    for pid in g['params']:assert int(final['optimizer']['state'][pid]['step'])==3000
for u in logs:
    assert u['interface']['mu_gradient_nonzero_channels']==512
    assert all(v['delta_l2']>0 for v in u['actual_updates'].values())
    t=1+9*min((u['update']-1)/99,1)
    for name,lr in u['lr'].items():assert abs(lr-t*(1e-6 if name=='encoder_mu' else 1e-5))<1e-15
checks=initial['diagnostic']['checks']
hashes={}
for step in checks:
    e=json.loads((r/f'eval-update{step:04d}.json').read_text())
    cp=r/f'checkpoint-update{step:04d}.pt'
    h=hashlib.sha256()
    with cp.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    assert h.hexdigest()==e['checkpoint_sha256']
    hashes[str(step)]=h.hexdigest()
    assert e['edge']['tp']+e['edge']['fn']==2406
    assert e['face']['tp']+e['face']['fn']==1604
    assert e['gt_face_candidates']+e['missing_gt_face_candidates']==1604
    assert e['perfect']==(e['edge']['fp']==e['edge']['fn']==e['face']['fp']==e['face']['fn']==0)
# Verify restarting the zero-update instrumentation failure reproduced identical initialization.
attempt=load(r/'attempt0_logging_preflight/checkpoint-update0000.pt')
for n,p in initial['model'].items():
    if isinstance(p,torch.Tensor):assert torch.equal(p,attempt['model'][n]),n
result=dict(completed_updates=3000,contiguous_log=True,all_checkpoint_hashes_verified=hashes,
            initial_adam_empty=True,final_adam_all_steps3000=True,logvar_weights_unchanged=True,
            all_trainable_groups_nonzero_updates_each_step=True,all512_mu_channels_have_gradient_each_step=True,
            warmup_and_constant_lr_verified=True,zero_update_restart_reproduced_identical_initial_weights=True,
            final_checkpoint_sha256=hashes['3000'])
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
