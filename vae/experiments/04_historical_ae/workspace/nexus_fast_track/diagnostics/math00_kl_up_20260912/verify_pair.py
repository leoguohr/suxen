"""Read-only checks of paired continuations, Adam provenance and actual RNG tensors."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json,hashlib
from pathlib import Path
import torch

torch.set_num_threads(1)
BASE=Path(__file__).resolve().parent.parent
PARENT=BASE/'math00_kl_warm_20260912'
A=BASE/'math00_kl_hold_20260912';B=BASE/'math00_kl_up_20260912'
def digest(x):return hashlib.sha256(x.cpu().contiguous().numpy().tobytes()).hexdigest()
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
parent=torch.load(PARENT/'checkpoint-update0200.pt',map_location='cpu',weights_only=False,mmap=True)
result={}
for path in [A,B]:
    zero=torch.load(path/'checkpoint-update0000.pt',map_location='cpu',weights_only=False,mmap=True)
    end=torch.load(path/'checkpoint-update0200.pt',map_location='cpu',weights_only=False,mmap=True)
    assert equal(parent['model'],zero['model']) and equal(parent['optimizer'],zero['optimizer'])
    assert equal(parent['training_rng_state'],zero['training_rng_state'])
    assert zero['training_noise_draws']==800 and end['training_noise_draws']==1200
    assert list(zero['diagnostic_manifest']['groups'])==list(parent['diagnostic_manifest']['groups'])
    steps={}
    for group in end['optimizer']['param_groups']:
        steps[group['name']]=sorted(set(float(end['optimizer']['state'][i]['step']) for i in group['params']))
        assert steps[group['name']]==([600.] if group['name']=='logvar' else [800.])
    result[path.name]=dict(initial_weights_exact=True,all_five_initial_Adam_states_exact=True,initial_RNG_exact=True,final_Adam_steps=steps,final_rng_sha256=digest(end['training_rng_state']))
    del zero,end
old_hashes={h for p in [PARENT,BASE/'math00_fresh_sampling_20260912'] for s in (p/'updates.jsonl').read_text().splitlines() for h in json.loads(s)['epsilon_sha256']}
logs=[[json.loads(s) for s in (p/'updates.jsonl').read_text().splitlines()] for p in [A,B]]
assert len(logs[0])==len(logs[1])==200
assert all(r['beta']==1e-4 for r in logs[0])
assert all(abs(r['beta']-(1e-4+2e-4*min(1.,max(0.,(r['update']-1)/49.))))<1e-15 for r in logs[1])
for key in ['reconstruction_before_update','kl_before_update','actual_updates','gradient_norms_preclip']:
    assert logs[0][0][key]==logs[1][0][key],key
result['first_update_identical_at_beta_1e4']=True
for path in [A,B]:
    manifest=json.loads((path/'manifest.json').read_text())
    for filename,key in [('run.py','script_sha'),('backend00.py','backend_sha')]:
        assert hashlib.sha256((path/filename).read_bytes()).hexdigest()==manifest[key]
result['executed_code_hashes_verified']=True
seen=set();previous=parent['training_rng_state']
generator=torch.Generator(device='cuda');generator.set_state(previous)
for step,(la,lb) in enumerate(zip(*logs),1):
    assert la['epsilon_sha256']==lb['epsilon_sha256']
    assert la['rng_before_sha256']==lb['rng_before_sha256'] and la['rng_after_sha256']==lb['rng_after_sha256']
    a=torch.load(A/'training_noise'/f'{step:04d}.pt',weights_only=False)
    b=torch.load(B/'training_noise'/f'{step:04d}.pt',weights_only=False)
    assert equal(a,b) and equal(a['rng_before'],previous)
    for saved,h in zip(a['epsilon'],la['epsilon_sha256']):
        new=torch.randn(saved.shape,dtype=saved.dtype,device='cuda',generator=generator).cpu()
        assert torch.equal(saved,new) and digest(saved)==h and h not in seen|old_hashes
        seen.add(h)
    assert equal(generator.get_state(),a['rng_after'])
    previous=a['rng_after']
assert len(seen)==400
assert digest(previous)==result[A.name]['final_rng_sha256']==result[B.name]['final_rng_sha256']
result['paired_noise']=dict(steps=200,epsilon_tensors=400,all_actual_tensors_equal=True,replayed_exactly_from_parent_RNG=True,no_reuse_from_prior_training=True)
for path in [A,B]:
    seeds=json.loads((path/'evaluation_seeds.json').read_text())
    oldseeds=json.loads((PARENT/'evaluation_seeds.json').read_text())
    assert seeds['evaluation']==oldseeds['evaluation']
    assert not set(sum(seeds['final_unseen'],[])) & set(sum(oldseeds['final_unseen']+oldseeds['evaluation'],[])+oldseeds['training'])
result['monitor_set_preserved_and_final_set_new']=True
(B/'pair_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
