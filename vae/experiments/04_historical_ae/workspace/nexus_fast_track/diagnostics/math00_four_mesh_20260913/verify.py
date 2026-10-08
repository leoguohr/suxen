"""Read-only restoration, equal-mesh reduction, and training RNG verification."""
import os
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import torch
ROOT=Path(__file__).resolve().parent;torch.set_num_threads(1)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def digest(t):return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
parent=torch.load(ROOT.parent/'math00_kl_hold_20260912/checkpoint-update0200.pt',map_location='cpu',weights_only=False,mmap=True)
zero=torch.load(ROOT/'checkpoint-update0000.pt',map_location='cpu',weights_only=False,mmap=True)
end=torch.load(ROOT/'checkpoint-update1000.pt',map_location='cpu',weights_only=False,mmap=True)
assert equal(parent['model'],zero['model']) and equal(parent['optimizer'],zero['optimizer'])
assert equal(parent['training_rng_state'],zero['training_rng_state'])
assert zero['training_noise_draws']==1200 and end['training_noise_draws']==5200
steps={}
for g in end['optimizer']['param_groups']:
 steps[g['name']]=sorted(set(float(end['optimizer']['state'][p]['step']) for p in g['params']))
 assert steps[g['name']]==([1600.] if g['name']=='logvar' else [1800.])
manifest=json.loads((ROOT/'manifest.json').read_text())
assert manifest['mesh_count']==4 and manifest['beta_target']==1e-4
assert hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()==manifest['script_sha']
assert hashlib.sha256((ROOT/'backend00.py').read_bytes()).hexdigest()==manifest['backend_sha']
assert all(manifest['pool_sha256'][u]==parent['diagnostic_manifest']['pool_sha256'][u] for u in manifest['selected_uids'][:2])
rows=[json.loads(s) for s in (ROOT/'updates.jsonl').read_text().splitlines()];assert len(rows)==1000
previous=parent['training_rng_state'];rng=torch.Generator(device='cuda');rng.set_state(previous);seen=set()
for row in rows:
 assert row['beta']==1e-4 and len(row['parts_before_update'])==len(row['posterior_before_update'])==4
 expected=sum(x['edge']+x['face'] for x in row['parts_before_update'])/4
 assert abs(expected-row['reconstruction_before_update'])<1e-5
 assert abs(sum(x['total'] for x in row['kl_before_update']['per_mesh'])/4-row['kl_before_update']['total'])<1e-5
 noise=torch.load(ROOT/'training_noise'/f"{row['update']:04d}.pt",weights_only=False)
 assert equal(previous,noise['rng_before']) and len(noise['epsilon'])==4
 for e,h in zip(noise['epsilon'],row['epsilon_sha256']):
  actual=torch.randn(e.shape,dtype=e.dtype,device='cuda',generator=rng).cpu()
  assert equal(actual,e) and digest(e)==h and h not in seen;seen.add(h)
 assert equal(rng.get_state(),noise['rng_after']);previous=noise['rng_after']
assert equal(previous,end['training_rng_state']) and len(seen)==4000
r=dict(parent_weights_exact=True,five_Adam_states_restored_exact=True,initial_RNG_exact=True,final_Adam_steps=steps,updates=1000,meshes_per_update=4,unique_noise_tensors=4000,RNG_replayed_exact=True,old_candidate_pools_unchanged=True,equal_mesh_reconstruction_and_KL_verified_each_step=True,executed_code_hashes_verified=True)
(ROOT/'verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
