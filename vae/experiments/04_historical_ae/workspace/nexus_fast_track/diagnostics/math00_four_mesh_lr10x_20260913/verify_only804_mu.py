"""Read-only tensor and trajectory verification; writes a small audit result."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--start-only', action='store_true')
args = parser.parse_args()
torch.set_num_threads(1)
base = Path(__file__).resolve().parent
root = base / 'only804_mu'

def read(path):
    return json.loads(path.read_text())

def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b

def load(path):
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)

def tensor_hash(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

parent = load(base / 'continue2000/checkpoint-update2000.pt')
start = load(root / 'checkpoint-update0000.pt')
for key in ['model', 'optimizer', 'training_rng_state', 'training_noise_draws', 'args']:
    assert equal(parent[key], start[key]), key
assert start['completed_updates'] == 3400 and start['training_noise_draws'] == 14800
for mode in ['mu', 'sample']:
    before = read(base / f'continue2000/{mode}_step2000.json')
    after = read(root / f'{mode}_step0000.json')
    for key in ['rec', 'loss', 'kl', 'parts', 'posterior']:
        assert equal(before[key], after[key]), (mode, key)
result=dict(start_weights_Adam_RNG_exact=True,start_mu_and_fixed_sampling_exact=True)
if not args.start_only:
    rows=[json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows]==list(range(1,1001))
    parent_hash=tensor_hash(parent['training_rng_state'])
    for r in rows:
        assert r['beta']==0 and r['latent']=='mu' and r['epsilon_sha256']==[]
        assert r['rng_before_sha256']==r['rng_after_sha256']==parent_hash
        assert r['logvar_grad_is_none'] and r['actual_updates']['logvar']['delta_l2']==0
        assert all(v['delta_l2']>0 for k,v in r['actual_updates'].items() if k!='logvar')
        assert all(p['actual_noise_nonzero_elements']==0 for p in r['posterior_before_update'])
    checks=[]
    lv_names=parent['diagnostic_manifest']['groups']['logvar']
    for step in range(0,1001,200):
        cp=load(root/f'checkpoint-update{step:04d}.pt')
        assert cp['completed_updates']==3400+step and cp['training_noise_draws']==14800
        assert equal(cp['training_rng_state'],parent['training_rng_state'])
        assert equal(cp['optimizer']['param_groups'],parent['optimizer']['param_groups'])
        for name in lv_names:assert equal(cp['model']['autoencoder.'+name],parent['model']['autoencoder.'+name])
        for g in cp['optimizer']['param_groups']:
            for i in g['params']:
                if g['name']=='logvar':assert equal(cp['optimizer']['state'][i],parent['optimizer']['state'][i])
                else:assert float(cp['optimizer']['state'][i]['step'])==4200+step
        new=read(root/f'mu_step{step:04d}.json')['rec']
        old=read(base/f'only804/mu_step{step:04d}.json')['rec']
        checks.append(dict(step=step,mu804=new[3],sampling_KL_control_mu804=old[3],other_meshes=new[:3]))
    result.update(updates=1000,logvar_parameters_and_Adam_state_unchanged=True,four_reconstruction_groups_updated=True,training_rng_unchanged=True,mu_comparison=checks)
    write_path=root/'comparison.json'
    write_path.write_text(json.dumps(checks,indent=2)+'\n')
out=root/('start_verification.json' if args.start_only else 'verification.json')
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='mu_comparison'}))
