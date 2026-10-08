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
root = base / 'continue2000'

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

parent = load(base / 'mid3x/checkpoint-update0400.pt')
start = load(root / 'checkpoint-update0000.pt')
for key in ['model', 'optimizer', 'training_rng_state', 'training_noise_draws', 'args']:
    assert equal(parent[key], start[key]), key
assert start['completed_updates'] == 1400 and start['training_noise_draws'] == 6800
for mode in ['mu', 'sample']:
    before = read(base / f'mid3x/{mode}_step0400.json')
    after = read(root / f'{mode}_step0000.json')
    for key in ['rec', 'loss', 'kl', 'parts', 'posterior']:
        assert equal(before[key], after[key]), (mode, key)
old_noise = [json.loads(s) for s in (base / 'mid3x/independent_step0400.jsonl').read_text().splitlines()]
new_noise = [json.loads(s) for s in (root / 'monitoring_step0000.jsonl').read_text().splitlines()]
assert len(old_noise) == len(new_noise) == 50
for old, new in zip(old_noise, new_noise):
    for key in ['seeds', 'epsilon_sha256', 'rec', 'loss', 'kl', 'parts']:
        assert equal(old[key], new[key]), ('monitoring0', key)
result = dict(start_model_Adam_RNG_args_exact=True, start_mu_fixed_and_50_monitoring_exact=True)
if not args.start_only:
    rows = [json.loads(s) for s in (root / 'updates.jsonl').read_text().splitlines()]
    assert [r['update'] for r in rows] == list(range(1, 2001))
    assert rows[0]['rng_before_sha256'] == tensor_hash(parent['training_rng_state'])
    assert all(a['rng_after_sha256'] == b['rng_before_sha256'] for a, b in zip(rows, rows[1:]))
    previous = {h for s in (base / 'mid3x/updates.jsonl').read_text().splitlines() for h in json.loads(s)['epsilon_sha256']}
    hashes = [h for r in rows for h in r['epsilon_sha256']]
    assert len(set(hashes)) == 8000 and not previous & set(hashes)
    checkpoints = []
    for step in range(0, 2001, 200):
        cp = load(root / f'checkpoint-update{step:04d}.pt')
        assert cp['completed_updates'] == 1400 + step
        assert cp['additional_updates'] == step
        assert cp['training_noise_draws'] == 6800 + 4 * step
        assert equal(cp['optimizer']['param_groups'], parent['optimizer']['param_groups'])
        for group in cp['optimizer']['param_groups']:
            expected = step + (2000 if group['name'] == 'logvar' else 2200)
            assert all(float(cp['optimizer']['state'][i]['step']) == expected for i in group['params'])
        expected_rng = rows[step-1]['rng_after_sha256'] if step else tensor_hash(parent['training_rng_state'])
        assert tensor_hash(cp['training_rng_state']) == expected_rng
        assert all((root / f'{mode}_step{step:04d}.json').exists() for mode in ['mu', 'sample'])
        checkpoints.append(step)
    monitoring = {}
    for step in [0, 400, 800, 1200, 1600, 1800, 2000]:
        items = [json.loads(s) for s in (root / f'monitoring_step{step:04d}.jsonl').read_text().splitlines()]
        assert len(items) == 50
        assert all(r['seeds'] == initial['seeds'] and r['epsilon_sha256'] == initial['epsilon_sha256'] for r, initial in zip(items, new_noise))
        monitoring[step] = read(root / f'monitoring_summary_step{step:04d}.json')
        assert monitoring[step]['rng_unchanged']
    final_noise = [json.loads(s) for s in (root / 'final_unseen_step2000.jsonl').read_text().splitlines()]
    assert len(final_noise) == 50
    old_seeds = read(base / 'evaluation_seeds.json')
    used = {s for key in ['evaluation', 'final_unseen'] for row in old_seeds[key] for s in row} | set(old_seeds['training'])
    assert not used & {s for r in final_noise for s in r['seeds']}
    complete = read(root / 'complete.json')
    assert complete['all_groups_updated_every_step'] and complete['nonzero_sampling_every_step']
    stable = all(read(root / f'mu_step{s:04d}.json')['all_meshes_perfect'] and monitoring[s]['all_perfect'] == 50 for s in [1600, 1800, 2000]) and all(r['all_meshes_perfect'] for r in final_noise)
    result.update(updates=2000, cumulative_updates=3400, unique_fresh_epsilon=8000, RNG_chain_exact=True, checkpoints=checkpoints, monitoring_steps=list(monitoring), final_noise_disjoint=True, all_five_groups_updated_each_step=True, stable_four_mesh_pass=stable)
out = root / ('start_verification.json' if args.start_only else 'verification.json')
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
