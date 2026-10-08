"""Replay only the independent training RNG, with no model/optimizer updates."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json
import hashlib
from pathlib import Path
import torch
torch.set_num_threads(1);torch.cuda.set_device(0)
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'evaluation_seeds.json').read_text())
gen=torch.Generator(device='cuda').manual_seed(cfg['training_rng_seed'])
records=[json.loads(s) for s in (ROOT/'updates.jsonl').read_text().splitlines()]
assert len(records)==200
digest=lambda t:hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()
for step,r in enumerate(records,1):
    saved=torch.load(ROOT/'training_noise'/f'{step:04d}.pt',map_location='cpu',weights_only=False)
    assert torch.equal(gen.get_state(),saved['rng_before'])
    assert digest(saved['rng_before'])==r['rng_before_sha256']
    for i,expected in enumerate(saved['epsilon']):
        actual=torch.randn(expected.shape,dtype=expected.dtype,device='cuda',generator=gen)
        assert torch.equal(actual.cpu(),expected)
        assert digest(expected)==r['epsilon_sha256'][i]
    assert torch.equal(gen.get_state(),saved['rng_after'])
    assert digest(saved['rng_after'])==r['rng_after_sha256']
    assert r['noise_draws_total']==2*step and r['backward_rng_unchanged']
cp=torch.load(ROOT/'checkpoint-update0200.pt',map_location='cpu',mmap=True,weights_only=False)
assert cp['training_noise_draws']==400
assert torch.equal(cp['training_rng_state'],gen.get_state())
result=dict(replayed_updates=200,exact_epsilon_tensors=400,continuous_independent_rng=True,backward_did_not_advance_rng=True,final_checkpoint_rng_exact=True)
(ROOT/'rng_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
