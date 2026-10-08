"""Reload the same one-head checkpoints on all fixed 100 meshes and export final all-pair logits."""
import json
from head_core import ROOT,T,np,load,forward,metrics,summary,cycle,write,sha
cfg,meta,head,data,scoring=load();scale=meta['scales']['edge_logit_scale']
plan=json.loads((ROOT/'training_plan.json').read_text());out=ROOT/'run'
traces=[json.loads(s) for s in (out/'evaluations.jsonl').read_text().splitlines()]
updates=[json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
assert [r['step'] for r in traces]==list(range(plan['updates']+1))
assert [r['update'] for r in updates]==list(range(1,plan['updates']+1))
uids=[d['uid'] for d in data];results={}
for r in traces:
    assert [m['uid'] for m in r['meshes']]==uids
    assert r['objective']==sum(m['edge_soft4'] for m in r['meshes'])/100
    assert r['edge_perfect']==sum(m['fp']==m['fn']==0 for m in r['meshes'])
for u in updates:assert u['meshes']==100 and u['mesh_coefficient']==.01
for step in plan['checks']:
    path=out/f'checkpoint-step{step:04d}.pt';cp=T.load(path,map_location='cpu',weights_only=False)
    assert cp['step']==step and set(cp['head'])=={'weight','bias'} and cp['head']['weight'].shape==(32,1024)
    head.load_state_dict(cp['head'],strict=True)
    rows=cycle(head,data,scoring,scale,False);assert rows==traces[step]['meshes']==cp['metrics']['meshes']
    if step:
        assert len(cp['optimizer']['state'])==2
        assert all(int(s['step'])==step for s in cp['optimizer']['state'].values())
    results[str(step)]=dict(reload_all100_exact=True,sha256=sha(path),edge_perfect=sum(r['perfect'] for r in rows))
finaldir=out/'final_outputs';finaldir.mkdir(exist_ok=False)
with T.no_grad():
    for d in data:
        v=forward(head,d,scoring,scale);again=forward(head,d,scoring,scale)
        assert all(T.equal(a,b) for a,b in zip(v,again))
        assert metrics(d,v)==traces[-1]['meshes'][uids.index(d['uid'])]
        np.savez_compressed(finaldir/f"{d['uid']}.npz",edge_head_raw=v[0].cpu().numpy(),edge_embedding_scoring=v[1].cpu().numpy(),logits=v[2].cpu().numpy())
for row in meta['meshes']:assert sha(ROOT/'cache'/f"{row['uid']}.npz")==row['sha256']
assert sha(cfg['source_checkpoint'])==cfg['source_sha256']
write(out/'verification.json',dict(checkpoints=results,all100_each_step=True,equal_mesh_objective=True,
    cache_unchanged=True,source_network_checkpoint_unchanged=True,continuous_updates=len(updates),
    four_or_100_independent_heads=False,one_shared_head=True,repeated_final_forward_bitwise=True,
    all_final_pair_logits_exported=True))
write(ROOT/'status.json',dict(state='verified_complete',updates=plan['updates']))
print('VERIFIED',len(updates),'updates',len(uids),'fixed meshes',flush=True)
