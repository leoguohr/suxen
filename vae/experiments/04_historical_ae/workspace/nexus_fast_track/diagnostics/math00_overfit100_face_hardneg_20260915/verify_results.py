"""Verify completion, traversal and actual updates from saved records only."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from pathlib import Path
import json, hashlib
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
run=ROOT/'run'
manifest=json.loads((run/'manifest.json').read_text())
parent=torch.load(manifest['parent_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
assert parent['completed_updates']==12500 and parent['epoch']==500
shuffle=np.random.default_rng();shuffle.bit_generator.state=parent['shuffle_rng']
updates=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
assert [r['update'] for r in updates]==list(range(12501,15001))
assert [r['additional_update'] for r in updates]==list(range(1,2501))
lrs={g['name']:g['lr'] for g in parent['optimizer']['param_groups']}
assert len(lrs)==4
for epoch in range(501,601):
    order=shuffle.permutation(manifest['uids']).tolist()
    assert json.loads((run/f'epoch-order-{epoch}.json').read_text())['uids']==order
    group=updates[(epoch-501)*25:(epoch-500)*25]
    assert [m['uid'] for r in group for m in r['meshes']]==order
    for r in group:
        assert r['epoch']==epoch and r['lr']==lrs and len(r['meshes'])==4
        assert all(m['participation']==epoch for m in r['meshes'])
        assert set(r['actual_updates'])==set(lrs)
        assert all(v['delta_l2']>0 for v in r['actual_updates'].values())
summaries=[]
for epoch in [500,525,550,575,600]:
    rows=[json.loads(x) for x in (run/f'eval-epoch{epoch}.jsonl').read_text().splitlines()]
    assert [r['uid'] for r in rows]==manifest['uids'] and len(rows)==100
    s=json.loads((run/f'eval-summary-epoch{epoch}.json').read_text())
    assert s['joint_perfect']==sum(r['joint_perfect'] for r in rows)
    summaries.append({k:s[k] for k in ['epoch','updates','joint_perfect','total_edge_fp','total_edge_fn','total_face_fp','total_face_fn','face_incomplete']})
complete=json.loads((run/'complete.json').read_text())
assert complete['updates']==15000 and complete['epochs']==600 and complete['stopped_at_budget']
assert set(complete['per_mesh_participations'].values())=={600}
path=Path(complete['final_evaluation']['checkpoint'])
with path.open('rb') as f:
    h=hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
if h is None:
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    h=digest.hexdigest()
assert h==complete['final_evaluation']['checkpoint_sha256']
result=dict(completed=True,additional_updates=2500,end_updates=15000,end_epoch=600,
    parent_rng_replayed_all100_orders=True,each_mesh_additional_participations=100,
    four_reconstruction_groups_nonzero_all2500_updates=True,lrs_unchanged_from_parent=lrs,
    final_checkpoint_sha256=h,full_evaluations=summaries)
(ROOT/'completion_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
