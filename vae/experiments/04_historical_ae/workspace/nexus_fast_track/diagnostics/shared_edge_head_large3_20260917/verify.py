"""Reload one shared head and evaluate all three complete edge graphs at each saved point."""
import json
from pathlib import Path
from probe import ROOT, T, np, setup, forward, evaluate, write

cfg,head,data,scoring,hashes=setup()
out=ROOT/'run';results={}
names=[f'checkpoint-step{s:04d}' for s in cfg['checks']]
if (out/'first-joint-perfect.pt').exists():names.append('first-joint-perfect')
for name in names:
    cp=T.load(out/(name+'.pt'),map_location='cpu',weights_only=False)
    assert set(cp['head'])=={'weight','bias'} and cp['head']['weight'].shape==(32,1024)
    head.load_state_dict(cp['head'],strict=True)
    actual=evaluate(head,data,scoring,cp['step'],cfg['outer_coefficient'])
    for d,row,expected in zip(data,actual['meshes'],cp['metrics']['meshes']):
        assert row['uid']==expected['uid']
        for k in ['tp','fp','fn','tn','edge_soft4','min_margin_gt','min_margin_non_gt']:assert row[k]==expected[k],(name,d['uid'],k)
        arrays=np.load(ROOT/'snapshots'/d['uid']/'representations_and_gradients.npz')
        faces=arrays['gt_faces'];n=len(arrays['vertices'])
        gt=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1)
        keys=np.unique(gt[:,0]*n+gt[:,1]);pairs=d['pairs'].cpu().numpy()
        assert np.array_equal(np.isin(pairs[:,0]*n+pairs[:,1],keys),d['labels'].cpu().numpy())
        saved=np.load(out/(name+'-'+d['uid']+'.npz'))
        with T.no_grad():
            values=forward(head,d,scoring);repeat=forward(head,d,scoring)
        assert all(T.equal(a,b) for a,b in zip(values,repeat))
        for key,value in zip(['edge_head_raw','edge_embedding_scoring','logits'],values[:3]):
            assert np.array_equal(saved[key],value.cpu().numpy()),(name,d['uid'],key)
    if name=='first-joint-perfect':assert actual['joint_perfect']
    if cp['step']:
        assert len(cp['optimizer']['state'])==2
        assert all(int(state['step'])==cp['step'] for state in cp['optimizer']['state'].values())
    results[name]=dict(step=cp['step'],one_shared_weight_and_bias=True,joint_perfect=actual['joint_perfect'],
        meshes=actual['meshes'],saved_logits_bitwise=True,repeated_forward_bitwise=True,gt_labels_from_faces_verified=True)
rows=[json.loads(s) for s in (out/'updates.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(cfg['updates']+1))
for row in rows:
    assert [r['uid'] for r in row['meshes']]==cfg['uids']
    assert row['joint_perfect']==all(r['fp']==r['fn']==0 for r in row['meshes'])
    assert abs(row['objective']-cfg['outer_coefficient']*sum(r['edge_soft4'] for r in row['meshes'])/3)<1e-10
complete=json.loads((out/'complete.json').read_text())
perfect=[r['step'] for r in rows[1:] if r['joint_perfect']]
assert len(perfect)==complete['joint_perfect_updates']
assert (perfect[0] if perfect else None)==complete['first_joint_perfect_step']
write(out/'verification.json',dict(checkpoints=results,contiguous_updates=cfg['updates'],
    identical_uids_every_step=True,mesh_equal_weight_objective_verified=True,one_head_all_meshes=True))
print('VERIFIED',cfg['updates'],'updates; shared head checkpoints',list(results),flush=True)
