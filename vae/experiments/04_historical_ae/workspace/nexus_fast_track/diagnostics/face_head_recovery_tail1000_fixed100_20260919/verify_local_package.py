"""Independent archive/count audit; does not load Torch or run/update any model."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);z=zipfile.ZipFile(p);assert z.testzip() is None
entries={}
for line in z.read('SHA256SUMS.txt').decode().splitlines():
    h,n=line.split('  ',1);entries[n]=h;digest=hashlib.sha256()
    with z.open(n) as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==h,n
read=lambda n:json.loads(z.read(n))
cfg=read('config.json');complete=read('run/complete.json');verification=read('run/verification.json');meta=read('cache/manifest.json')
uids=read('selection.json')['uids'];assert len(uids)==len(set(uids))==100
updates=[json.loads(x) for x in z.read('run/updates.jsonl').decode().splitlines()]
assert [u['update'] for u in updates]==list(range(1,501))
assert all(u['mesh_count']==100 and len(u['per_mesh_losses'])==100 and u['lr']==1e-4 for u in updates)
assert all(u['parameter_updates']['weight']['changed_elements']>0 for u in updates)
for u in updates:
    s=0.
    for v in u['per_mesh_losses']:s+=v
    assert s/100==u['mean_face_soft4_before']
states=[json.loads(x) for x in z.read('run/evaluations.jsonl').decode().splitlines()]
assert [e['step'] for e in states]==[0,50,100,200,300,400,500]
initial={r['uid'] for r in states[0]['meshes'] if r['joint_perfect']};assert len(initial)==55
for e,ret in zip(states,complete['retention']):
    rows=e['meshes'];assert [r['uid'] for r in rows]==uids
    for r,a in zip(rows,states[0]['meshes']):
        assert r['edge']==a['edge'] and r['face']['complete']
        assert r['face']['tp']+r['face']['fn']==r['gt_faces']
        assert r['edge']['tp']+r['edge']['fn']==r['gt_edges']
        assert r['gt_face_candidates']+r['missing_gt_face_candidates']==r['gt_faces']
        assert r['face']['fn']>=r['missing_gt_face_candidates']
        assert r['edge_perfect']==(r['edge']['fp']==r['edge']['fn']==0)
        assert r['face_perfect']==(r['face']['fp']==r['face']['fn']==0)
        assert r['joint_perfect']==(r['edge_perfect'] and r['face_perfect'])
        assert r['face']['scored_candidates']==a['face']['scored_candidates']
    for kind in ['edge','face']:
        for key in ['tp','fp','fn']:assert sum(r[kind][key] for r in rows)==e[kind+'_'+key]
    for key in ['edge_perfect','face_perfect','joint_perfect']:assert sum(r[key] for r in rows)==e[key]
    assert (e['edge_perfect'],e['edge_fp'],e['edge_fn'])==(75,130832,1)
    now={r['uid'] for r in rows if r['joint_perfect']}
    assert set(ret['retained'])==initial&now and set(ret['lost'])==initial-now and set(ret['gained'])==now-initial
    assert set(ret['success_uids'])==now
assert complete['updates']==500 and complete['budget_stopped'] and complete['state']=='complete'
assert states[0]==complete['initial'] and states[-1]==complete['final']==read('run/final_real_network.json')
for k in ['source_file_unchanged','frozen_weights_buffers_metadata_unchanged','all_cached_hidden_unchanged','all_actual_candidates_unchanged','all_pool_files_unchanged','final_head_checkpoint_reloaded','all100_real_network_matches_final_cache']:assert verification[k]
assert cfg['source_sha256']==complete['source_sha256']=='342fffbb7c2aaea4082dbbbde577757bb985909b4c613c56c4c7709a51c1fa63'
checked=0
if f'cache/{uids[0]}.npz' in z.namelist():
    for uid,r,rec in zip(uids,states[-1]['meshes'],meta['meshes']):
        assert entries[f'cache/{uid}.npz']==rec['cache_sha256']
        with np.load(io.BytesIO(z.read(f'cache/{uid}.npz'))) as c,np.load(io.BytesIO(z.read(f'run/final_outputs/{uid}.npz'))) as f:
            y=c['actual_labels'];pred=f['actual_logits']>0
            tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=len(c['gt_faces'])-tp
            assert (tp,fp,fn)==tuple(r['face'][k] for k in ['tp','fp','fn'])
            assert int((pred&~y&~c['actual_in_training_pool']).sum())==r['face']['actual_fp_outside_augmented_pool']
            assert int(((f['gt_logits']>0)&c['gt_covered']).sum())==tp
            n=len(c['hidden']);pk=np.unique(c['predicted_edges']@np.array([n,1]));gk=np.unique(c['edges']@np.array([n,1]));etp=int(np.isin(pk,gk).sum())
            assert (etp,len(pk)-etp,len(gk)-etp)==tuple(r['edge'][k] for k in ['tp','fp','fn'])
            checked+=len(y)
    assert checked==meta['actual_candidates']
result=dict(package=str(p),hashes_verified=len(entries),updates_verified=500,per_mesh_evaluations_verified=700,all100_every_update=True,fixed_edge_counts_verified=True,retention_verified=True,saved_actual_face_candidates_recounted=checked,scope='Local read-only archive, logs, IDs and optional saved logits. Full network and freeze verification are server evidence, not rerun locally.')
p.with_suffix('.local-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
