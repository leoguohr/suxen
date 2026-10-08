"""Read-only local archive/count audit, with optional saved all-pair Edge logits."""
import hashlib,io,json,sys,zipfile
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);z=zipfile.ZipFile(p);assert z.testzip() is None
hashes={}
for line in z.read('SHA256SUMS.txt').decode().splitlines():
    h,n=line.split('  ',1);d=hashlib.sha256()
    with z.open(n) as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
    assert d.hexdigest()==h,n;hashes[n]=h
read=lambda n:json.loads(z.read(n))
cfg=read('config.json');complete=read('run/complete.json');verify=read('run/verification.json');restore=read('restore_verification.json')
assert complete['new_updates']==500 and complete['cumulative_tail_updates']==1500 and complete['cumulative_joint_updates']==1000 and complete['stopped_at_budget']
assert restore['adam_initial_step']==1000 and restore['adam_all_moments_steps_and_groups_exact'] and restore['all_three_groups_restored_together']
assert cfg['source_sha256']=='5a36ab88ee147ef160945cfcec11d1d5cb98026e20c536ce602f6303f1448f7e'
assert cfg['parent_checkpoint_sha256']=='4e328fcff09d85d001af29548cfd559bf39306054b8c05825249cc95e2ad3b68'
uids=read('selection.json')['uids'];assert len(uids)==len(set(uids))==100
trace=[json.loads(s) for s in z.read('run/edge_trace.jsonl').decode().splitlines()]
updates=[json.loads(s) for s in z.read('run/updates.jsonl').decode().splitlines()]
actual=[json.loads(s) for s in z.read('run/actual_evaluations.jsonl').decode().splitlines()]
assert [e['step'] for e in trace]==list(range(501))
assert [u['update'] for u in updates]==list(range(1,501))
assert [e['step'] for e in actual]==[0,100,200,300,400,500]
for u in updates:
    assert u['mesh_count']==100 and u['cumulative_tail_update']==1000+u['update']
    assert u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4,'face_head':1e-4,'decoder14':1e-5}
    assert all(v['delta_norm']>0 for v in u['actual_updates'].values())
for e in trace:
    assert [m['uid'] for m in e['meshes']]==uids
    assert e['total_fp']==sum(m['fp'] for m in e['meshes']) and e['total_fn']==sum(m['fn'] for m in e['meshes'])
    assert all(m['perfect']==(m['fp']==m['fn']==0) for m in e['meshes'])
    assert e['edge_perfect']==sum(m['perfect'] for m in e['meshes'])
    s=0.
    for m in e['meshes']:s+=m['edge_soft4']
    # Match server Python's sequential FP64 summation, including on Python >=3.12.
    f=0.
    for m in e['meshes']:f+=m['face_soft4']
    assert e['edge_objective']==s/100 and e['face_objective']==f/100
    assert e['objective']==e['edge_objective']+e['face_objective']
base={m['uid'] for m in actual[0]['meshes'] if m['joint_perfect']};assert len(base)==71
for e,r in zip(actual,read('retention.json')):
    assert [m['uid'] for m in e['meshes']]==uids
    for m,ed in zip(e['meshes'],trace[e['step']]['meshes']):
        assert m['face']['complete'] and m['face']['tp']+m['face']['fn']==m['gt_faces']
        assert m['gt_face_candidates']+m['missing_gt_face_candidates']==m['gt_faces']
        assert m['face']['fn']>=m['missing_gt_face_candidates']
        assert all(m['edge'][k]==ed[k] for k in ['tp','fp','fn','tn'])
        assert m['edge_perfect']==(m['edge']['fp']==m['edge']['fn']==0)
        assert m['face_perfect']==(m['face']['fp']==m['face']['fn']==0)
        assert m['joint_perfect']==(m['edge_perfect'] and m['face_perfect'])
    for kind in ['edge','face']:
        for k in ['tp','fp','fn']:assert e[kind+'_'+k]==sum(m[kind][k] for m in e['meshes'])
    for k in ['edge_perfect','face_perfect','joint_perfect']:assert e[k]==sum(m[k] for m in e['meshes'])
    for k in ['edge_perfect','joint_perfect']:
        now={m['uid'] for m in e['meshes'] if m[k]}
        assert set(r[k]['retained'])==base&now and set(r[k]['lost'])==base-now and set(r[k]['gained'])==now-base
assert complete['initial']==actual[0] and complete['final']==actual[-1]
for a,b in zip(read('run/final_real_network.json')['meshes'],actual[-1]['meshes']):assert {k:v for k,v in a.items() if k!='face_seconds'}=={k:v for k,v in b.items() if k!='face_seconds'}
for k in ['original_three_adam_groups_final1500','new_decoder14_adam_final500','endpoint_tail_checkpoint_reloaded','all100_real_network_matches_cache','frozen_parameters_buffers_metadata_unchanged','face_head_trainable','face_adam_final_step1500','parent_joint_full_model_unchanged','parent_joint_adam_checkpoint_unchanged','cached_pre_block14_inputs_unchanged','face_candidates_reenumerated_from_each_predicted_edge_graph','all_actual_face_evaluations_complete']:assert verify[k]
parent=read('parent_actual_baseline.json')
for a,b in zip(actual[0]['meshes'],parent['meshes']):
    assert {k:v for k,v in a.items() if k!='face_seconds'}=={k:v for k,v in b.items() if k!='face_seconds'}
assert restore['new_decoder14_adam_empty'] and 'decoder_blocks.14.norm' in restore['cache_boundary']
control=[json.loads(s) for s in z.read('control_evidence/run/actual_evaluations.jsonl').decode().splitlines()]
assert [e['step'] for e in control]==[0,100,200,300,400,500]
for a,b in zip(control[0]['meshes'],actual[0]['meshes']):assert {k:v for k,v in a.items() if k!='face_seconds'}=={k:v for k,v in b.items() if k!='face_seconds'}
recovery=read('resume_replay_verification.json')
assert recovery['replayed_updates']==list(range(101,180)) and recovery['all_forward_rows_gradients_clipping_and_actual_updates_exact']
old_updates=[json.loads(t) for t in z.read('interrupted_attempt0/run/updates.jsonl').decode().splitlines()]
old_trace=[json.loads(t) for t in z.read('interrupted_attempt0/run/edge_trace.jsonl').decode().splitlines()]
assert old_updates==updates[:179] and old_trace==trace[:179]
assert complete['total_optimizer_executions_including_lost_attempt']==579
assert read('cache_gradient_verification.json')['all100_real_vs_cached_combined_gradient_bitwise']
checked=0
if f'run/final_outputs/{uids[0]}.npz' in z.namelist():
    for uid,m in zip(uids,trace[-1]['meshes']):
        with np.load(io.BytesIO(z.read(f'run/final_outputs/{uid}.npz'))) as a:
            n=len(a['vertices']);i,j=np.triu_indices(n,1);gt=a['edges'];keys=gt[:,0]*n+gt[:,1];q=i*n+j;at=np.searchsorted(keys,q);y=(at<len(keys))&(keys[np.minimum(at,len(keys)-1)]==q);pred=a['edge_logits']>0
            count=dict(tp=int((pred&y).sum()),fp=int((pred&~y).sum()),fn=int((~pred&y).sum()),tn=int((~pred&~y).sum()))
            assert all(m[k]==v for k,v in count.items());checked+=len(pred)
    assert checked==84669234
result=dict(package=str(p),hashes_verified=len(hashes),new_updates=500,extra_replayed_executions=79,total_optimizer_executions=579,edge_eval_records=50100,actual_eval_records=600,retention_verified=True,final_saved_edge_logits_recounted=checked,scope='Local read-only ZIP/hash/count audit. Full-network, frozen-parameter and optimizer-state checks are server evidence, not rerun locally.')
p.with_suffix('.local-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
