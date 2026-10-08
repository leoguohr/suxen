"""CPU-only independent checks of saved parameter points, predictions and counts."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv, hashlib, json, math
from pathlib import Path
import torch as T

R=Path(__file__).resolve().parent
T.set_num_threads(1)
d=json.loads((R/'result.json').read_text())
m=d['manifest']; parts=m['parts']
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def dot(xs,ys): return sum(float((x.double()*y.double()).sum()) for x,y in zip(xs,ys))
def close(x,y): assert math.isclose(x,y,rel_tol=2e-10,abs_tol=1e-13),(x,y)
def write_csv(name,rows):
    with (R/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
assert sha(m['source_checkpoint'])==m['source_sha256']
assert sha(m['saved_vectors'])==m['saved_vectors_sha256']
theta=T.load(m['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)['tail']
v=T.load(m['saved_vectors'],map_location='cpu',mmap=True,weights_only=False)
names=v['parameter_names']; delta=v['cloned_adam_actual_fp32_displacement']
partition=json.loads((R/'partition.json').read_text())
success=set(partition['success71']); failed=set(partition['failed29'])
baseline=next(r for r in d['points'] if r['lambda']==0)
base_losses=baseline['losses']
checks=[]; table=[]; mesh_table=[]; count_table=[]
for point in d['points']:
    path=R/'points'/point['point']/'tail-fp32.pt'
    assert sha(path)==point['state_sha256']
    state=T.load(path,map_location='cpu',mmap=True,weights_only=False)
    value=point['lambda']
    # Independent arithmetic in FP64, rounded once to FP32, from original theta0.
    assert all(T.equal(state[n],(theta[n].double()+value*dd.double()).float()) for n,dd in zip(names,delta))
    actual=[state[n].double()-theta[n].double() for n in names]
    dn=math.sqrt(dot(actual,actual)); close(dn,point['geometry']['norm'])
    rows=[json.loads(l) for l in (path.parent/'meshes.jsonl').read_text().splitlines()]
    assert len(rows)==100 and {r['uid'] for r in rows}==success|failed
    assert all(r['face']['complete'] for r in rows)
    losses={p:0. for p in parts}
    for r in rows:
        group='success' if r['uid'] in success else 'failed'
        assert r['fixed_group']==group
        losses[group+'_edge']+=r['edge_soft4']/100
        losses[group+'_face']+=r['face_soft4']/100
        assert r['edge_perfect']==(r['edge']['fp']==r['edge']['fn']==0)
        assert r['face_perfect']==(r['face']['fp']==r['face']['fn']==0)
        assert r['joint_perfect']==(r['edge_perfect'] and r['face_perfect'])
        mesh_table.append(dict(point=point['point'],lambda_value=value,uid=r['uid'],group=group,vertices=r['vertices'],
            edge_loss=r['edge_soft4'],face_loss=r['face_soft4'],
            **{'edge_'+k:r['edge'][k] for k in ['tp','fp','fn']},
            **{'face_'+k:r['face'][k] for k in ['tp','fp','fn']},
            missing_gt_face_candidates=r['missing_gt_face_candidates'],
            actual_face_candidates=r['face']['scored_candidates'],face_fp_outside_pool=r['face']['actual_fp_outside_training_pool'],
            joint_perfect=r['joint_perfect'],**r['margins']))
    assert losses==point['losses']
    for part in parts:
        pred=dot(v['gradients'][part],actual)
        close(pred,point['geometry']['predictions'][part])
        change=losses[part]-base_losses[part]
        close(change,point['comparisons'][part]['delta_loss'])
        table.append(dict(point=point['point'],lambda_value=value,part=part,loss=losses[part],delta_loss=change,
            prediction=pred,residual=change-pred,ratio=change/pred if abs(pred)>1e-12 else None,
            actual_displacement_l2=dn,actual_relative_displacement=point['geometry']['relative_norm']))
    for group in ['all','success','failed']:
        rs=[r for r in rows if group=='all' or r['fixed_group']==group]
        for kind in ['edge','face']:
            for key in ['tp','fp','fn','tn']:
                assert sum(r[kind][key] for r in rs)==point['summary'][group][kind][key]
        assert sum(r['joint_perfect'] for r in rs)==point['summary'][group]['joint_perfect']
        count_table.append(dict(point=point['point'],lambda_value=value,group=group,
            edge_perfect=point['summary'][group]['edge_perfect'],joint_perfect=point['summary'][group]['joint_perfect'],
            **{kind+'_'+k:point['summary'][group][kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},
            missing_gt_face_candidates=point['summary'][group]['missing_gt_face_candidates']))
    checks.append(dict(point=point['point'],state_sha256=point['state_sha256'],independent_fp64_then_fp32_construction=True,
        actual_displacement_and_predictions_recomputed=True,all100_counts_and_losses_recomputed=True))
assert len(checks)==11 and len(mesh_table)==1100
assert d['verification']['optimizer_steps']==d['verification']['backward_calls']==0
write_csv('loss_displacement.csv',table)
write_csv('actual_counts.csv',count_table)
write_csv('actual_per_mesh.csv',mesh_table)
write_csv('centered_secants.csv',d['centered_secants'])
(R/'independent_audit.json').write_text(json.dumps(dict(points=checks,point_count=11,mesh_count=1100,
    complete_face_enumerations=1100,source_checkpoint_hash_verified=True,saved_vector_hash_verified=True,
    original_model_not_updated=True,independent_forward_rerun=False),indent=2)+'\n')
print('AUDIT PASSED: 11 independent saved FP32 points, 1100 mesh records; no optimizer')
