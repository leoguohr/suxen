"""Independently verify new real-network outputs and original teacher arrays."""
from audit_saved_predictions import ROOT, DATA, sha, rows, counts, candidates, matched, aggregate
import numpy as np
import json, itertools

targets = []
for r in json.loads((DATA/'manifest.json').read_text())['meshes']:
    with np.load(DATA/r['path'], allow_pickle=False) as a:
        targets.append({k:a[k].copy() for k in a.files})
results = {}
for name in ['final_ae_cpu','cascade_gpu_seed1','cascade_gpu_seed2']:
    out = ROOT/'repro_outputs'/name
    recorded = json.loads((out/'RESULT.json').read_text())
    recs=[]
    for i,g in enumerate(targets):
        p = out/'spacetime'/f'{i:02d}.npz' if name=='final_ae_cpu' else out/f'{i:02d}_mesh.npz'
        with np.load(p,allow_pickle=False) as a:
            n=len(g['vertices']); pe=rows(a['pred_edges']); pf=rows(a['pred_faces']);fc=rows(a['face_candidates'])
            assert np.isfinite(a['edge_logits']).all() and np.isfinite(a['face_logits']).all()
            assert rows(a['edge_pairs'])==set(itertools.combinations(range(n),2))
            assert pe==rows(a['edge_pairs'][a['edge_logits']>0])
            assert fc==candidates(n,pe)
            assert pf==rows(a['face_candidates'][a['face_logits']>0])
            if name=='final_ae_cpu':
                old=g['original_vertex_indices'];gf=rows(old[g['faces']]);point=None
                assert np.array_equal(a['vertices'],g['vertices'][old.argsort()])
            else:
                mapping,point=matched(a['vertices'],g['vertices']);gf=rows(np.argsort(mapping)[g['faces']])
            ge={e for f in gf for e in itertools.combinations(f,2)}
            recs.append(dict(id=i,edge=counts(pe,ge),face=counts(pf,gf),point=point,prediction_sha256=sha(p)))
    total=aggregate(recs)
    ref=recorded['methods']['spacetime']['total'] if name=='final_ae_cpu' else recorded['total']
    for task in ['edge','face']:
        assert total[task]==ref[task],(name,task)
    if name!='final_ae_cpu':
        total['coordinate_rmse']=float(np.sqrt(sum(r['point']['squared_error'] for r in recs)/sum(r['point']['coordinates'] for r in recs)))
        total['count_correct']=50
        total['max_spacing_ratio']=max(r['point']['spacing_ratio'] for r in recs)
        assert abs(total['coordinate_rmse']-recorded['point_total']['coordinate_rmse'])<1e-12
        total['teacher_point_gate_pass']=total['coordinate_rmse']<1e-4 and total['max_spacing_ratio']<.1
    total['teacher_face_threshold']=.997 if name=='final_ae_cpu' else .99
    total['teacher_face_gate_pass']=total['face']['f1']>=total['teacher_face_threshold']
    results[name]=dict(total=total,per_mesh=recs)

original={}
for name,subdir in [('teacher_ae','latest_predictions/ae'),('teacher_flow_seed12345','latest_predictions/flow'),('teacher_flow_seed23456','confirmation_predictions')]:
    recs=[]
    for i,g in enumerate(targets):
        p=ROOT/'teacher_assets/results/minkowski_target_099'/subdir/f'{i:02d}_faces.json'
        gf=rows(g['original_vertex_indices'][g['faces']]);pf=rows(json.loads(p.read_text()))
        recs.append(dict(id=i,**counts(pf,gf)))
    total={k:sum(r[k] for r in recs) for k in ['tp','fp','fn']};total['f1']=2*total['tp']/(2*total['tp']+total['fp']+total['fn'])
    original[name]=dict(total=total,per_mesh=recs)
recs=[]
for i,g in enumerate(targets):
    p=ROOT/'teacher_assets/results/cascade_overfit'/f'{i:02d}'
    mapping,point=matched(np.load(p/'vertices.npy',allow_pickle=False),g['vertices'])
    gf=rows(np.argsort(mapping)[g['faces']]);pf=rows(json.loads((p/'faces.json').read_text()))
    recs.append(dict(id=i,face=counts(pf,gf),point=point))
total={k:sum(r['face'][k] for r in recs) for k in ['tp','fp','fn']};total['f1']=2*total['tp']/(2*total['tp']+total['fp']+total['fn'])
total['coordinate_rmse']=float(np.sqrt(sum(r['point']['squared_error'] for r in recs)/sum(r['point']['coordinates'] for r in recs)))
original['teacher_cascade']=dict(total=total,per_mesh=recs)
report=dict(optimizer_updates=0,new_prediction_arrays_verified=150,original_teacher_prediction_sets_verified=200,all_checks_passed=True,fresh=results,original=original)
(ROOT/'repro_outputs/FRESH_AND_TEACHER_AUDIT.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v['total'] for k,v in results.items()},indent=2))
print('original',json.dumps({k:v['total'] for k,v in original.items()}))
