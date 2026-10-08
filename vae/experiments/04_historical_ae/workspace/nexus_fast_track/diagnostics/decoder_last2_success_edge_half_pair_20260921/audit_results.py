"""Independent audit of two finite training budgets and saved actual predictions."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
import torch as T
T.set_num_threads(1)
R=Path(__file__).resolve().parent
cfg=json.loads((R/'config.json').read_text());complete=json.loads((R/'complete.json').read_text())
assert complete['total_branch_optimizer_steps']==1000 and complete['per_branch_updates']==500
partition=json.loads((R/'partition.json').read_text());S=set(partition['success71']);F=set(partition['failed29'])
protocol=json.loads((R/'protocol_verification.json').read_text());original72=set(protocol['original_joint72'])
parts=['success_edge','success_face','failed_edge','failed_face']
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def csvout(name,rows):
    with (R/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ekeys(a,n): return a[:,0].astype(np.int64)*n+a[:,1]
def fkeys(a,n): return (a[:,0].astype(np.int64)*n+a[:,1])*n+a[:,2]
def same(a,b):
    if T.is_tensor(a): assert T.equal(a.cpu(),b.cpu()) and a.dtype==b.dtype
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
all_evals={};trends=[];meshes=[];pred_verified=[];checkpoints=[]
parent=T.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
gv=T.load(R.parent/'decoder_last2_gradient_groups_20260920/gradient_vectors.pt',map_location='cpu',mmap=True,weights_only=False)
for branch,coefficient in cfg['branches'].items():
    b=R/branch
    first=T.load(b/'first_step_tensors.pt',map_location='cpu',mmap=True,weights_only=False)
    names=first['parameter_names'];assert names==gv['parameter_names']
    exact=[first['theta_after'][n].double()-first['theta_before'][n].double() for n in names]
    assert all(T.equal(first['theta_after'][n]-first['theta_before'][n],dd) for n,dd in zip(names,first['actual_fp32_displacement']))
    exact_dot={k:sum(float((g.double()*dd).sum()) for g,dd in zip(gv['gradients'][k],exact)) for k in parts}
    (b/'first_step_exact_displacement.json').write_text(json.dumps(dict(
        definition='FP64 difference of the actual before/after FP32 parameter values; no optimizer execution',
        gradient_dot_displacement=exact_dot,actual_delta_l2=math.sqrt(sum(float(dd.square().sum()) for dd in exact)),
        source_tensor_sha256=sha(b/'first_step_tensors.pt'),
        fp32_subtraction_also_saved=True),indent=2)+'\n')
    del first,exact
    logs=[json.loads(l) for l in (b/'run/updates.jsonl').read_text().splitlines()]
    trace=[json.loads(l) for l in (b/'run/training_trace.jsonl').read_text().splitlines()]
    assert [r['update'] for r in logs]==list(range(1,501))
    assert [r['new_step'] for r in trace]==list(range(501))
    assert all([r['uid'] for r in t['meshes']]==protocol['uids'] for t in trace)
    for step,log in enumerate(logs,1):
        assert log['mesh_count']==100 and log['success_edge_coefficient']==coefficient
        assert all(x['delta_norm']>0 and math.isfinite(x['delta_norm']) for x in log['actual_updates'].values())
        assert log['adam_steps']==dict(decoder_tail=[1500+step],edge_head=[1500+step],face_head=[1500+step],decoder14=[500+step])
        assert log['components_before']==trace[step-1]['components']
    all_evals[branch]={}
    for step in cfg['checkpoints']:
        p=b/'run'/f'checkpoint-new{step:04d}.pt';e=json.loads((b/'run'/f'actual-new{step:04d}.json').read_text())
        assert sha(p)==e['checkpoint_sha256']
        cp=T.load(p,map_location='cpu',mmap=True,weights_only=False)
        assert cp['new_updates']==step and all(v==step for v in cp['per_uid_new_training_participations'].values())
        assert cp['branch']==branch and cp['success_edge_coefficient']==coefficient
        for g in cp['optimizer']['param_groups']:
            assert g['lr']==cfg['lr'][g['name']] and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
            old=500 if g['name']=='decoder14' else 1500
            assert all(float(cp['optimizer']['state'][i]['step'])==old+step for i in g['params'])
        if step==0:same(cp['tail'],parent['tail']);same(cp['optimizer'],parent['optimizer']);same(cp['rng'],parent['rng']);same(cp['cuda_rng'],parent['cuda_rng'])
        assert len(e['meshes'])==100 and [r['uid'] for r in e['meshes']]==protocol['uids']
        components={k:0. for k in parts};joint=set()
        for row,file in zip(e['meshes'],e['prediction_files']):
            uid=row['uid'];prefix='success' if uid in S else 'failed';assert row['fixed_group']==('S' if uid in S else 'F')
            components[prefix+'_edge']+=row['edge_soft4']/100;components[prefix+'_face']+=row['face_soft4']/100
            if row['joint_perfect']:joint.add(uid)
            path=R/file['path'];assert sha(path)==file['sha256'] and uid==file['uid']
            with np.load(path) as q:
                n=len(q['vertices']);pe=q['predicted_edge_ids'];el=q['predicted_edge_logits'];ge=q['gt_edges'];gf=q['gt_faces']
                ek=ekeys(pe,n);gk=ekeys(ge,n);labels=np.isin(ek,gk)
                assert len(np.unique(ek))==len(ek) and (pe[:,0]<pe[:,1]).all() and np.isfinite(el).all() and (el>0).all()
                tp=int(labels.sum());fp=len(labels)-tp;fn=len(ge)-tp
                assert [tp,fp,fn]==[row['edge'][k] for k in ['tp','fp','fn']]
                fi=q['actual_face_candidate_ids'];fl=q['actual_face_candidate_logits'];fk=fkeys(fi,n);gft=fkeys(gf,n)
                y=np.isin(fk,gft);pred=fl>0
                assert len(np.unique(fk))==len(fk) and np.isfinite(fl).all() and np.array_equal(y,q['actual_face_candidate_gt'])
                assert (fi[:,0]<fi[:,1]).all() and (fi[:,1]<fi[:,2]).all()
                for i,j in [(0,1),(0,2),(1,2)]:assert np.isin(fi[:,i].astype(np.int64)*n+fi[:,j],ek).all()
                ftp=int((y&pred).sum());ffp=int((~y&pred).sum());ffn=len(gf)-ftp
                assert [ftp,ffp,ffn]==[row['face'][k] for k in ['tp','fp','fn']]
                covered=np.ones(len(gf),dtype=bool)
                for i,j in [(0,1),(0,2),(1,2)]:covered&=np.isin(gf[:,i].astype(np.int64)*n+gf[:,j],ek)
                assert np.array_equal(covered,q['gt_face_covered'])
                assert int((~covered).sum())==row['missing_gt_face_candidates']
                # Independent set-intersection count: all predicted graph 3-cliques are present.
                neighbors=[set() for _ in range(n)]
                for i,j in pe:neighbors[int(i)].add(int(j))
                triangle_count=sum(len(neighbors[i].intersection(neighbors[j])) for i,js in enumerate(neighbors) for j in js)
                assert triangle_count==len(fi)==row['face']['scored_candidates']
            assert row['joint_perfect']==(fp==fn==ffp==ffn==0)
            meshes.append(dict(branch=branch,new_updates=step,uid=uid,vertices=n,group=row['fixed_group'],
                edge_soft4=row['edge_soft4'],face_soft4=row['face_soft4'],
                **{kind+'_'+k:row[kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},
                missing_gt_face_candidates=row['missing_gt_face_candidates'],joint_perfect=row['joint_perfect'],
                **row['margins']))
        assert components==e['components']==trace[step]['components']
        assert e['original_objective']==sum(components.values())
        assert e['optimized_objective']==sum(components.values())+(coefficient-1)*components['success_edge']
        assert sorted(joint)==e['joint_perfect_uids'] and sorted(joint&original72)==e['retained_source72']
        assert sorted(original72-joint)==e['lost_source72'] and sorted(joint-original72)==e['new_over_source72']
        for group in ['all','S','F']:
            rows=[r for r in e['meshes'] if group=='all' or r['fixed_group']==group];summary=e['summary'][group]
            assert len(rows)==summary['meshes']
            assert sum(r['joint_perfect'] for r in rows)==summary['joint_perfect']
            for kind in ['edge','face']:
                for k in ['tp','fp','fn','tn']:assert sum(r[kind][k] for r in rows)==summary[kind][k]
        a=e['summary']['all']
        trends.append(dict(branch=branch,new_updates=step,success_edge_coefficient=coefficient,**components,
            original_objective=e['original_objective'],optimized_objective=e['optimized_objective'],
            edge_perfect=a['edge_perfect'],joint_perfect=a['joint_perfect'],
            **{kind+'_'+k:a[kind][k] for kind in ['edge','face'] for k in ['fp','fn']},
            retained72=len(e['retained_source72']),lost72=len(e['lost_source72']),new_success=len(e['new_over_source72']),
            S_edge_fp=e['summary']['S']['edge']['fp'],S_edge_fn=e['summary']['S']['edge']['fn'],
            F_edge_fp=e['summary']['F']['edge']['fp'],F_edge_fn=e['summary']['F']['edge']['fn']))
        pred_verified.append(dict(branch=branch,step=step,npz_files=100,all_prediction_counts_and_complete_graph_triangles_verified=True))
        checkpoints.append(dict(branch=branch,step=step,sha256=e['checkpoint_sha256'],adam_states_steps_groups_checked=True))
        all_evals[branch][step]=e
        print('AUDIT',branch,step,flush=True)
    final=T.load(b/'run/checkpoint-new0500.pt',map_location='cpu',mmap=True,weights_only=False)
    fullpath=Path(complete['branches'][branch]['full_model']);assert sha(fullpath)==complete['branches'][branch]['full_model_sha256']
    full=T.load(fullpath,map_location='cpu',mmap=True,weights_only=False)
    mapping={'block':'autoencoder.decoder_blocks.15','final_norm':'autoencoder.decoder_output_norm',
             'head':'autoencoder.edge_embedding','face':'autoencoder.face_embedding','penultimate':'autoencoder.decoder_blocks.14'}
    for n,t in final['tail'].items():
        module,rest=n.split('.',1);assert T.equal(t,full['model'][mapping[module]+'.'+rest])
    del final,full
assert all_evals['A_control'][0]['components']==all_evals['B_success_edge_half'][0]['components']
assert all_evals['A_control'][0]['summary']==all_evals['B_success_edge_half'][0]['summary']
A=all_evals['A_control'][500];B=all_evals['B_success_edge_half'][500]
comparison=dict(A=A['summary'],B=B['summary'],component_difference_B_minus_A={k:B['components'][k]-A['components'][k] for k in parts},
    A_retained72=A['retained_source72'],B_retained72=B['retained_source72'],A_lost72=A['lost_source72'],B_lost72=B['lost_source72'],
    A_new=A['new_over_source72'],B_new=B['new_over_source72'],
    B_only_joint_success=sorted(set(B['joint_perfect_uids'])-set(A['joint_perfect_uids'])),
    A_only_joint_success=sorted(set(A['joint_perfect_uids'])-set(B['joint_perfect_uids'])))
(R/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
csvout('actual_trend.csv',trends);csvout('actual_per_mesh.csv',meshes)
paired=[]
for step in cfg['checkpoints']:
    for a,b in zip(all_evals['A_control'][step]['meshes'],all_evals['B_success_edge_half'][step]['meshes']):
        paired.append(dict(step=step,uid=a['uid'],vertices=a['vertices'],group=a['fixed_group'],
            A_joint=a['joint_perfect'],B_joint=b['joint_perfect'],
            **{tag+'_'+kind+'_'+k:r[kind][k] for tag,r in [('A',a),('B',b)] for kind in ['edge','face'] for k in ['fp','fn']},
            A_edge_loss=a['edge_soft4'],B_edge_loss=b['edge_soft4'],A_face_loss=a['face_soft4'],B_face_loss=b['face_soft4']))
csvout('paired_per_mesh.csv',paired);csvout('paired_large_mesh.csv',[r for r in paired if r['vertices']>1500])
(R/'independent_audit.json').write_text(json.dumps(dict(completed_updates_per_branch=500,total_actual_optimizer_executions=1000,
    A_first_update_resumed_not_replayed=True,checkpoint_checks=checkpoints,prediction_checks=pred_verified,
    identical_step0_weights_adam_rng=True,continuous_logs_and_fixed_uid_order=True,
    all100_real_network_evidence_per_checkpoint=True,full_model_tail_weights_match_final_checkpoint=True,
    verification_scope='Independent CPU file/state/prediction audit; no independent network rerun'),indent=2)+'\n')
print('AUDIT COMPLETE',flush=True)
