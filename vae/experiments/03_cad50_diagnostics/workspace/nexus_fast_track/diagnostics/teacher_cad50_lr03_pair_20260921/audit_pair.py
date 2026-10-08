"""Read-only audit/report/package of both finite CAD50 branches; no optimizer execution."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
from loader import sha

R=Path(__file__).resolve().parent
P=R.parent/'teacher_cad50_fresh512_20260921'
SOURCE=P/'run/checkpoint-step2000.pt'
SOURCE_SHA='ae7c2835fe7ad9da72edd413f66373b415721b2cfac72df8c6b3fc141b987358'
BRANCHES=['A_control_lr1','B_lr03']
assert sha(SOURCE)==SOURCE_SHA
parent=json.loads((R/'parent_eval-step2000.json').read_text());partition=json.loads((R/'partition.json').read_text())
assert parent['perfect_uids']==partition['success30']
data_manifest=json.loads((P/'data/manifest.json').read_text());uids=data_manifest['uids']
for x in data_manifest['meshes']:assert sha(P/'data'/x['path'])==x['sha256']
pool_manifest=json.loads((P/'pool_manifest.json').read_text())
for x in pool_manifest['records']:assert sha(P/x['path'])==x['sha256']
artifacts=[dict(path=str(SOURCE),bytes=SOURCE.stat().st_size,sha256=SOURCE_SHA,role='common parent')]
flat=[];trends=[];step_mesh=[];all_eval={};stability={};verified_predictions=0;verified_checkpoints=0

def percentile(values):
    x=np.asarray(values,dtype=np.float64)
    assert np.isfinite(x).all()
    return dict(max=float(x.max()),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),median=float(np.median(x)))

def same(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

parent_state=torch.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
for branch in BRANCHES:
    out=R/branch;done=json.loads((out/'complete.json').read_text());cfg=json.loads((out/'config.json').read_text())
    assert done['stopped_at_budget'] and done['new_updates']==500 and done['completed_updates']==2500
    assert not (out/'failure.json').exists()
    logs=[json.loads(x) for x in (out/'updates.jsonl').read_text().splitlines()]
    assert [x['new_update'] for x in logs]==list(range(1,501))
    for x in logs:
        assert x['completed_updates']==2000+x['new_update'] and x['state_before']==x['completed_updates']-1
        assert [m['uid'] for m in x['meshes']]==uids and x['participation_per_mesh']==x['completed_updates']
        assert x['lrs']==cfg['lrs'] and set(x['adam_steps'].values())=={x['completed_updates']}
        assert np.isfinite(x['total_loss_before']) and all(np.isfinite(v['delta_l2']) for v in x['actual_updates'].values())
        assert x['edge_perfect_uids_before']==[m['uid'] for m in x['meshes'] if m['fp']==m['fn']==0]
        for m in x['meshes']:step_mesh.append(dict(branch=branch,new_update=x['new_update'],state_before=x['state_before'],**m))
    top=sorted(logs,key=lambda x:x['total_loss_before'],reverse=True)[:10]
    spikes=[]
    for i,x in enumerate(logs):
        if i<20:continue
        baseline=float(np.median([v['total_loss_before'] for v in logs[i-20:i]]))
        if x['total_loss_before']>2*baseline:spikes.append(x)
    def detail(x):
        return dict(new_update=x['new_update'],state_before=x['state_before'],total_loss=x['total_loss_before'],
            gradient_norm=x['total_grad_norm'],actual_updates=x['actual_updates'],
            largest_uid_losses=sorted(x['meshes'],key=lambda m:m['edge']+m['face'],reverse=True)[:8])
    stability[branch]=dict(total_loss=percentile([x['total_loss_before'] for x in logs]),
        gradient_norm=percentile([x['total_grad_norm'] for x in logs]),
        top10_total_loss=[detail(x) for x in top],
        spike_definition='pre-update total loss >2x median of previous20 pre-update records; descriptive only',
        spikes=[detail(x) for x in spikes],
        watched_uid_losses={uid:dict(edge=percentile([next(m['edge'] for m in x['meshes'] if m['uid']==uid) for x in logs]),
            face=percentile([next(m['face'] for m in x['meshes'] if m['uid']==uid) for x in logs])) for uid in ['teacher_cad50_12','teacher_cad50_28']})
    es=[]
    for step in range(0,501,50):
        e=json.loads((out/f'eval-new{step:04d}.json').read_text());es.append(e)
        assert e['new_step']==step and e['completed_updates']==2000+step and [x['uid'] for x in e['meshes']]==uids
        checkpoint=Path(e['checkpoint']);assert sha(checkpoint)==e['checkpoint_sha256']
        cp=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        assert cp['completed_updates']==2000+step and set(cp['participation'].values())=={2000+step}
        assert all(int(s['step'])==2000+step for s in cp['optimizer']['state'].values())
        assert len(cp['optimizer']['state'])==sum(len(g['params']) for g in cp['optimizer']['param_groups'])
        assert {g['name']:g['lr'] for g in cp['optimizer']['param_groups']}==cfg['lrs']
        for g in cp['optimizer']['param_groups']:
            assert tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
        for name,v in cp['model'].items():
            if name.startswith('autoencoder.log_variance.'):assert same(v,parent_state['model'][name])
        if step==0:
            assert same(cp['model'],parent_state['model']) and same(cp['rng'],parent_state['rng'])
            assert same(cp['optimizer']['state'],parent_state['optimizer']['state'])
            for a,b in zip(cp['optimizer']['param_groups'],parent_state['optimizer']['param_groups']):
                assert {k:v for k,v in a.items() if k!='lr'}=={k:v for k,v in b.items() if k!='lr'}
        artifacts.append(dict(branch=branch,path=str(checkpoint),sha256=e['checkpoint_sha256'],bytes=checkpoint.stat().st_size,new_step=step))
        verified_checkpoints+=1
        for row in e['meshes']:
            path=R/row['prediction_path'];assert sha(path)==row['prediction_sha256']
            with np.load(path) as q:
                n=len(q['vertices']);pe=q['predicted_edge_ids'];ge=q['gt_edges'];fi=q['actual_face_candidate_ids'];gf=q['gt_faces']
                ek=lambda a:a[:,0].astype(np.int64)*n+a[:,1]
                fk=lambda a:(a[:,0].astype(np.int64)*n+a[:,1])*n+a[:,2]
                pred_edges=ek(pe);assert len(np.unique(pred_edges))==len(pe) and (q['predicted_edge_logits']>0).all()
                tp=int(np.isin(pred_edges,ek(ge)).sum());fp=len(pe)-tp;fn=len(ge)-tp
                assert [tp,fp,fn]==[row['edge'][k] for k in ['tp','fp','fn']]
                labels=np.isin(fk(fi),fk(gf));pred=q['actual_face_candidate_logits']>0
                assert len(np.unique(fk(fi)))==len(fi) and np.array_equal(labels,q['actual_face_candidate_gt'])
                tp=int((pred&labels).sum());fp=int((pred&~labels).sum());fn=len(gf)-tp
                assert [tp,fp,fn]==[row['face'][k] for k in ['tp','fp','fn']]
                adj=[set() for _ in range(n)]
                for i,j in pe:adj[int(i)].add(int(j))
                assert sum(len(adj[i]&adj[j]) for i in range(n) for j in adj[i])==len(fi)
                for i,j in [(0,1),(0,2),(1,2)]:assert np.isin(fi[:,i].astype(np.int64)*n+fi[:,j],pred_edges).all()
                covered=np.isin(fk(gf),fk(fi));assert row['missing_gt_face_candidates']==int((~covered).sum())
                assert np.array_equal(covered,q['gt_face_covered'])
                assert row['face']['fn_missing_candidate']+row['face']['fn_present_but_negative']==row['face']['fn']
                with np.load(P/'pools'/f"{row['uid']}.npz") as pool:
                    inpool=np.isin(fk(fi),fk(np.concatenate([pool['positive'],pool['mixed']])))
                assert int((pred&~labels&inpool).sum())==row['face']['actual_fp_inside_training_pool']
                assert int((pred&~labels&~inpool).sum())==row['face']['actual_fp_outside_training_pool']
                assert row['joint_perfect']==(row['edge']['fp']==row['edge']['fn']==row['face']['fp']==row['face']['fn']==0)
            verified_predictions+=1
            flat.append(dict(branch=branch,new_step=step,completed_updates=2000+step,uid=row['uid'],vertices=row['vertices'],
                parent_group='success30' if row['uid'] in partition['success30'] else 'failed20',**row['parts'],
                **{kind+'_'+k:row[kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},joint_perfect=row['joint_perfect'],
                face_fn_missing=row['face']['fn_missing_candidate'],face_fn_present=row['face']['fn_present_but_negative'],
                face_fp_in_pool=row['face']['actual_fp_inside_training_pool'],face_fp_out_pool=row['face']['actual_fp_outside_training_pool']))
        perfect=[x['uid'] for x in e['meshes'] if x['joint_perfect']]
        assert e['perfect_uids']==perfect and e['joint_perfect']==len(perfect)
        assert e['retained']==sorted(set(perfect)&set(partition['success30']))
        assert e['lost']==sorted(set(partition['success30'])-set(perfect))
        assert e['new']==sorted(set(perfect)-set(partition['success30']))
        trends.append(dict(branch=branch,new_step=step,joint=e['joint_perfect'],retained=len(e['retained']),lost=len(e['lost']),new=len(e['new']),
            large_joint=e['size_groups']['66_to_274']['joint_perfect'],**{kind+'_'+k:e['counts'][kind][k] for kind in ['edge','face'] for k in ['fp','fn']},**e['losses']))
        del cp
        print('AUDIT',branch,step,flush=True)
    all_eval[branch]=es
    full=Path(done['full_model']);assert sha(full)==done['full_model_sha256']
    artifacts.append(dict(branch=branch,path=str(full),sha256=done['full_model_sha256'],bytes=full.stat().st_size,role='full inference model'))

reps=[]
for folder in [R/'common/representations-new0000']+[R/b/'representations-new0500' for b in BRANCHES]:
    manifest=json.loads((folder/'manifest.json').read_text());assert len(manifest)==8
    for row in manifest:
        p=R/row['path'];assert sha(p)==row['sha256']
        with np.load(p) as q,np.load(P/'data/meshes'/f"{row['uid']}.npz") as data:
            n=len(data['vertices']);assert np.array_equal(q['vertices'],data['vertices']) and q['vertex_mask'].all()
            assert np.array_equal(q['local_vertex_indices'],np.arange(n))
            for key,dim in [('encoder_vertex_hidden',512),('mu',512),('decoder_hidden',1024),('edge_embedding',32),('face_embedding',32)]:
                assert q[key].shape==(n,dim) and q[key].dtype==np.float32 and np.isfinite(q[key]).all()
            assert len(q['edge_pair_ids'])==n*(n-1)//2 and len(q['gt_face_ids'])==len(data['face_set'])
        reps.append(row)
assert len(reps)==24 and verified_predictions==1100 and verified_checkpoints==22
assert sha(SOURCE)==SOURCE_SHA
def csv_write(name,rows):
    with (R/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csv_write('per_mesh.csv',flat);csv_write('actual_trend.csv',trends);csv_write('step_per_uid.csv',step_mesh)
csv_write('watched_uids.csv',[x for x in step_mesh if x['uid'] in ['teacher_cad50_12','teacher_cad50_28','teacher_cad50_02','teacher_cad50_24','teacher_cad50_34']])
(R/'stability.json').write_text(json.dumps(stability,indent=2)+'\n')
(R/'MODEL_ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
audit=dict(branches=2,updates_each=500,final_steps_each=2500,verified_checkpoints=verified_checkpoints,
    prediction_npz_verified=verified_predictions,representation_npz_verified=len(reps),complete_triangle_enumerations=True,
    original_parent_unchanged=True,adam_and_rng_equal_at_start_except_lr=True,logvar_unchanged=True,
    no_extra_training=True,scope='independent CPU saved-state and prediction verification; no new model execution')
(R/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
a=all_eval[BRANCHES[0]][-1];b=all_eval[BRANCHES[1]][-1]
answer=f"同预算末尾，A联合严格成功{a['joint_perfect']}/50，B为{b['joint_perfect']}/50；B相对A{'扩大' if b['joint_perfect']>a['joint_perfect'] else '未扩大'}成功数量。B保留父成功{len(b['retained'])}/30，丢失{len(b['lost'])}条，新增{len(b['new'])}条；66—274点组A为{a['size_groups']['66_to_274']['joint_perfect']}/16，B为{b['size_groups']['66_to_274']['joint_perfect']}/16。"
lines=['# CAD50 step2000：原LR与0.3倍LR配对续训','',answer,'',
    '两支均各完成500次全50条累积更新并停止，各自累计2500；不是将两支相加为一条累计3000。只改变四组LR，完整继承同一父权重、Adam和全部RNG。',
    '父30/20分组固定，仅分析用；没有改变任何mesh权重。全部检查点来自真实完整网络，Face由当前预测Edge图完整枚举。原100条任务未启动、停止或改动，本实验仅使用启动时空闲的CAD GPU。','',
    '## 末尾及最佳完整检查点','',
    '| 分支 | 末尾联合 | 父30保留/丢失/新增 | 66—274点成功 | Edge FP/FN | 实际Face FP/FN | 最佳联合及新增step |',
    '|---|---:|---:|---:|---:|---:|---|']
for branch,e in [(BRANCHES[0],a),(BRANCHES[1],b)]:
    best=max(x['joint_perfect'] for x in all_eval[branch]);steps=[x['new_step'] for x in all_eval[branch] if x['joint_perfect']==best]
    lines.append(f"| {branch} | {e['joint_perfect']}/50 | {len(e['retained'])}/{len(e['lost'])}/{len(e['new'])} | {e['size_groups']['66_to_274']['joint_perfect']}/16 | {e['counts']['edge']['fp']}/{e['counts']['edge']['fn']} | {e['counts']['face']['fp']}/{e['counts']['face']['fn']} | {best}/50，{steps} |")
    lines += ['',f"{branch}末尾丢失UID：{e['lost']}；新增UID：{e['new']}。"]
lines+=['','## 完整检查点轨迹','','| 新增step | A联合 | B联合 | A Edge FP/FN | B Edge FP/FN | A实际Face FP/FN | B实际Face FP/FN |','|---:|---:|---:|---:|---:|---:|---:|']
for x,y in zip(all_eval[BRANCHES[0]],all_eval[BRANCHES[1]]):
    lines.append(f"| {x['new_step']} | {x['joint_perfect']} | {y['joint_perfect']} | {x['counts']['edge']['fp']}/{x['counts']['edge']['fn']} | {y['counts']['edge']['fp']}/{y['counts']['edge']['fn']} | {x['counts']['face']['fp']}/{x['counts']['face']['fn']} | {y['counts']['face']['fp']}/{y['counts']['face']['fn']} |")
lines+=['','## 稳定性（辅助指标）','','| 分支 | 总loss Max/P95/P99 | 裁剪前梯度范数 Max/P95/P99 |','|---|---|---|']
for branch,s in stability.items():
    fmt=lambda d:'/'.join(f"{d[k]:.9g}" for k in ['max','p95','p99'])
    lines.append(f"| {branch} | {fmt(s['total_loss'])} | {fmt(s['gradient_norm'])} |")
lines+=['','两支均用500条更新前记录，P95/P99使用numpy默认线性分位数。stability.json保留最高10次loss的UID分解与实际位移，并列出高于此前20步中位数2倍的事件。12、28两项loss和02、24、34的Edge计数见watched_uids.csv。该事后描述不参与回滚、选样或训练权重。',
    '每条训练日志明确state_before；执行累计2001前的loss属于父2000状态。逐步Edge不等于逐步实际Face验收。',
    '','## 限制与交付','',
    '更平滑或loss较低不能替代严格成功。若两支类似，不能把共同进步都归功于降LR；若以丢失旧成功换新增，必须报告取舍。本次有限配置不证明全局容量不足，也不保证继续训练必然50/50。',
    'Review包括代码、差异、配置、数据/pool清单、完整逐步日志、逐mesh评价、统计和审计。Predictions包括两支22点的1100份实际预测，以及共同起点和两支末尾24份指定样本表示。共同起点完整预测另外存于Review的common目录。',
    '中间表示均同一缓存顶点编号，位置/形状/mask/坐标见表示manifest。所有8条hook预检验证输出、完整参数梯度与RNG不变；没有表示MSE或额外目标，不据cosine/范数直接宣布坍缩。',
    '完整模型与可续训model/Adam/RNG留在服务器，路径/大小/SHA见MODEL_ARTIFACTS.json。训练实际执行，全部检查点实际完整网络验收；独立CPU审计没有额外执行模型或optimizer。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
review=[];predictions=[]
for p in R.rglob('*'):
    if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix in ['.pt','.tmp','.zip'] or p.name in ['package_manifest.json','execution.lock','launcher.lock']:continue
    name=str(p.relative_to(R))
    if p.suffix=='.npz' and (name.startswith(tuple(BRANCHES)) or 'representations-new' in name):predictions.append((p,name))
    elif p.suffix in ['.py','.sh','.md','.json','.jsonl','.csv','.log','.npz','.txt','.diff']:review.append((p,name))
for n in ['data/manifest.json','data/README.md']:
    if (P/n).exists():review.append((P/n,n))
for p in (P/'data/meshes').glob('*.npz'):review.append((p,str(p.relative_to(P))))
for p in (P/'pools').glob('*.npz'):review.append((p,str(p.relative_to(P))))
def package(name,files):
    path=R/name;manifest=[];seen=set()
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p,n in sorted(files,key=lambda x:x[1]):
            if n in seen:continue
            seen.add(n);manifest.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p)));z.write(p,n)
        z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,files=len(manifest))
packages=dict(review=package('TeacherCAD50_LR1_vs_LR03_500_Review.zip',review),
    predictions=package('TeacherCAD50_LR1_vs_LR03_500_Predictions_Representations.zip',predictions))
(R/'package_manifest.json').write_text(json.dumps(packages,indent=2)+'\n')
(R/'pair_complete.json').write_text(json.dumps(dict(state='complete',branches=BRANCHES,new_updates_each=500,stopped=True,answer=answer,packages=packages),indent=2)+'\n')
print('PAIR_COMPLETE',answer,json.dumps(packages),flush=True)
