"""Verify the bounded continuation and package its actual-structure results."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import torch as T
R=Path(__file__).resolve().parent;run=R/'run'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def table(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def same(a,b):
    if T.is_tensor(a):assert a.dtype==b.dtype and T.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
cfg=read(R/'config.json');complete=read(run/'complete.json');meta=read(R/'cache/manifest.json')
assert complete['state']=='complete' and complete['new_updates']==500 and complete['cumulative_tail_updates']==1500 and complete['cumulative_joint_updates']==1000 and complete['stopped_at_budget']
assert not (R/'failure.txt').exists()
trace=[json.loads(s) for s in (run/'edge_trace.jsonl').read_text().splitlines()]
updates=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()]
actual=[json.loads(s) for s in (run/'actual_evaluations.jsonl').read_text().splitlines()]
uids=[m['uid'] for m in meta['meshes']];sizes={m['uid']:m['vertices'] for m in meta['meshes']}
assert len(uids)==len(set(uids))==100 and uids==read(R/'selection.json')['uids']
assert [e['step'] for e in trace]==list(range(501)) and [u['update'] for u in updates]==list(range(1,501))
assert [e['step'] for e in actual]==cfg['checkpoint_steps']
for u in updates:
    assert u['mesh_count']==100 and u['cumulative_tail_update']==1000+u['update']
    assert u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4,'face_head':1e-4,'decoder14':1e-5} and all(x['delta_norm']>0 for x in u['actual_updates'].values())
old_edge={m['uid'] for m in trace[0]['meshes'] if m['perfect']};old_joint={m['uid'] for m in actual[0]['meshes'] if m['joint_perfect']};assert old_edge==old_joint and len(old_edge)==71
remaining=set(uids)-old_edge;assert len(remaining)==29
groups={'all100':set(uids),'initial_failed29':remaining,'N_gt1500':{u for u in uids if sizes[u]>1500},'initial_success71':old_edge}
group_trace=[];flat=[]
for e in trace:
    assert [m['uid'] for m in e['meshes']]==uids and e['cumulative_tail_updates']==e['step']+1000
    assert e['total_fp']==sum(m['fp'] for m in e['meshes']) and e['total_fn']==sum(m['fn'] for m in e['meshes']) and e['edge_perfect']==sum(m['perfect'] for m in e['meshes'])
    for name,ids in groups.items():
        ms=[m for m in e['meshes'] if m['uid'] in ids]
        group_trace.append(dict(new_step=e['step'],tail_step=e['step']+1000,group=name,meshes=len(ms),fp=sum(m['fp'] for m in ms),fn=sum(m['fn'] for m in ms),strict=sum(m['perfect'] for m in ms),mean_edge_soft4=sum(m['edge_soft4'] for m in ms)/len(ms)))
    for m in e['meshes']:flat.append(dict(new_step=e['step'],tail_step=e['step']+1000,vertices=sizes[m['uid']],**m))
retention=[];trend=[];faceflat=[]
for e in actual:
    assert [m['uid'] for m in e['meshes']]==uids
    er={m['uid']:m for m in trace[e['step']]['meshes']}
    for m in e['meshes']:
        assert m['face']['complete'] and all(m['edge'][k]==er[m['uid']][k] for k in ['tp','fp','fn','tn'])
        assert m['face']['tp']+m['face']['fn']==m['gt_faces'] and m['face']['fn']>=m['missing_gt_face_candidates']
        rr=dict(new_step=e['step'],tail_step=e['step']+1000,uid=m['uid'],vertices=m['vertices'],face_pool_soft4=m['face_pool_soft4'],**{kind+'_'+k:m[kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},**{k:m[k] for k in ['edge_perfect','face_perfect','joint_perfect','gt_face_candidates','missing_gt_face_candidates']},face_candidates=m['face']['scored_candidates'],face_fn_candidate_present=m['face']['fn']-m['missing_gt_face_candidates'])
        faceflat.append(rr)
    for kind in ['edge','face']:
        for k in ['tp','fp','fn']:assert e[kind+'_'+k]==sum(m[kind][k] for m in e['meshes'])
    for k in ['edge_perfect','face_perfect','joint_perfect']:assert e[k]==sum(m[k] for m in e['meshes'])
    ret=dict(step=e['step'])
    for key,base in [('edge_perfect',old_edge),('joint_perfect',old_joint)]:
        now={m['uid'] for m in e['meshes'] if m[key]};ret[key]=dict(retained=sorted(base&now),lost=sorted(base-now),gained=sorted(now-base),current=sorted(now))
    retention.append(ret)
    trend.append({k:v for k,v in e.items() if k!='meshes'})
assert complete['initial']==actual[0] and complete['final']==actual[-1]
real=read(run/'final_real_network.json')
for x,y in zip(real['meshes'],actual[-1]['meshes']):assert {k:v for k,v in x.items() if k!='face_seconds'}=={k:v for k,v in y.items() if k!='face_seconds'}
# Independently verify the saved optimizer start/end and persisted full-model freeze boundary.
parent=T.load(cfg['parent_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
first=T.load(run/'checkpoint-new0000-tail1000.pt',map_location='cpu',mmap=True,weights_only=False)
same(parent['tail'],{k:v for k,v in first['tail'].items() if not k.startswith('penultimate.')})
same(parent['optimizer']['state'],first['optimizer']['state']);same(parent['optimizer']['param_groups'],first['optimizer']['param_groups'][:3])
assert all(p not in first['optimizer']['state'] for p in first['optimizer']['param_groups'][3]['params'])
same(parent['rng'],first['rng']);same(parent['cuda_rng'],first['cuda_rng'])
assert all(float(v['step'])==1000 for v in first['optimizer']['state'].values())
last=T.load(run/'checkpoint-new0500-tail1500.pt',map_location='cpu',mmap=True,weights_only=False)
assert all(float(last['optimizer']['state'][p]['step'])==1500 for g in last['optimizer']['param_groups'][:3] for p in g['params'])
assert all(float(last['optimizer']['state'][p]['step'])==500 for p in last['optimizer']['param_groups'][3]['params'])
assert last['completed_updates']==1500 and last['new_updates']==500 and last['cumulative_joint_updates']==1000
source=T.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
derived=T.load(complete['inference_copy'],map_location='cpu',mmap=True,weights_only=False)
prefixes={'penultimate.':'autoencoder.decoder_blocks.14.','block.':'autoencoder.decoder_blocks.15.','final_norm.':'autoencoder.decoder_output_norm.','head.':'autoencoder.edge_embedding.','face.':'autoencoder.face_embedding.'}
mapping={k:next(v+k[len(p):] for p,v in prefixes.items() if k.startswith(p)) for k in last['tail']}
assert len(mapping)==18 and source['model'].keys()==derived['model'].keys()
for k,v in first['tail'].items():
    if k.startswith('penultimate.'):same(v,source['model'][mapping[k]])
for k,v in source['model'].items():
    if k not in mapping.values():same(v,derived['model'][k])
for k,dest in mapping.items():same(last['tail'][k],derived['model'][dest])
assert sha(cfg['source_checkpoint'])==cfg['source_sha256'] and sha(cfg['parent_checkpoint'])==cfg['parent_checkpoint_sha256'] and sha(complete['inference_copy'])==complete['inference_sha256']
del parent,first,last,source,derived
# Recount all saved final Edge logits, without invoking a model or optimizer.
checked=0
for uid,m in zip(uids,trace[-1]['meshes']):
    with np.load(run/'final_outputs'/f'{uid}.npz') as d:
        n=len(d['vertices']);a,z=np.triu_indices(n,1);gt=d['edges'];keys=gt[:,0]*n+gt[:,1];q=a*n+z;at=np.searchsorted(keys,q);y=(at<len(keys))&(keys[np.minimum(at,len(keys)-1)]==q);p=d['edge_logits']>0
        counts=dict(tp=int((p&y).sum()),fp=int((p&~y).sum()),fn=int((~p&y).sum()),tn=int((~p&~y).sum()))
        assert all(m[k]==v for k,v in counts.items());checked+=len(p)
assert checked==84669234
table(R/'edge_per_mesh_trace.csv',flat);table(R/'edge_group_trace.csv',group_trace);table(R/'actual_per_mesh.csv',faceflat);table(R/'actual_trend.csv',trend);write(R/'retention.json',retention)
comparison=[]
for i,u in enumerate(uids):
    a,z=trace[0]['meshes'][i],trace[-1]['meshes'][i]
    comparison.append(dict(uid=u,vertices=sizes[u],initial_failed=u in remaining,initial_fp=a['fp'],final_fp=z['fp'],initial_fn=a['fn'],final_fn=z['fn'],initial_strict=a['perfect'],final_strict=z['perfect'],initial_edge_soft4=a['edge_soft4'],final_edge_soft4=z['edge_soft4']))
table(R/'edge_per_mesh_comparison.csv',comparison)
audit=dict(new_updates=500,cumulative_tail_updates=1500,cumulative_joint_updates=1000,each_mesh_new_training_participations=500,edge_eval_records=50100,actual_eval_records=600,checkpoint_start_weights_moments_steps_rng_exact=True,original_three_adam_groups_final1500=True,new_decoder14_adam_final500=True,new_decoder14_initial_state_empty=True,persisted_full_model_only_allowed18_tensors_can_change=True,all_three_adam_groups_restored_together=True,final_saved_edge_logits_recounted=checked,remaining29_progress=dict(fp_down=sum(r['final_fp']<r['initial_fp'] for r in comparison if r['initial_failed']),fp_equal=sum(r['final_fp']==r['initial_fp'] for r in comparison if r['initial_failed']),fp_up=sum(r['final_fp']>r['initial_fp'] for r in comparison if r['initial_failed']),strict_final=sum(r['final_strict'] for r in comparison if r['initial_failed'])),clip_updates=sum(u['clip_coefficient']<1 for u in updates))
write(R/'independent_audit.json',audit)

# Parent evidence is prior continuous history, not a fresh paired Control.
manifest=read(R/'source_manifest.json')
for uid,h in manifest['augmented_pool_sha256'].items():assert sha(R/'augmented_pools'/f'{uid}_pool.npz')==h
assert sha(cfg['preserve_joint74'])==cfg['preserve_joint74_sha256']
assert sha(cfg['preserve_joint72'])==cfg['preserve_joint72_sha256']
parent_dir=R/'parent_evidence';parent_dir.mkdir(exist_ok=True)
for n in ['config.json','restore_verification.json','run/actual_evaluations.jsonl','run/verification.json','run/complete.json']:
    target=parent_dir/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(cfg['parent'])/n,target)
shutil.copyfile(Path(cfg['preserve_joint74']).parent/'final_real_network.json',R/'preserved_joint74_reference.json')
previous=[json.loads(s) for s in (parent_dir/'run/actual_evaluations.jsonl').read_text().splitlines()]
for x,y in zip(previous[-1]['meshes'],actual[0]['meshes']):
    assert {k:v for k,v in x.items() if k!='face_seconds'}=={k:v for k,v in y.items() if k!='face_seconds'}
late=[{m['uid'] for m in e['meshes'] if m['joint_perfect']} for e in actual if e['step'] in [300,400,500]]
progress=[];group_actual=[]
for uid in uids:
    before=next(m for m in actual[0]['meshes'] if m['uid']==uid);after=next(m for m in actual[-1]['meshes'] if m['uid']==uid)
    row=dict(uid=uid,vertices=sizes[uid],initial_failed=uid in remaining)
    for prefix,m in [('initial',before),('final',after)]:
        for k in ['edge_perfect','face_perfect','joint_perfect','missing_gt_face_candidates']:row[prefix+'_'+k]=m[k]
        for kind in ['edge','face']:
            for k in ['fp','fn']:row[prefix+'_'+kind+'_'+k]=m[kind][k]
    row['first_edge_strict_new_step']=next((e['step'] for e in trace if next(m for m in e['meshes'] if m['uid']==uid)['perfect']),None)
    row['first_joint_strict_evaluation']=next((e['step'] for e in actual if next(m for m in e['meshes'] if m['uid']==uid)['joint_perfect']),None)
    progress.append(row)
for e in actual:
    for name,ids in groups.items():
        ms=[m for m in e['meshes'] if m['uid'] in ids]
        row=dict(new_step=e['step'],cumulative_joint_step=e['step']+500,group=name,meshes=len(ms))
        for k in ['edge_perfect','joint_perfect','missing_gt_face_candidates']:row[k]=sum(m[k] for m in ms)
        for kind in ['edge','face']:
            for k in ['fp','fn']:row[kind+'_'+k]=sum(m[kind][k] for m in ms)
        group_actual.append(row)
table(R/'per_mesh_progress.csv',progress);table(R/'actual_group_trend.csv',group_actual)
table(R/'initial_failed29_progress.csv',[r for r in progress if r['initial_failed']])
continuous=[]
for segment,seq,offset in [('previous_joint500',previous,0),('current_continue500',actual[1:],500)]:
    for e in seq:continuous.append(dict(segment=segment,cumulative_joint_updates=e['step']+offset,**{k:e[k] for k in ['edge_perfect','joint_perfect','edge_fp','edge_fn','face_fp','face_fn']}))
table(R/'continuous_joint_history.csv',continuous)
changes=[]
for start in [0,300,400]:
    a=next(e for e in actual if e['step']==start);b=actual[-1]
    row=dict(from_new_step=start,to_new_step=500)
    for k in ['edge_fp','edge_fn','face_fp','face_fn','missing_gt_face_candidates']:
        row[k+'_before']=a[k];row[k+'_after']=b[k];row[k+'_delta']=b[k]-a[k];row[k+'_decrease_pct']=100*(a[k]-b[k])/a[k] if a[k] else None
    row.update(joint_before=a['joint_perfect'],joint_after=b['joint_perfect']);changes.append(row)
table(R/'late_changes.csv',changes)
audit.update(all_fixed_pools_unchanged=True,joint74_baseline_unchanged=True,parent_endpoint_baseline_exact=True,late_300_400_500_joint_intersection=sorted(set.intersection(*late)),initial_failed29_joint_final=sum(r['final_joint_perfect'] for r in progress if r['initial_failed']),all_checkpoints_joint_intersection=sorted(set.intersection(*[{m['uid'] for m in e['meshes'] if m['joint_perfect']} for e in actual])))
write(R/'independent_audit.json',audit)
# Paired control has the same source, old Adam/RNG, all100 order and candidates.
control_dir=R/'control_evidence';control_dir.mkdir(exist_ok=True)
for n in ['config.json','restore_verification.json','run/actual_evaluations.jsonl','run/edge_trace.jsonl','run/verification.json','run/complete.json']:
    dst=control_dir/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(cfg['paired_control'])/n,dst)
control_cfg=read(control_dir/'config.json');assert control_cfg['parent_checkpoint_sha256']==cfg['parent_checkpoint_sha256'] and control_cfg['source_sha256']==cfg['source_sha256']
control=[json.loads(s) for s in (control_dir/'run/actual_evaluations.jsonl').read_text().splitlines()]
control_trace=[json.loads(s) for s in (control_dir/'run/edge_trace.jsonl').read_text().splitlines()]
assert [e['step'] for e in control]==cfg['checkpoint_steps']
for e in control_trace:assert [m['uid'] for m in e['meshes']]==uids
for a,b in zip(control[0]['meshes'],actual[0]['meshes']):assert {k:v for k,v in a.items() if k!='face_seconds'}=={k:v for k,v in b.items() if k!='face_seconds'}
control_cp=T.load(Path(cfg['paired_control'])/'run/checkpoint-new0000-tail1000.pt',map_location='cpu',mmap=True,weights_only=False)
new_cp=T.load(run/'checkpoint-new0000-tail1000.pt',map_location='cpu',mmap=True,weights_only=False)
same(control_cp['tail'],{k:v for k,v in new_cp['tail'].items() if not k.startswith('penultimate.')});same(control_cp['optimizer']['state'],new_cp['optimizer']['state']);same(control_cp['optimizer']['param_groups'],new_cp['optimizer']['param_groups'][:3]);same(control_cp['rng'],new_cp['rng']);same(control_cp['cuda_rng'],new_cp['cuda_rng']);del control_cp,new_cp
paired=[];paired_mesh=[];paired_groups=[]
for a,b in zip(control,actual):
    row=dict(new_step=b['step'])
    for k in ['edge_perfect','joint_perfect','edge_fp','edge_fn','face_fp','face_fn','missing_gt_face_candidates']:
        row['control_'+k]=a[k];row['last2_'+k]=b[k];row['last2_minus_control_'+k]=b[k]-a[k]
    paired.append(row)
    for x,y in zip(a['meshes'],b['meshes']):
        row=dict(new_step=b['step'],uid=y['uid'],vertices=y['vertices'],initial_failed=y['uid'] in remaining)
        for name,m in [('control',x),('last2',y)]:
            for k in ['edge_perfect','joint_perfect','missing_gt_face_candidates']:row[name+'_'+k]=m[k]
            for kind in ['edge','face']:
                for k in ['fp','fn']:row[name+'_'+kind+'_'+k]=m[kind][k]
        paired_mesh.append(row)
    for name,ids in groups.items():
        row=dict(new_step=b['step'],group=name,meshes=len(ids))
        for label,e in [('control',a),('last2',b)]:
            ms=[m for m in e['meshes'] if m['uid'] in ids]
            for k in ['edge_perfect','joint_perfect']:row[label+'_'+k]=sum(m[k] for m in ms)
            for kind in ['edge','face']:
                for k in ['fp','fn']:row[label+'_'+kind+'_'+k]=sum(m[kind][k] for m in ms)
        paired_groups.append(row)
table(R/'paired_comparison.csv',paired);table(R/'paired_per_mesh.csv',paired_mesh);table(R/'paired_groups.csv',paired_groups)
audit.update(paired_same_source_old_adam_rng_exact=True,paired_all100_order_exact=True,paired_step0_all_actual_rows_exact=True,joint72_baseline_unchanged=True)
recovery=read(R/'resume_replay_verification.json')
assert recovery['replayed_updates']==list(range(101,180)) and recovery['all_forward_rows_gradients_clipping_and_actual_updates_exact']
interrupted_updates=[json.loads(t) for t in (R/'interrupted_attempt0/run/updates.jsonl').read_text().splitlines()]
interrupted_trace=[json.loads(t) for t in (R/'interrupted_attempt0/run/edge_trace.jsonl').read_text().splitlines()]
assert interrupted_updates==updates[:179] and interrupted_trace==trace[:179]
assert complete['total_optimizer_executions_including_lost_attempt']==579
same_interruption=dict(replay_scope='Exact equality of logged forward metrics, group gradient norms, clipping and parameter-displacement norms; per-step full gradient vectors were not archived.',checkpoint_new100_restored=True,replayed_updates101_to179_exact=True,unique_trajectory_updates=500,extra_replayed_executions=79,total_optimizer_executions_including_lost_attempt=579)
audit.update(interruption_recovery=same_interruption)
write(R/'independent_audit.json',audit)
lines=['# 最后两块Decoder联合训练：500次全100条更新对照','',
'新诊断已完成有效轨迹500次更新并按预算停止。共同起点为Joint step500 / 原三组Adam step1000；复用已完成的单末块Control，未重跑或覆盖Control。',
'唯一训练范围变化：原末块15、最终LayerNorm、Edge/Face head之外，再开放原Decoder第14块。所有权重来自同一父模型；原三组Adam逐项恢复，新第14块单独空状态起步。末尾原三组Adam为1500、新第14块为500。预算不能与Control相加为同一模型训练量。',
'中途服务器实例重启，首轮日志记录179次更新，最近完整checkpoint在100步。恢复100步全部状态后重放101—179，日志中的前向指标、梯度范数、clip系数和实际位移范数与中断记录逐项完全相同；然后续到500。有效轨迹500步，另有79次恢复重放，实际累计执行579次optimizer step。未增加终点训练预算。中断现场保存在interrupted_attempt0，恢复入口为resume_run.py。',
'每次更新完整累积原100条，目标mean100(Edge fully-diff Soft4 + Face fully-diff Soft4)。两个block及最终LayerNorm LR=1e-5，两个head各1e-4；mu、KL=0、两轮固定Face pool、32维评分、threshold=0、math00、clip=1、wd=0均不变。',
'缓存重新导出于第14块输入之前，第14/15块均留在可微路径。起点100条真实前向、loss与所有开放参数梯度逐位复现缓存路径；测量耗时/显存后开始更新。末尾安装相同权重回真实完整网络复核100条，实际Face从各检查点预测Edge图重新枚举。','',
'| 新增更新 | Control Edge/联合 | 两块 Edge/联合 | Control Edge FP/FN | 两块Edge FP/FN | Control实际Face FP/FN | 两块实际Face FP/FN |','|---:|---|---|---|---|---|---|']
for a,b in zip(control,actual):lines.append(f"| {b['step']} | {a['edge_perfect']}/{a['joint_perfect']} | {b['edge_perfect']}/{b['joint_perfect']} | {a['edge_fp']}/{a['edge_fn']} | {b['edge_fp']}/{b['edge_fn']} | {a['face_fp']}/{a['face_fn']} | {b['face_fp']}/{b['face_fn']} |")
q=retention[-1]['joint_perfect'];end=actual[-1]
lines += ['',f"两块末尾联合{end['joint_perfect']}/100，原71条保留{len(q['retained'])}、丢失{len(q['lost'])}、新增{len(q['gained'])}。后期300/400/500三个检查点共同成功{len(audit['late_300_400_500_joint_intersection'])}条。",'新增UID：'+(', '.join(q['gained']) or '无')+'。丢失UID：'+(', '.join(q['lost']) or '无')+'。','',
'| 预先固定组 | 条数 | Control末尾Edge FP/FN | 两块末尾Edge FP/FN | Control末尾Face FP/FN | 两块末尾Face FP/FN | Control/两块联合成功 |','|---|---:|---|---|---|---|---|']
for g in paired_groups:
    if g['new_step']==500:lines.append(f"| {g['group']} | {g['meshes']} | {g['control_edge_fp']}/{g['control_edge_fn']} | {g['last2_edge_fp']}/{g['last2_edge_fn']} | {g['control_face_fp']}/{g['control_face_fn']} | {g['last2_face_fp']}/{g['last2_face_fn']} | {g['control_joint_perfect']}/{g['last2_joint_perfect']} |")
lines += ['',
'按同预算实际FP/FN、成功UID保持及困难大mesh清错判读，不能只按loss。开放第14块同时引入其新Adam状态、改变总梯度与clip；这是训练范围策略对照，不是纯容量证明或梯度冲突证明。旧72/74条模型与共同父状态完整保留。','',
'材料索引：','',
'- run/updates.jsonl：500条完整更新，四组LR、五模块位移/梯度、全局clip；edge_trace.jsonl：501×100条Edge计数与两项训练loss。',
'- run/actual-new*.json、actual_evaluations.jsonl：600条实际Edge/Face计数、margin、候选覆盖与pool内诊断；final_real_network.json：末尾真实网络复核。',
'- paired_comparison.csv、paired_per_mesh.csv、paired_groups.csv：对齐Control的检查点、逐UID、预先固定组别；retention.json保存原71条的保留/丢失/新增。',
'- restore_verification.json、cache_gradient_verification.json、freeze_contract.json、benchmark.json、run/verification.json、independent_audit.json：缓存位置、梯度、成本、Adam状态、冻结边界、84,669,234条末尾Edge logit计数复算。',
'- run.py、prior_core.py、joint_core.py、head_core.py、face_core.py、runtime.py、evaluate.py、effective_code、runtime_dependencies：实际执行公式与数值路径。prior_core仅score/scoring_module/metrics/summary/norm被调用，历史训练入口未使用；run.py.diff/prior_core.py.diff明确本轮改变。',
'- Review含完整日志、评价、源码和初末尾开放模块checkpoint/Adam/RNG；FullEvidence另含第14块输入缓存、末尾hidden/embedding/全部Edge logits、固定Face pools和中间checkpoint。',
'- 巨大全模型不放ZIP，路径/SHA见EXCLUDED_FILES.json；Face全候选逐项logits没有另存，保留完整实际计数与评分表示。Face固定pool loss不能当作实际候选全集loss。','',
'共同父完整模型：`'+cfg['source_checkpoint']+'`，SHA `'+cfg['source_sha256']+'`。',
'共同父Adam：`'+cfg['parent_checkpoint']+'`，SHA `'+cfg['parent_checkpoint_sha256']+'`。',
'新模型：`'+complete['inference_copy']+'`，SHA `'+complete['inference_sha256']+'`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copytree(Path(cfg['parent'])/'runtime_dependencies',R/'runtime_dependencies',dirs_exist_ok=True)
shutil.copyfile(Path(cfg['parent'])/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(15,4))
for name,seq in [('Control: block15',control),('New: blocks14+15',actual)]:
    x=[e['step'] for e in seq]
    axs[0].plot(x,[e['edge_perfect'] for e in seq],'--',label=name+' Edge')
    axs[0].plot(x,[e['joint_perfect'] for e in seq],'o-',label=name+' Joint')
    axs[1].plot(x,[e['edge_fp'] for e in seq],'o-',label=name)
    axs[2].plot(x,[e['face_fp'] for e in seq],'o-',label=name+' FP')
    axs[2].plot(x,[e['face_fn'] for e in seq],'s--',label=name+' FN')
axs[0].set_ylabel('Strict meshes / 100');axs[1].set_ylabel('Edge FP');axs[2].set_ylabel('Actual Face errors');axs[2].set_yscale('log')
for ax in axs:ax.set_xlabel('New full100 joint updates');ax.legend(fontsize=6);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'trend.png',dpi=160);plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.log','.txt','.diff'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json','finalize.log']]
for folder in ['run','cache','effective_code','runtime_dependencies','source_archive','review_runtime','parent_evidence','control_evidence','interrupted_attempt0','resume_import_attempt0','augmented_pools']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='model-last2-joint1000-tail1500-block14new500-inference.pt')
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=base if kind=='FullEvidence' else [p for p in base if p.suffix!='.npz' and not (p.name.startswith('checkpoint-new') and p.suffix=='.pt' and p.name not in ['checkpoint-new0000-tail1000.pt','checkpoint-new0500-tail1500.pt'])]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(parent_joint_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),parent_joint_adam=dict(path=cfg['parent_checkpoint'],sha256=cfg['parent_checkpoint_sha256']),preserved_joint72_model=dict(path=cfg['preserve_joint72'],sha256=cfg['preserve_joint72_sha256']),preserved_joint74_model=dict(path=cfg['preserve_joint74'],sha256=cfg['preserve_joint74_sha256']),derived_model=dict(path=complete['inference_copy'],sha256=complete['inference_sha256']),actual_face_logits='Not separately exported; full hard counts and scoring embeddings are retained.')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    p=R/f'Nexus_DecoderLast2_Joint500_{kind}_20260920.zip'
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in files:z.write(f,str(f.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(p),bytes=p.stat().st_size,sha256=sha(p),files_verified=len(hashes)));write(R/'package_verification.json',metadata)
    print('PACKAGED',json.dumps(metadata[-1]),flush=True)
