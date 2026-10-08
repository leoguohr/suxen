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
    assert u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4,'face_head':1e-4} and all(x['delta_norm']>0 for x in u['actual_updates'].values())
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
same(parent['tail'],first['tail']);same(parent['optimizer'],first['optimizer'])
same(parent['rng'],first['rng']);same(parent['cuda_rng'],first['cuda_rng'])
assert all(float(v['step'])==1000 for v in first['optimizer']['state'].values())
last=T.load(run/'checkpoint-new0500-tail1500.pt',map_location='cpu',mmap=True,weights_only=False)
assert all(float(v['step'])==1500 for v in last['optimizer']['state'].values())
assert last['completed_updates']==1500 and last['new_updates']==500 and last['cumulative_joint_updates']==1000
source=T.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
derived=T.load(complete['inference_copy'],map_location='cpu',mmap=True,weights_only=False)
prefixes={'block.':'autoencoder.decoder_blocks.15.','final_norm.':'autoencoder.decoder_output_norm.','head.':'autoencoder.edge_embedding.','face.':'autoencoder.face_embedding.'}
mapping={k:next(v+k[len(p):] for p,v in prefixes.items() if k.startswith(p)) for k in last['tail']}
assert len(mapping)==12 and source['model'].keys()==derived['model'].keys()
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
audit=dict(new_updates=500,cumulative_tail_updates=1500,cumulative_joint_updates=1000,each_mesh_new_training_participations=500,edge_eval_records=50100,actual_eval_records=600,checkpoint_start_weights_moments_steps_rng_exact=True,all_adam_final_step1500=True,persisted_full_model_only_allowed12_tensors_can_change=True,all_three_adam_groups_restored_together=True,final_saved_edge_logits_recounted=checked,remaining29_progress=dict(fp_down=sum(r['final_fp']<r['initial_fp'] for r in comparison if r['initial_failed']),fp_equal=sum(r['final_fp']==r['initial_fp'] for r in comparison if r['initial_failed']),fp_up=sum(r['final_fp']>r['initial_fp'] for r in comparison if r['initial_failed']),strict_final=sum(r['final_strict'] for r in comparison if r['initial_failed'])),clip_updates=sum(u['clip_coefficient']<1 for u in updates))
write(R/'independent_audit.json',audit)

# Parent evidence is prior continuous history, not a fresh paired Control.
manifest=read(R/'source_manifest.json')
for uid,h in manifest['augmented_pool_sha256'].items():assert sha(R/'augmented_pools'/f'{uid}_pool.npz')==h
assert sha(cfg['preserve_joint74'])==cfg['preserve_joint74_sha256']
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
lines=['# Joint分支新增500次全100条联合更新','',
'已完成并按预算停止。本轮是从Joint step500 / Tail与Face Adam step1000续接的新500次更新，末尾Joint累计1000次，三组Adam均到1500。不是500个epoch，也不是原主线mini-batch数。',
'同一个原100条集合，每次遍历全部100条、按mesh等权累积联合梯度后统一clip与Adam更新。目标mean100(Edge fully-diff Soft4 + Face fully-diff Soft4)，各项Soft4内部固定除4；没有额外诊断系数。',
'仅Decoder末块15、最终LayerNorm、Edge head、Face head可训练；末端LR=1e-5、两head各1e-4，三组Adam连同CPU/CUDA RNG从同一父checkpoint整体恢复。没有重置、重新warmup、刷新两轮Face pool、增加其他目标或解冻上游。math00、mu路径、KL=0、clip=1、wd=0均保持。',
'更新前全100条真实网络与缓存路径输出、联合loss/梯度逐位一致；重复完整梯度逐位一致。末尾重新加载endpoint并由真实输入复核全部100条；Face由每个检查点的预测Edge图重新枚举。','',
'| 本轮新增更新 | Joint累计 | Edge成功/100 | 联合成功/100 | Edge FP/FN | 实际Face FP/FN | 缺失GT Face候选 |',
'|---:|---:|---:|---:|---|---|---:|']
for e in actual:lines.append(f"| {e['step']} | {e['step']+500} | {e['edge_perfect']} | {e['joint_perfect']} | {e['edge_fp']}/{e['edge_fn']} | {e['face_fp']}/{e['face_fn']} | {e['missing_gt_face_candidates']} |")
q=retention[-1]['joint_perfect'];end=actual[-1]
lines += ['',f"末尾联合严格成功{end['joint_perfect']}/100；起点71条保留{len(q['retained'])}、丢失{len(q['lost'])}、新增{len(q['gained'])}。后期300/400/500三次共同成功{len(audit['late_300_400_500_joint_intersection'])}条。",f"新增成功UID：{', '.join(q['gained']) or '无'}。丢失UID：{', '.join(q['lost']) or '无'}。",'',
'| 预先固定组 | 条数 | 起点Edge FP/FN | 末尾Edge FP/FN | 起点Face FP/FN | 末尾Face FP/FN | 末尾联合成功 |','|---|---:|---|---|---|---|---:|']
for name in groups:
    a=next(r for r in group_actual if r['group']==name and r['new_step']==0);b=next(r for r in group_actual if r['group']==name and r['new_step']==500)
    lines.append(f"| {name} | {a['meshes']} | {a['edge_fp']}/{a['edge_fn']} | {b['edge_fp']}/{b['edge_fn']} | {a['face_fp']}/{a['face_fn']} | {b['face_fp']}/{b['face_fn']} | {b['joint_perfect']} |")
lines += ['',f"原失败29条：Edge FP减少{audit['remaining29_progress']['fp_down']}条、相等{audit['remaining29_progress']['fp_equal']}条、增加{audit['remaining29_progress']['fp_up']}条；其中末尾Edge严格成功{audit['remaining29_progress']['strict_final']}条、联合严格成功{audit['initial_failed29_joint_final']}条。",'',
'| 本轮区间 | Edge FP下降比例 | Face FP下降比例 | Face FN变化 | 联合成功变化 |','|---|---:|---:|---|---|']
for r in changes:
    a=r['edge_fp_decrease_pct'];b=r['face_fp_decrease_pct']
    lines.append(f"| {r['from_new_step']}→500 | {a:.4f}% | {b:.4f}% | {r['face_fn_before']}→{r['face_fn_after']} | {r['joint_before']}→{r['joint_after']} |")
lines += ['',
'结果按成功UID和实际FP/FN判读。有限预算没有突破不能证明容量不可能；loss下降也不能替代结构清错或严格成功。此次无自动追加，旧74条模型仍完整保留，不能把两个模型的成功UID拼成一份模型结果。',
'continuous_joint_history.csv提供此前500次与本次500次的连续轨迹，前后起点不同，不把前一段当成本轮新的配对Control。旧74条模型来自另一条分阶段路径，不按本轮等预算对照排名。','',
'材料索引：',
'- run/updates.jsonl：500条完整更新、三组LR、四模块实际位移、梯度与clip；edge_trace.jsonl：501×100条Edge计数、margin及Edge/Face训练pool loss。',
'- run/actual-new*.json与actual_evaluations.jsonl：6次、600条完整Edge与实际Face计数，pool内计数、候选覆盖与margin；final_real_network.json：末尾真实输入复核。',
'- actual_per_mesh.csv、per_mesh_progress.csv、initial_failed29_progress.csv、actual_group_trend.csv、late_changes.csv、retention.json：逐mesh与组别趋势、成功保留/丢失/新增和末段变化。',
'- restore_verification.json、cache_gradient_verification.json、benchmark.json、run/verification.json、independent_audit.json：状态恢复、真实/缓存梯度、冻结边界与84,669,234个末尾Edge logits的独立计数复算。',
'- runtime.py、joint_core.py、prior_core.py、head_core.py、face_core.py、evaluate.py、effective_code、runtime_dependencies：实际公式、评分与运行时替换；continuation_changes.diff记录相对父入口的续训改动。prior_core的历史训练入口本轮未调用。',
'- Review：日志、逐mesh结果、实际源码、初末尾末端/两个head权重及Adam/RNG；FullEvidence额外包含缓存输入、末尾hidden与评分embedding、全部Edge logits、中间checkpoint、固定Face pool。',
'- 巨大全模型未放ZIP；EXCLUDED_FILES.json列出服务器路径、已知SHA与排除范围。Face全候选逐项logits未单独保存，包含完整实际计数与评分表示；Face训练pool loss不等于全实际候选loss。',
'- 本轮始终是mu训练与验收，没有posterior噪声成功率结论。','',
'父完整模型：`'+cfg['source_checkpoint']+'`；SHA `'+cfg['source_sha256']+'`。',
'父Adam checkpoint：`'+cfg['parent_checkpoint']+'`；SHA `'+cfg['parent_checkpoint_sha256']+'`。',
'新完整模型：`'+complete['inference_copy']+'`；SHA `'+complete['inference_sha256']+'`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copytree(Path(cfg['parent'])/'runtime_dependencies',R/'runtime_dependencies',dirs_exist_ok=True)
shutil.copyfile(Path(cfg['parent'])/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(15,4));x=[e['step'] for e in actual]
axs[0].plot(x,[e['edge_perfect'] for e in actual],'o-',label='Edge strict');axs[0].plot(x,[e['joint_perfect'] for e in actual],'s--',label='Edge + actual Face strict');axs[0].axhline(71,color='gray',linestyle=':',label='Initial 71')
axs[1].plot(x,[e['edge_fp'] for e in actual],'o-',label='Edge FP');axs[2].plot(x,[e['face_fp'] for e in actual],'o-',label='Actual Face FP');axs[2].plot(x,[e['face_fn'] for e in actual],'s--',label='Actual Face FN');axs[2].set_yscale('log')
axs[0].set_ylabel('Strict meshes / 100');axs[1].set_ylabel('Edge FP');axs[2].set_ylabel('Actual Face errors')
for ax in axs:ax.set_xlabel('New joint updates (Adam 1000 + x)');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'trend.png',dpi=160);plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.log','.txt','.diff'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json','finalize.log']]
for folder in ['run','cache','effective_code','runtime_dependencies','source_archive','review_runtime','parent_evidence','augmented_pools']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='model-joint1000-tail1500-face1500-inference.pt')
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=base if kind=='FullEvidence' else [p for p in base if p.suffix!='.npz' and not (p.name.startswith('checkpoint-new') and p.suffix=='.pt' and p.name not in ['checkpoint-new0000-tail1000.pt','checkpoint-new0500-tail1500.pt'])]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(parent_joint_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),parent_joint_adam=dict(path=cfg['parent_checkpoint'],sha256=cfg['parent_checkpoint_sha256']),preserved_joint74_model=dict(path=cfg['preserve_joint74'],sha256=cfg['preserve_joint74_sha256']),derived_model=dict(path=complete['inference_copy'],sha256=complete['inference_sha256']),actual_face_logits='Not separately exported; full hard counts and scoring embeddings are retained.')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    p=R/f'Nexus_DecoderTail_JointContinue500_{kind}_20260920.zip'
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in files:z.write(f,str(f.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(p),bytes=p.stat().st_size,sha256=sha(p),files_verified=len(hashes)));write(R/'package_verification.json',metadata)
    print('PACKAGED',json.dumps(metadata[-1]),flush=True)
