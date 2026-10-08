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
assert complete['state']=='complete' and complete['new_updates']==500 and complete['cumulative_tail_updates']==1000 and complete['stopped_at_budget']
assert not (R/'failure.txt').exists()
trace=[json.loads(s) for s in (run/'edge_trace.jsonl').read_text().splitlines()]
updates=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()]
actual=[json.loads(s) for s in (run/'actual_evaluations.jsonl').read_text().splitlines()]
uids=[m['uid'] for m in meta['meshes']];sizes={m['uid']:m['vertices'] for m in meta['meshes']}
assert len(uids)==len(set(uids))==100 and uids==read(R/'selection.json')['uids']
assert [e['step'] for e in trace]==list(range(501)) and [u['update'] for u in updates]==list(range(1,501))
assert [e['step'] for e in actual]==cfg['checkpoint_steps']
for u in updates:
    assert u['mesh_count']==100 and u['cumulative_tail_update']==500+u['update']
    assert u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4,'face_head':1e-4} and all(x['delta_norm']>0 for x in u['actual_updates'].values())
old_edge={m['uid'] for m in trace[0]['meshes'] if m['perfect']};old_joint={m['uid'] for m in actual[0]['meshes'] if m['joint_perfect']};assert old_edge==old_joint and len(old_edge)==71
remaining=set(uids)-old_edge;assert len(remaining)==29
groups={'all100':set(uids),'initial_failed29':remaining,'N_gt1500':{u for u in uids if sizes[u]>1500},'initial_success71':old_edge}
group_trace=[];flat=[]
for e in trace:
    assert [m['uid'] for m in e['meshes']]==uids and e['cumulative_tail_updates']==e['step']+500
    assert e['total_fp']==sum(m['fp'] for m in e['meshes']) and e['total_fn']==sum(m['fn'] for m in e['meshes']) and e['edge_perfect']==sum(m['perfect'] for m in e['meshes'])
    for name,ids in groups.items():
        ms=[m for m in e['meshes'] if m['uid'] in ids]
        group_trace.append(dict(new_step=e['step'],tail_step=e['step']+500,group=name,meshes=len(ms),fp=sum(m['fp'] for m in ms),fn=sum(m['fn'] for m in ms),strict=sum(m['perfect'] for m in ms),mean_edge_soft4=sum(m['edge_soft4'] for m in ms)/len(ms)))
    for m in e['meshes']:flat.append(dict(new_step=e['step'],tail_step=e['step']+500,vertices=sizes[m['uid']],**m))
retention=[];trend=[];faceflat=[]
for e in actual:
    assert [m['uid'] for m in e['meshes']]==uids
    er={m['uid']:m for m in trace[e['step']]['meshes']}
    for m in e['meshes']:
        assert m['face']['complete'] and all(m['edge'][k]==er[m['uid']][k] for k in ['tp','fp','fn','tn'])
        assert m['face']['tp']+m['face']['fn']==m['gt_faces'] and m['face']['fn']>=m['missing_gt_face_candidates']
        rr=dict(new_step=e['step'],tail_step=e['step']+500,uid=m['uid'],vertices=m['vertices'],face_pool_soft4=m['face_pool_soft4'],**{kind+'_'+k:m[kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},**{k:m[k] for k in ['edge_perfect','face_perfect','joint_perfect','gt_face_candidates','missing_gt_face_candidates']},face_candidates=m['face']['scored_candidates'],face_fn_candidate_present=m['face']['fn']-m['missing_gt_face_candidates'])
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
first=T.load(run/'checkpoint-new0000-tail0500.pt',map_location='cpu',mmap=True,weights_only=False)
same(parent['tail'],{k:v for k,v in first['tail'].items() if not k.startswith('face.')})
same(parent['optimizer']['param_groups'],first['optimizer']['param_groups'][:2])
for k,v in parent['optimizer']['state'].items():same(v,first['optimizer']['state'][k])
same(parent['rng'],first['rng']);same(parent['cuda_rng'],first['cuda_rng'])
facecp=T.load(cfg['face_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
same(facecp['face_head'],{k.removeprefix('face.'):v for k,v in first['tail'].items() if k.startswith('face.')})
for a,b in zip(facecp['optimizer']['param_groups'][0]['params'],first['optimizer']['param_groups'][2]['params']):same(facecp['optimizer']['state'][a],first['optimizer']['state'][b])
assert first['optimizer']['param_groups'][2]['lr']==1e-4
control_start=T.load(Path(cfg['edge_only_control'])/'run/checkpoint-new0000-tail0500.pt',map_location='cpu',mmap=True,weights_only=False)
same(parent['tail'],control_start['tail']);same(parent['optimizer'],control_start['optimizer']);same(parent['rng'],control_start['rng']);same(parent['cuda_rng'],control_start['cuda_rng'])
del facecp,control_start
last=T.load(run/'checkpoint-new0500-tail1000.pt',map_location='cpu',mmap=True,weights_only=False)
assert all(float(s['step'])==1000 for s in last['optimizer']['state'].values())
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
audit=dict(new_updates=500,cumulative_tail_updates=1000,each_mesh_new_training_participations=500,edge_eval_records=50100,actual_eval_records=700,checkpoint_start_weights_moments_steps_rng_exact=True,all_adam_final_step1000=True,persisted_full_model_only_allowed12_tensors_can_change=True,face_adam_restored_exact=True,final_saved_edge_logits_recounted=checked,remaining29_progress=dict(fp_down=sum(r['final_fp']<r['initial_fp'] for r in comparison if r['initial_failed']),fp_equal=sum(r['final_fp']==r['initial_fp'] for r in comparison if r['initial_failed']),fp_up=sum(r['final_fp']>r['initial_fp'] for r in comparison if r['initial_failed']),strict_final=sum(r['final_strict'] for r in comparison if r['initial_failed'])),clip_updates=sum(u['clip_coefficient']<1 for u in updates))
write(R/'independent_audit.json',audit)

# Lock the only intended data change to zero, and preserve both successful baseline files.
manifest=read(R/'source_manifest.json')
for uid,h in manifest['augmented_pool_sha256'].items():assert sha(R/'augmented_pools'/f'{uid}_pool.npz')==h
assert sha(cfg['face_checkpoint'])==cfg['face_checkpoint_sha256']
assert sha(cfg['preserve_joint74'])==cfg['preserve_joint74_sha256']
control_dir=R/'control_evidence';control_dir.mkdir(exist_ok=True)
shutil.copyfile(Path(cfg['preserve_joint74']).parent/'final_real_network.json',R/'preserved_joint74_reference.json')
for n in ['config.json','restore_verification.json','run/actual_evaluations.jsonl','run/edge_trace.jsonl','run/verification.json','run/complete.json']:
    target=control_dir/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(cfg['edge_only_control'])/n,target)
control=[json.loads(x) for x in (control_dir/'run/actual_evaluations.jsonl').read_text().splitlines()]
assert [x['step'] for x in control]==cfg['checkpoint_steps']
control_edges=[json.loads(x) for x in (control_dir/'run/edge_trace.jsonl').read_text().splitlines()]
assert [x['step'] for x in control_edges]==list(range(501))
for a,b in zip(actual[0]['meshes'],control[0]['meshes']):
    assert a['uid']==b['uid'] and a['edge']==b['edge']
    assert all(a['face'][k]==b['face'][k] for k in ['tp','fp','fn','tn','scored_candidates'])
    assert a['face_pool_soft4']==b['face_pool_soft4']
paired=[];pairmesh=[];groupcomp=[]
for a,b in zip(control,actual):
    row={'new_step':b['step']}
    for k in ['edge_perfect','joint_perfect','edge_fp','edge_fn','face_fp','face_fn','missing_gt_face_candidates']:
        row['control_'+k]=a[k];row['joint_'+k]=b[k];row['joint_minus_control_'+k]=b[k]-a[k]
    row.update(control_edge_soft4=control_edges[b['step']]['objective'],joint_edge_soft4=trace[b['step']]['edge_objective'],control_face_soft4=sum(m['face_pool_soft4'] for m in a['meshes'])/100,joint_face_soft4=sum(m['face_pool_soft4'] for m in b['meshes'])/100)
    paired.append(row)
    for x,y in zip(a['meshes'],b['meshes']):
        assert x['uid']==y['uid']
        z=dict(new_step=b['step'],uid=y['uid'],vertices=y['vertices'])
        for name,m in [('control',x),('joint',y)]:
            z[name+'_strict']=m['joint_perfect']
            for kind in ['edge','face']:
                for k in ['tp','fp','fn']:z[name+'_'+kind+'_'+k]=m[kind][k]
            z[name+'_face_fn_missing_candidate']=m['missing_gt_face_candidates']
            z[name+'_face_fn_present_candidate']=m['face']['fn']-m['missing_gt_face_candidates']
            z[name+'_face_pool_soft4']=m['face_pool_soft4']
        pairmesh.append(z)
    for group,ids in groups.items():
        z=dict(new_step=b['step'],group=group,meshes=len(ids))
        for name,rows in [('control',a['meshes']),('joint',b['meshes'])]:
            ms=[m for m in rows if m['uid'] in ids]
            for k in ['edge_perfect','joint_perfect','missing_gt_face_candidates']:z[name+'_'+k]=sum(m[k] for m in ms)
            for kind in ['edge','face']:
                for k in ['fp','fn']:z[name+'_'+kind+'_'+k]=sum(m[kind][k] for m in ms)
        groupcomp.append(z)
table(R/'paired_comparison.csv',paired);table(R/'paired_per_mesh.csv',pairmesh);table(R/'paired_groups.csv',groupcomp)
late=[{r['uid'] for r in e['meshes'] if r['joint_perfect']} for e in actual if e['step'] in [300,400,500]]
audit.update(same_start_actual_counts_and_pool_loss=True,control_tail_adam_and_rng_identical=True,all_fixed_pools_unchanged=True,joint74_baseline_unchanged=True,late_300_400_500_joint_intersection=sorted(set.intersection(*late)))
write(R/'independent_audit.json',audit)
lines=['# 100条末端联合训练：与已有Edge-only配对对照','',
'已完成500次新增联合更新并按预算停止。从旧71条成功完整模型分叉；只训练Decoder最后一个block、最终LayerNorm、Edge head和Face head。Encoder、mu、Decoder前15块和logvar全部冻结。',
'原100条每次完整累积后更新一次；目标为mean100(Edge Soft4 + Face Soft4)，两项内部固定除4、fully-differentiable，不额外加权。原两轮Face pool不变、mu路径、KL=0、clip=1、wd=0。LR分别为末端1e-5、Edge1e-4、Face1e-4。',
'末端与Edge Adam恢复到与Control相同的step500；Face新增组恢复共同71条起点对应Face-recovery step500 Adam。全部三组末尾Adam step1000，不重新warmup。Face解冻及其既有状态是本次策略的一部分，不是单独的梯度冲突证明。',
'使用同一缓存末块输入，逐mesh顺序与Control一致；起点100条真实网络与缓存联合loss/梯度逐位核对，训练前重复完整梯度逐位一致。末尾回到真实输入完成100条复核。实际Face在每个检查点重新从预测Edge图枚举，无超时截断。',
'', '| 新增更新 | Control Edge/联合成功 | Joint Edge/联合成功 | Control Edge FP/FN | Joint Edge FP/FN | Control Face FP/FN | Joint Face FP/FN |', '|---:|---|---|---|---|---|---|']
for a,b in zip(control,actual):lines.append(f"| {b['step']} | {a['edge_perfect']}/{a['joint_perfect']} | {b['edge_perfect']}/{b['joint_perfect']} | {a['edge_fp']}/{a['edge_fn']} | {b['edge_fp']}/{b['edge_fn']} | {a['face_fp']}/{a['face_fn']} | {b['face_fp']}/{b['face_fn']} |")
q=retention[-1]['joint_perfect'];a=control[-1];b=actual[-1]
lines += ['',f"Joint末尾{b['joint_perfect']}/100联合严格成功、{b['edge_perfect']}/100 Edge严格成功。原71条联合成功保留{len(q['retained'])}、丢失{len(q['lost'])}、新增{len(q['gained'])}。后期300/400/500共同成功集合为{len(audit['late_300_400_500_joint_intersection'])}条。",'',
'## 预先固定的大mesh与失败样本组','',
'| 组 | 条数 | Control末尾Edge FP/FN | Joint末尾Edge FP/FN | Control末尾联合成功 | Joint末尾联合成功 |','|---|---:|---|---|---:|---:|']
for g in groupcomp:
    if g['new_step']==500:lines.append(f"| {g['group']} | {g['meshes']} | {g['control_edge_fp']}/{g['control_edge_fn']} | {g['joint_edge_fp']}/{g['joint_edge_fn']} | {g['control_joint_perfect']} | {g['joint_joint_perfect']} |")
lines += ['', '## 判读与材料范围','',
'比较相同步数的Edge/实际Face FP/FN与UID保持。Control只有Edge目标，不能以新联合total loss对Control Edge loss直接排名。旧74条模型还包含另加500次Face恢复，其预算多一段，单独保留且不作为本轮等预算Control。',
'有限预算的成功或失败不单独证明梯度冲突或容量不可能。本轮没有自动追加、刷新pool、改变架构或扩大解冻范围。','',
'- run/updates.jsonl：500条更新、三组LR、四模块实际位移、分组梯度和clip；edge_trace.jsonl包含501×100条Edge/Face loss与Edge计数。',
'- run/actual-new*.json、actual_evaluations.jsonl：700条完整实际Edge/Face、候选覆盖、training pool计数和margin。',
'- paired_comparison.csv、paired_per_mesh.csv、paired_groups.csv：逐检查点、逐UID、预先固定组的同预算对照。retention.json保存成功UID保留/丢失/新增。',
'- restore_verification.json、cache_gradient_verification.json、benchmark.json、run/verification.json与independent_audit.json：源/Adam/RNG、真实与缓存梯度、冻结范围、末尾重载验证及84,669,234个Edge logits独立计数复算。',
'- runtime.py、prior_core.py、face_core.py、effective_code与runtime_dependencies：实际执行公式与动态替换依赖；protocol_changes.diff对应旧Edge-only执行入口的改动。prior_core历史train/export入口不在本轮调用。',
'- Review含日志、逐mesh评价、实际代码、初末尾末端/双head权重与Adam/RNG；FullEvidence另含缓存末块输入、末尾hidden与两个embedding、全部Edge logits、中间checkpoint与固定Face pool。',
'- 最终实际Face逐候选logits未单独导出；已保留完整计数、评分表示、固定pool和实际枚举实现。训练pool loss不等于实际Face全候选loss。',
'- 巨大全网络权重不放ZIP；EXCLUDED_FILES.json提供服务器路径与SHA。旧71/74成功模型与Control均完整保留。',
'', '源71条模型：`'+cfg['source_checkpoint']+'`，SHA `'+cfg['source_sha256']+'`。',
'新联合模型：`'+complete['inference_copy']+'`，SHA `'+complete['inference_sha256']+'`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
shutil.copytree(Path(cfg['parent'])/'runtime_dependencies',R/'runtime_dependencies',dirs_exist_ok=True)
shutil.copyfile(Path(cfg['parent'])/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(15,4))
for name,seq in [('Edge-only Control',control),('Tail Joint',actual)]:
    x=[e['step'] for e in seq]
    axs[0].plot(x,[e['edge_perfect'] for e in seq],'--',label=name+' Edge')
    axs[0].plot(x,[e['joint_perfect'] for e in seq],'o-',label=name+' Joint')
    axs[1].plot(x,[e['edge_fp'] for e in seq],'o-',label=name)
    axs[2].plot(x,[e['face_fp'] for e in seq],'o-',label=name+' FP')
    axs[2].plot(x,[e['face_fn'] for e in seq],'s--',label=name+' FN')
axs[0].set_ylabel('Strict meshes / 100');axs[1].set_ylabel('Edge FP');axs[2].set_ylabel('Actual Face errors');axs[2].set_yscale('log')
for ax in axs:ax.set_xlabel('New updates');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'trend.png',dpi=160);plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.log','.txt','.diff'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json']]
for folder in ['run','cache','effective_code','runtime_dependencies','source_archive','review_runtime','control_evidence','augmented_pools']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='model-joint500-tail1000-face1000-inference.pt')
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=base if kind=='FullEvidence' else [p for p in base if p.suffix!='.npz' and not (p.name.startswith('checkpoint-new') and p.suffix=='.pt' and p.name not in ['checkpoint-new0000-tail0500.pt','checkpoint-new0500-tail1000.pt'])]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(source_joint71_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),preserved_joint74_model=dict(path=cfg['preserve_joint74'],sha256=cfg['preserve_joint74_sha256']),derived_model=dict(path=complete['inference_copy'],sha256=complete['inference_sha256']),actual_face_logits='Not separately exported; complete hard counts and scoring embeddings are retained.')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    p=R/f'Nexus_DecoderTail_Joint500_{kind}_20260919.zip'
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in files:z.write(f,str(f.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(p),bytes=p.stat().st_size,sha256=sha(p),files_verified=len(hashes)));write(R/'package_verification.json',metadata)
    print('PACKAGED',json.dumps(metadata[-1]),flush=True)
