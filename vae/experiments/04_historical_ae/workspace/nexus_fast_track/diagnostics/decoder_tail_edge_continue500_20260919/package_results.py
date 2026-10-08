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
    assert u['lrs']=={'decoder_tail':1e-5,'edge_head':1e-4} and all(x['delta_norm']>0 for x in u['actual_updates'].values())
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
same(parent['tail'],first['tail']);same(parent['optimizer'],first['optimizer']);same(parent['rng'],first['rng']);same(parent['cuda_rng'],first['cuda_rng'])
last=T.load(run/'checkpoint-new0500-tail1000.pt',map_location='cpu',mmap=True,weights_only=False)
assert all(float(s['step'])==1000 for s in last['optimizer']['state'].values())
source=T.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
derived=T.load(complete['inference_copy'],map_location='cpu',mmap=True,weights_only=False)
prefixes={'block.':'autoencoder.decoder_blocks.15.','final_norm.':'autoencoder.decoder_output_norm.','head.':'autoencoder.edge_embedding.'}
mapping={k:next(v+k[len(p):] for p,v in prefixes.items() if k.startswith(p)) for k in last['tail']}
assert len(mapping)==10 and source['model'].keys()==derived['model'].keys()
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
audit=dict(new_updates=500,cumulative_tail_updates=1000,each_mesh_new_training_participations=500,edge_eval_records=50100,actual_eval_records=700,checkpoint_start_weights_moments_steps_rng_exact=True,all_adam_final_step1000=True,persisted_full_model_only_allowed10_tensors_can_change=True,persisted_face_head_unchanged=True,final_saved_edge_logits_recounted=checked,remaining29_progress=dict(fp_down=sum(r['final_fp']<r['initial_fp'] for r in comparison if r['initial_failed']),fp_equal=sum(r['final_fp']==r['initial_fp'] for r in comparison if r['initial_failed']),fp_up=sum(r['final_fp']>r['initial_fp'] for r in comparison if r['initial_failed']),strict_final=sum(r['final_strict'] for r in comparison if r['initial_failed'])),clip_updates=sum(u['clip_coefficient']<1 for u in updates))
write(R/'independent_audit.json',audit)
lines=['# Decoder末端Edge续训：新增500次更新','',
    '71条联合成功的模型完整保留；独立分支沿原末端Edge step500的权重和Adam继续到tail step1000。每次100条共同参与，mean100(fully-diff Edge Soft4)，全84,669,234 pair；仅Decoder末块＋最终LayerNorm＋Edge head更新。LR=1e-5/1e-4，clip=1，wd=0，无warmup，无Face loss、KL或sampling。',
    'Face head使用恢复后的Face-step500权重并冻结，没有加载其新Adam。初始尾部权重在Edge-step500与71条完整成功模型之间逐位相同；初始Adam一二阶矩、step及CPU/CUDA RNG与原Edge-step500完全一致。',
    '所有训练步骤仍使用缓存的末块输入，不是最终hidden。起点与末尾均从真实输入重跑100条，核验缓存结果；实际Face在每个检查点从该点预测Edge图重新枚举，没有沿用旧候选图。',
    '', '| 新增更新 | 末端累计更新 | Edge严格成功 | 联合严格成功 | Edge FP/FN | 实际Face FP/FN |','|---:|---:|---:|---:|---|---|']
for e in actual:lines.append(f"| {e['step']} | {e['step']+500} | {e['edge_perfect']} | {e['joint_perfect']} | {e['edge_fp']}/{e['edge_fn']} | {e['face_fp']}/{e['face_fn']} |")
lines+=['','## 固定组的Edge进展','','| 组 | 条数 | 起点FP/FN | 末尾FP/FN | 起点/末尾严格成功 |','|---|---:|---|---|---|']
for name in groups:
    a=next(x for x in group_trace if x['new_step']==0 and x['group']==name);z=next(x for x in group_trace if x['new_step']==500 and x['group']==name)
    lines.append(f"| {name} | {a['meshes']} | {a['fp']}/{a['fn']} | {z['fp']}/{z['fn']} | {a['strict']}/{z['strict']} |")
lines+=['',f"原29条失败样本中，{audit['remaining29_progress']['fp_down']}条FP减少、{audit['remaining29_progress']['fp_equal']}条不变、{audit['remaining29_progress']['fp_up']}条增加；其中{audit['remaining29_progress']['strict_final']}条末尾Edge严格成功。",'']
for key,label in [('edge_perfect','原71条Edge成功'),('joint_perfect','原71条联合成功')]:
    q=retention[-1][key];lines.append(f"{label}：末尾保留{len(q['retained'])}、丢失{len(q['lost'])}、新增{len(q['gained'])}。UID清单见retention.json。")
lines+=['','共享Decoder末端更新后，Face embedding会变化，即使Face head权重不变。Edge收益不能自动继承原联合成功标签；本轮按实际Face重新验收。有限预算内仍有错误不构成容量不可能的证明。本轮已停止，没有刷新pool、扩大解冻范围或自动延长。',
    '', '## 文件索引及核验范围','',
    '- run/updates.jsonl：500条更新、分组实际位移、梯度、clip和LR；run/edge_trace.jsonl：501个状态×100条全部Edge计数、loss及margin。',
    '- run/actual-new*.json、actual_evaluations.jsonl、actual_per_mesh.csv：7次完整实际Edge/Face与GT候选覆盖；Face pool仅作诊断。',
    '- edge_group_trace.csv：预先固定的原29条、>1500顶点组与原71成功组；edge_per_mesh_comparison.csv给出逐UID变化。',
    '- restore_verification.json：恢复原Edge Adam，不使用Face Adam；run/verification.json：冻结、缓存及真实网络重接检查。',
    '- cache/manifest.json保留最初epoch900导出的旧baseline字段，用于缓存来源核验；本轮真正step0见run/actual-new0000.json和edge_trace首条，不能把旧缓存baseline当作本轮起点。',
    '- independent_audit.json：读取持久化起点/末尾checkpoint独立验证Adam、RNG和10个可变张量边界，并重算全部84,669,234个末尾Edge logits。',
    '- run/checkpoint-new*.pt：尾部三模块及Adam/RNG；末尾为new0500-tail1000，不把Face训练更新计入Edge累计。',
    '- actual候选枚举没有超时截断。最终Face候选逐条logit未全部导出；完整计数、评分表示和实际执行枚举代码保留。',
    '- FullEvidence另含缓存末块输入、原mesh、末尾hidden/Edge/Face表示和全部Edge logits、中间尾部checkpoint。Review保留日志、代码、初末尾尾部权重/Adam和核验结果。',
    '- 大型原完整模型与本轮派生完整推理模型不入ZIP；位置和SHA见EXCLUDED_FILES.json。原缓存与所有旧分支保持不变。',
    '', '保留71条基线：`'+cfg['source_checkpoint']+'`，SHA256 `'+cfg['source_sha256']+'`。',
    '本轮候选完整模型：`'+complete['inference_copy']+'`，SHA256 `'+complete['inference_sha256']+'`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
for folder in ['runtime_dependencies']:shutil.copytree(Path(cfg['parent'])/folder,R/folder,dirs_exist_ok=True)
shutil.copyfile(Path(cfg['parent'])/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4))
for group in ['all100','N_gt1500','initial_failed29']:
    seq=[r for r in group_trace if r['group']==group]
    axs[0].plot([r['new_step'] for r in seq],[r['fp'] for r in seq],label=group)
axs[0].set(ylabel='Edge FP');axs[0].legend(fontsize=8)
axs[1].plot([x['step'] for x in trace],[x['edge_perfect'] for x in trace],label='Edge')
axs[1].plot([x['step'] for x in actual],[x['joint_perfect'] for x in actual],'o-',label='Edge + actual Face');axs[1].legend();axs[1].set(ylabel='Strict meshes / 100')
for key in ['face_fp','face_fn']:axs[2].plot([x['step'] for x in actual],[x[key] for x in actual],'o-',label=key)
axs[2].set_yscale('log');axs[2].legend();axs[2].set(ylabel='Actual Face errors')
for ax in axs:ax.set_xlabel('New Edge-tail updates');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'trend.png',dpi=160);plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.log','.txt'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json']]
for folder in ['run','cache','effective_code','runtime_dependencies','source_archive','review_runtime']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='model-tail1000-face500-inference.pt')
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=base if kind=='FullEvidence' else [p for p in base if p.suffix!='.npz' and not (p.name.startswith('checkpoint-new') and p.suffix=='.pt' and p.name not in ['checkpoint-new0000-tail0500.pt','checkpoint-new0500-tail1000.pt'])]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(source_joint71_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256']),derived_model=dict(path=complete['inference_copy'],sha256=complete['inference_sha256']),face_pools='Original augmented binaries remain server-side; paths/hashes in source_manifest.json; no Face training or pool changes in this experiment.')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    p=R/f'Nexus_DecoderTail_Continue500_{kind}_20260919.zip'
    with zipfile.ZipFile(p,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in files:z.write(f,str(f.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(p),bytes=p.stat().st_size,sha256=sha(p),files_verified=len(hashes)));write(R/'package_verification.json',metadata)
    print('PACKAGED',json.dumps(metadata[-1]),flush=True)
