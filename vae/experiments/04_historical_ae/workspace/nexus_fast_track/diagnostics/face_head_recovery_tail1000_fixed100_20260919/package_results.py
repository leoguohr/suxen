"""Read-only result audit and review/full-evidence archives after the fixed500 budget."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def table(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sequential_mean(values):
    s=0.
    for v in values:s+=v
    return s/len(values)

cfg=read(R/'config.json');complete=read(R/'run/complete.json');meta=read(R/'cache/manifest.json')
assert complete['state']=='complete' and complete['updates']==500 and complete['budget_stopped']
assert not (R/'failure.txt').exists()
evaluations=[json.loads(x) for x in (R/'run/evaluations.jsonl').read_text().splitlines()]
updates=[json.loads(x) for x in (R/'run/updates.jsonl').read_text().splitlines()]
assert [e['step'] for e in evaluations]==cfg['checks']
assert [u['update'] for u in updates]==list(range(1,501))
assert all(u['mesh_count']==100 and len(u['per_mesh_losses'])==100 and u['lr']==1e-4 and u['parameter_updates']['weight']['changed_elements']>0 for u in updates)
assert all(u['mean_face_soft4_before']==sequential_mean(u['per_mesh_losses']) for u in updates)
uids=[m['uid'] for m in meta['meshes']];assert len(uids)==len(set(uids))==100
assert uids==read(R/'selection.json')['uids']
baseline=evaluations[0];final=evaluations[-1]
assert complete['initial']==baseline and complete['final']==final==read(R/'run/final_real_network.json')
rows=[];trend=[];retention=[]
initial_success={m['uid'] for m in baseline['meshes'] if m['joint_perfect']};assert len(initial_success)==55
for e in evaluations:
    assert [m['uid'] for m in e['meshes']]==uids
    assert e['mean_face_soft4']==sequential_mean([m['face_soft4'] for m in e['meshes']])
    for kind in ['edge','face']:
        for key in ['tp','fp','fn']:assert e[kind+'_'+key]==sum(m[kind][key] for m in e['meshes'])
    for key in ['edge_perfect','face_perfect','joint_perfect']:assert e[key]==sum(m[key] for m in e['meshes'])
    assert (e['edge_perfect'],e['edge_fp'],e['edge_fn'])==(75,130832,1)
    assert all(m['face']['complete'] for m in e['meshes'])
    now={m['uid'] for m in e['meshes'] if m['joint_perfect']}
    ret=dict(step=e['step'],retained=sorted(initial_success&now),lost=sorted(initial_success-now),gained=sorted(now-initial_success),success_uids=sorted(now))
    retention.append(ret)
    t={k:v for k,v in e.items() if k!='meshes'}
    t.update(retained=len(ret['retained']),lost=len(ret['lost']),gained=len(ret['gained']),missing_gt_face_candidates=sum(m['missing_gt_face_candidates'] for m in e['meshes']),face_fp_outside_augmented_pool=sum(m['face']['actual_fp_outside_augmented_pool'] for m in e['meshes']))
    trend.append(t)
    for m in e['meshes']:
        b=baseline['meshes'][uids.index(m['uid'])]
        assert m['edge']==b['edge'] and m['face']['scored_candidates']==b['face']['scored_candidates']
        row=dict(step=e['step'],uid=m['uid'],vertices=m['vertices'],gt_edges=m['gt_edges'],gt_faces=m['gt_faces'],face_soft4=m['face_soft4'])
        for kind in ['edge','face']:
            for key in ['tp','fp','fn','f1']:row[kind+'_'+key]=m[kind][key]
        for key in ['edge_perfect','face_perfect','joint_perfect','gt_face_candidates','missing_gt_face_candidates']:row[key]=m[key]
        row.update(actual_face_candidates=m['face']['scored_candidates'],face_fn_candidate_present=m['face']['fn']-m['missing_gt_face_candidates'],face_fp_inside_augmented_pool=m['face']['actual_fp_inside_augmented_pool'],face_fp_outside_augmented_pool=m['face']['actual_fp_outside_augmented_pool'],min_face_gt_margin=m['margins']['face_gt_all'],min_face_non_gt_margin=m['margins']['face_non_gt_actual'])
        rows.append(row)
assert retention==complete['retention']
# Independent classification recount from saved final logits and frozen candidate IDs.
recount=[]
for rec,m in zip(meta['meshes'],final['meshes']):
    uid=m['uid'];p=R/'cache'/f'{uid}.npz';assert sha(p)==rec['cache_sha256']
    with np.load(p) as cache,np.load(R/'run/final_outputs'/f'{uid}.npz') as out:
        y=cache['actual_labels'];pred=out['actual_logits']>0
        tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=len(cache['gt_faces'])-tp
        assert (tp,fp,fn)==tuple(m['face'][k] for k in ['tp','fp','fn'])
        assert int((pred&~y&~cache['actual_in_training_pool']).sum())==m['face']['actual_fp_outside_augmented_pool']
        assert int(((out['gt_logits']>0)&cache['gt_covered']).sum())==tp
        assert np.array_equal(out['gt_logits']>0,(out['train_logits'][:len(cache['gt_faces'])]>0))
        assert len(pred)==rec['actual_candidates'] and len(out['train_logits'])==rec['train_candidates']
        assert int((~cache['gt_covered']).sum())==m['missing_gt_face_candidates']
        n=len(cache['hidden']);predkeys=np.unique(cache['predicted_edges']@np.array([n,1]));gtkeys=np.unique(cache['edges']@np.array([n,1]))
        etp=int(np.isin(predkeys,gtkeys).sum());efp=len(predkeys)-etp;efn=len(gtkeys)-etp
        assert (etp,efp,efn)==tuple(m['edge'][k] for k in ['tp','fp','fn'])
        trainpred=out['train_logits']>0;trainy=cache['train_labels'].astype(bool)
        for k,v in dict(tp=int((trainpred&trainy).sum()),fp=int((trainpred&~trainy).sum()),fn=int((~trainpred&trainy).sum()),tn=int((~trainpred&~trainy).sum())).items():assert m['training_pool'][k]==v
        recount.append(dict(uid=uid,edge_tp=etp,edge_fp=efp,edge_fn=efn,face_tp=tp,face_fp=fp,face_fn=fn))
write(R/'saved_logits_recount.json',dict(meshes=recount,all100_exact=True,optimizer_updates=0))
table(R/'trend.csv',trend);table(R/'per_mesh_evaluations.csv',rows);write(R/'retention.json',retention)
seven=['nexus_2k_'+s for s in ['000446','001957','000249','000766','001539','001969','000815']]
table(R/'seven_large_meshes.csv',[r for r in rows if r['uid'] in seven])
fnames={r['uid'] for r in final['meshes'] if r['joint_perfect']}
stable=set.intersection(*[{r['uid'] for r in e['meshes'] if r['joint_perfect']} for e in evaluations if e['step']>=300])
edge_perfect_face_failed=[m for m in final['meshes'] if m['edge_perfect'] and not m['face_perfect']]
stats=dict(updates=500,all100_every_update=True,per_mesh_direct_participation=500,records=700,strict_final=final['joint_perfect'],strict_intersection_steps300_400_500=len(stable),stable_uids=sorted(stable),retained=len(retention[-1]['retained']),lost=len(retention[-1]['lost']),gained=len(retention[-1]['gained']),edge_perfect_face_failed=edge_perfect_face_failed,clipped_updates=sum(u['clip_coefficient']<1 for u in updates),nonzero_weight_updates=sum(u['parameter_updates']['weight']['changed_elements']>0 for u in updates),nonzero_bias_updates=sum(u['parameter_updates']['bias']['changed_elements']>0 for u in updates))
write(R/'audit.json',stats)
previous=Path(cfg['previous_face_run'])
shutil.copyfile(previous/'run/final_real_network.json',R/'previous_joint71_reference.json')
old71={m['uid'] for m in read(R/'previous_joint71_reference.json')['meshes'] if m['joint_perfect']}
edge75={m['uid'] for m in baseline['meshes'] if m['edge_perfect']}
assert len(old71)==71 and len(edge75)==75 and initial_success<=old71<=edge75
lost16=old71-initial_success;newedge4=edge75-old71
historical=[]
for e in evaluations:
    now={m['uid'] for m in e['meshes'] if m['joint_perfect']}
    historical.append(dict(step=e['step'],previous71_retained=sorted(old71&now),previous71_missing=sorted(old71-now),previous71_gained=sorted(now-old71),lost16_recovered=sorted(lost16&now),new_edge4_joint_success=sorted(newedge4&now)))
write(R/'historical_joint71_retention.json',historical)
focus=edge75-initial_success
table(R/'initial_edge_perfect_face_failed20.csv',[r for r in rows if r['uid'] in focus])
# Inspect saved checkpoint files independently of the training process.
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import torch as T
source=T.load(cfg['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
first=T.load(R/'run/checkpoint-step0000.pt',map_location='cpu',mmap=True,weights_only=False)
last=T.load(R/'run/checkpoint-step0500.pt',map_location='cpu',mmap=True,weights_only=False)
derived=T.load(complete['inference_copy'],map_location='cpu',mmap=True,weights_only=False)
assert first['optimizer']['state']=={} and all(float(v['step'])==500 for v in last['optimizer']['state'].values())
assert source['model'].keys()==derived['model'].keys()
changed=[]
for k,v in source['model'].items():
    other=derived['model'][k];equal=T.equal(v,other) if T.is_tensor(v) else v==other
    if k.startswith('autoencoder.face_embedding.'):
        suffix=k.removeprefix('autoencoder.face_embedding.')
        assert T.equal(v,first['face_head'][suffix]) and T.equal(other,last['face_head'][suffix])
    else:assert equal,k
    if not equal:changed.append(k)
assert sha(Path(cfg['source_checkpoint']))==cfg['source_sha256'] and sha(Path(complete['inference_copy']))==complete['inference_sha256']
write(R/'checkpoint_file_verification.json',dict(source_sha256=cfg['source_sha256'],derived_sha256=complete['inference_sha256'],initial_face_weights_from_source=True,fresh_adam_state_empty=True,final_adam_steps_all500=True,only_face_head_allowed_to_change=True,changed_keys=changed,all_other_saved_weights_buffers_metadata_equal=True))
del source,first,last,derived
label={'step':'新增Face更新','edge_perfect':'Edge严格成功','face_perfect':'Face严格成功','joint_perfect':'联合严格成功','face_fp':'Face FP','face_fn':'Face FN'}
lines=['# 固定Tail1000：100条共享Face head恢复','',
    '已完成500次新增Face-head更新并停止。起点为本轮末端Edge累计1000步的完整模型，保留当时Face head初始化；Encoder、μ、logvar、全部Decoder、最终LayerNorm和Edge head完全冻结。只有同一份32×1024线性Face head及bias训练，共32800参数。',
    '沿用上轮Face恢复协议：fresh Adam，LR=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0，global clip=1；每次累积全部100条的完整固定pool，外层等权平均；fully-diff Soft4内部除4不变，无额外系数、MSE、Edge loss、sampling或KL。',
    '源checkpoint：`'+cfg['source_checkpoint']+'`；SHA256：`'+cfg['source_sha256']+'`。',
    f"从当前源模型重新导出安装后的hidden，FP32共{meta['hidden_total_bytes']:,}字节。原两轮扩充pool共{meta['train_candidates']:,}训练候选，当前固定预测Edge图实际产生{meta['actual_candidates']:,}个Face候选。候选图不是GT图；1个缺边导致的GT候选缺失始终计FN。",'',
    '| 新增更新 | Edge严格成功 | Face严格成功 | 联合严格成功 | 实际Face FP | 实际Face FN | Face Soft4均值 | 起点55条保留/丢失/新增 |','|---:|---:|---:|---:|---:|---:|---:|---|']
for t in trend:lines.append(f"| {t['step']} | {t['edge_perfect']} | {t['face_perfect']} | {t['joint_perfect']} | {t['face_fp']} | {t['face_fn']} | {t['mean_face_soft4']:.9f} | {t['retained']}/{t['lost']}/{t['gained']} |")
h=historical[-1]
lines+=['',f"末尾同一模型联合成功{final['joint_perfect']}/100。起点55条保留{stats['retained']}、丢失{stats['lost']}、新增{stats['gained']}。与较早的71条联合成功基线相比，保留{len(h['previous71_retained'])}、缺失{len(h['previous71_missing'])}、新增{len(h['previous71_gained'])}。",f"前轮丢失的16条中恢复{len(h['lost16_recovered'])}条，前轮新增的4条Edge成功中有{len(h['new_edge4_joint_success'])}条转化为联合成功。step300、400、500三个检查点的共同联合成功集合为{len(stable)}条；没有把未验收步骤声称为全部保持。",'',
    '全部7次检查，Edge保持75/100、FP130832、FN1。固定当前边图下联合成功最多75条；Face head本轮结果不能解决其他25条的Edge错误，也不代表100条全部通过。',
    '', '## 四条新Edge成功样本','', '| UID | 顶点 | 起点Face FP/FN | 末尾Face FP/FN | 末尾联合成功 |','|---|---:|---|---|---|']
for uid in sorted(newedge4):
    a=baseline['meshes'][uids.index(uid)];z=final['meshes'][uids.index(uid)]
    lines.append(f"| {uid} | {z['vertices']} | {a['face']['fp']}/{a['face']['fn']} | {z['face']['fp']}/{z['face']['fn']} | {z['joint_perfect']} |")
lines+=['','## 核验与材料范围','',
    '当前源模型的全部100条真实forward与前轮保存的hidden、Edge/Face embedding逐位相同；缓存训练与真实网络的Face表示、完整pool loss和Face-head梯度逐位一致。起点两次完整100条backward也逐位一致。',
    '末尾重载Face checkpoint，再从真实mesh输入完整推理100条，核对hidden、Edge表示、Face表示和实际重建复现缓存结果。所有冻结权重、buffer、metadata，训练pool、cached hidden、预测Edge图及实际Face候选均保持不变。',
    '独立读取落盘的源/派生完整模型，只有Face head两个参数允许变化；初始Adam确为空、末尾Adam为500步。保存的逐候选Face logits与标签已独立重算全部100条计数，GT缺失候选仍按FN处理。',
    'Face loss基于固定训练pool，不代表全部实际候选上的loss。较低loss不替代实际Edge/Face验收。该实验没有自动延长，也没有覆盖旧71条基线或本轮Edge候选模型。',
    '', '## 文件索引','',
    '- run/updates.jsonl：完整500条更新及每步100条Face loss、梯度、clip、实际位移。',
    '- run/evaluation-step*.json、evaluations.jsonl：7×100条实际Edge/Face、训练pool、候选覆盖与margin；trend.csv和per_mesh_evaluations.csv方便比较。',
    '- retention.json追踪本轮起点55条；historical_joint71_retention.json追踪旧71条、曾丢失16条与新增Edge4条；initial_edge_perfect_face_failed20.csv列出起点20条待恢复样本。',
    '- run/checkpoint-step*.pt：每个检查点的同一Face head及本轮新Adam；run/final_real_network.json为末尾真实网络复核。',
    '- restore/freeze信息见freeze_contract.json、cache/manifest.json、run/verification.json、checkpoint_file_verification.json；saved_logits_recount.json为独立计数复核。',
    '- runtime.py、face_core.py逐字复用上轮；protocol_adaptation.diff仅展示来源核验与结果路径适配。effective_face_scoring.py.txt及runtime_dependencies保留实际运行定义。',
    '- Review包含完整日志、代码、Face checkpoint/Adam、末尾Face表示与logits。FullEvidence另含全部新hidden、Edge表示、原mesh、固定训练/实际候选ID、label和预测Edge ID。',
    '- 原完整网络与派生完整网络不入ZIP，路径/SHA在EXCLUDED_FILES.json；Review省略的缓存也逐文件列明位置与SHA。',
    '', '末尾完整推理模型：`'+complete['inference_copy']+'`。SHA256：`'+complete['inference_sha256']+'`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
for folder in ['runtime_dependencies','effective_scoring']:shutil.copytree(previous/folder,R/folder,dirs_exist_ok=True)
shutil.copyfile(previous/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4));xs=[t['step'] for t in trend]
axs[0].plot(xs,[t['joint_perfect'] for t in trend],'o-',label='Edge + actual Face');axs[0].axhline(75,color='gray',ls='--',label='Fixed Edge upper bound');axs[0].legend();axs[0].set(ylabel='Strict meshes / 100')
for k in ['face_fp','face_fn']:axs[1].plot(xs,[t[k] for t in trend],'o-',label=k)
axs[1].set_yscale('log');axs[1].legend();axs[1].set(ylabel='Actual Face errors')
axs[2].plot(xs,[t['mean_face_soft4'] for t in trend],'o-');axs[2].set(ylabel='Mean fixed-pool Face Soft4')
for ax in axs:ax.set_xlabel('New Face-head updates');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'trend.png',dpi=160);plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.log','.txt','.diff'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json']]
for folder in ['run','cache','runtime_dependencies','effective_scoring','source_archive','review_runtime']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('model-tail1000-face1000'))
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=[p for p in base if kind=='FullEvidence' or not (p.parent==R/'cache' and p.suffix=='.npz')]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(source_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256'],bytes=Path(cfg['source_checkpoint']).stat().st_size),derived_model=dict(path=complete['inference_copy'],sha256=complete['inference_sha256'],bytes=Path(complete['inference_copy']).stat().st_size),pool_binary='Original files excluded; identical training candidate IDs/labels and vertices included in FullEvidence cache; provenance hashes in source_manifest.json.')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    dest=R/f'Nexus_FaceRecovery_Tail1000_{kind}_20260919.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files_verified=len(hashes)))
    write(R/'package_verification.json',metadata);print('PACKAGED',json.dumps(metadata[-1]),flush=True)
