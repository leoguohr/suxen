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
initial_success={m['uid'] for m in baseline['meshes'] if m['joint_perfect']}
for e in evaluations:
    assert [m['uid'] for m in e['meshes']]==uids
    assert e['mean_face_soft4']==sequential_mean([m['face_soft4'] for m in e['meshes']])
    for kind in ['edge','face']:
        for key in ['tp','fp','fn']:assert e[kind+'_'+key]==sum(m[kind][key] for m in e['meshes'])
    for key in ['edge_perfect','face_perfect','joint_perfect']:assert e[key]==sum(m[key] for m in e['meshes'])
    assert (e['edge_perfect'],e['edge_fp'],e['edge_fn'])==(71,156784,1)
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
lines=['# 固定100条hidden：共享Face head恢复实验','',
    '已完成500次更新并按预算停止。每次全部100条分别计算完整固定训练pool的Face Soft4，除100后累积，统一clip=1，再执行一次fresh Adam更新。只有一个32×1024线性Face head及bias可训练，共32800参数。',
    '起点为安装Decoder末端后的完整推理副本；没有使用安装前的旧hidden。LR=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0，无warmup。训练只含fully-differentiable Face Soft4，内部固定除4，无额外诊断系数，无Edge loss、KL、sampling或MSE。',
    '源文件：`'+cfg['source_checkpoint']+'`。SHA256：`'+cfg['source_sha256']+'`。',
    f"共缓存{meta['hidden_total_bytes']:,}字节FP32 hidden；原两轮扩充训练pool共{meta['train_candidates']:,}候选，固定预测Edge图产生{meta['actual_candidates']:,}个实际Face候选。两者没有混用或自动合并。所有GT中1个Face因缺边不可进入实际候选，验收始终计FN。",'',
    '| 更新 | Edge严格成功 | Face严格成功 | 联合严格成功 | 实际Face FP | 实际Face FN | Face Soft4均值 | 原50条保留/丢失/新增 |','|---:|---:|---:|---:|---:|---:|---:|---|']
for t in trend:lines.append(f"| {t['step']} | {t['edge_perfect']} | {t['face_perfect']} | {t['joint_perfect']} | {t['face_fp']} | {t['face_fn']} | {t['mean_face_soft4']:.9f} | {t['retained']}/{t['lost']}/{t['gained']} |")
lines+=['',f"末尾联合严格成功{final['joint_perfect']}/100；原50条保留{stats['retained']}、丢失{stats['lost']}、新增{stats['gained']}。step300、400、500三个检查点共同成功{len(stable)}条。这是检查点保持证据，没有把未评估步骤宣称为全部成功。",'',
    f"Edge在7次全量检查均为71/100、FP156784、FN1。Face FP由{baseline['face_fp']}变为{final['face_fp']}，FN由{baseline['face_fn']}变为{final['face_fn']}。较低训练pool loss不保证池外候选改善，实际FP/FN仍是独立验收。固定Edge图下，联合成功数上限71；其余Edge不完整样本不可能仅靠Face head成为完整mesh。",'',
    '## 7条新增Edge成功大mesh','', '| UID | 起点Face FP/FN | 末尾Face FP/FN | 末尾联合成功 |','|---|---:|---:|---|']
for uid in seven:
    a=baseline['meshes'][uids.index(uid)];z=final['meshes'][uids.index(uid)]
    lines.append(f"| {uid} | {a['face']['fp']}/{a['face']['fn']} | {z['face']['fp']}/{z['face']['fn']} | {z['joint_perfect']} |")
lines+=['','## 验证和边界','',
    '起点100条真实网络前向与缓存路径的Face embedding、完整pool loss和head梯度逐位相同；两次完整100条backward梯度逐位相同。全部冻结权重、buffer与metadata哈希不变；缓存hidden、训练pool文件和实际候选集合哈希不变。',
    '末尾重载Face head checkpoint，从真实mesh输入重跑100条Encoder→μ→Decoder，hidden、Edge embedding、Face embedding和完整实际重建逐位/逐项复现缓存验收。没有用GT图替代预测Edge图，没有安装独立mesh专属head。',
    '保存的末尾逐候选logits与label已经独立重算全部100条Face及训练pool计数；保存的预测Edge ID与GT ID也独立重算，和记录完全一致。完整源模型文件SHA未变，派生模型另存。',
    '只证明本轮有限预算下同一个Face head的结果，不证明100条全部通过，不将不同checkpoint的成功拼接，不自动追加预算。Face head之外参数没有更新；bias的数值变化仍记录，逐mesh中心化使bias对精确实数评分无影响。',
    '', '## 材料索引','',
    '- run/updates.jsonl：完整500条更新，逐mesh Face loss、梯度、clip和实际参数更新。',
    '- run/evaluation-step*.json、evaluations.jsonl：7次×100条实际Edge/Face、候选覆盖、margin和固定pool指标。',
    '- trend.csv、per_mesh_evaluations.csv、retention.json、seven_large_meshes.csv：趋势与UID级保留/丢失/新增。',
    '- run/checkpoint-step*.pt：7份共享Face head和对应新Adam状态；只训练Face head，不是完整网络的500轮训练。',
    '- run/verification.json、final_real_network.json、audit.json、saved_logits_recount.json：冻结核验、真实网络重接与独立计数复核。',
    '- cache/manifest.json：缓存和原两轮pool哈希。FullEvidence内cache/*.npz含安装后hidden、Edge表示、预测边图、实际Face/训练候选ID和label、原始mesh数组。',
    '- run/final_outputs/*.npz（两种包均含）：末尾Face embedding、完整训练pool/实际候选/全部GT Face logits，与FullEvidence缓存逐行对齐。',
    '- runtime.py、face_core.py、run.py、effective_face_scoring.py.txt、runtime_dependencies：实际执行代码与运行时替换。',
    '- EXCLUDED_FILES.json：大型完整网络源/派生checkpoint及Review省略缓存的服务器路径、大小和SHA。Review用于快速审阅；FullEvidence包含全部缓存，原始大型网络权重另存服务器。',
    '', '派生推理模型：`'+complete['inference_copy']+'`。SHA256：`'+complete['inference_sha256']+'`。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
parent=Path(cfg['parent'])
for folder in ['runtime_dependencies','effective_scoring']:shutil.copytree(parent/folder,R/folder,dirs_exist_ok=True)
shutil.copyfile(parent/'runtime_sources.json',R/'runtime_sources.json')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,3,figsize=(14,4))
xs=[t['step'] for t in trend]
axs[0].plot(xs,[t['joint_perfect'] for t in trend],'o-',label='Edge + actual Face');axs[0].axhline(71,color='gray',ls='--',label='Fixed Edge upper bound');axs[0].set(ylabel='Strict meshes / 100');axs[0].legend()
for key in ['face_fp','face_fn']:axs[1].plot(xs,[t[key] for t in trend],'o-',label=key)
axs[1].set_yscale('log');axs[1].set(ylabel='Actual Face error count');axs[1].legend()
axs[2].plot(xs,[t['mean_face_soft4'] for t in trend],'o-');axs[2].set(ylabel='Mean fixed-pool Face Soft4')
for ax in axs:ax.set_xlabel('Face-head updates');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(R/'trend.png',dpi=160);plt.close(fig)
base=[p for p in R.iterdir() if p.is_file() and p.suffix in ['.py','.json','.jsonl','.csv','.md','.png','.log','.txt'] and p.name not in ['package_verification.json','SHA256SUMS.txt','EXCLUDED_FILES.json']]
for folder in ['run','cache','runtime_dependencies','effective_scoring','source_archive','review_runtime']:
    base.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('model-tail500-face500'))
base=sorted(set(base));metadata=[]
for kind in ['Review','FullEvidence']:
    files=[p for p in base if kind=='FullEvidence' or not (p.parent==R/'cache' and p.suffix=='.npz')]
    excluded={str(p.relative_to(R)):dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in base if p not in files}
    excluded.update(source_model=dict(path=cfg['source_checkpoint'],sha256=cfg['source_sha256'],bytes=Path(cfg['source_checkpoint']).stat().st_size),derived_model=dict(path=complete['inference_copy'],sha256=complete['inference_sha256'],bytes=Path(complete['inference_copy']).stat().st_size),pool_binary='Original files excluded; identical training candidate IDs/labels and vertices are present in FullEvidence cache, provenance hashes in source_manifest.json.')
    extra=json.dumps(excluded,indent=2).encode();hashes={str(p.relative_to(R)):sha(p) for p in files};hashes['EXCLUDED_FILES.json']=hashlib.sha256(extra).hexdigest()
    dest=R/f'Nexus_FaceHead_Recovery100_{kind}_20260918.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(R)))
        z.writestr('EXCLUDED_FILES.json',extra);z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n,h in hashes.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
    metadata.append(dict(kind=kind,path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size,files_verified=len(hashes)))
    write(R/'package_verification.json',metadata);print('PACKAGED',json.dumps(metadata[-1]),flush=True)
