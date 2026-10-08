"""Combine two completed fixed-100 runs. Read saved evidence only."""
from pathlib import Path
import csv,hashlib,json,shutil,statistics,zipfile
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Nexus_100Mesh_Epoch000-400_FullEvaluation_20260915'
OUT.mkdir(exist_ok=True)

def unpack(archive,destination,strip_root=False):
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for f in z.infolist():
            p=Path(f.filename)
            assert not p.is_absolute() and '..' not in p.parts
            if strip_root:p=Path(*p.parts[1:])
            if f.is_dir() or str(p)=='.':continue
            target=destination/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(f))
    for checksum in destination.rglob('SHA256SUMS.txt'):
        for line in checksum.read_text().splitlines():
            h,n=line.split('  ',1)
            assert hashlib.sha256((checksum.parent/n).read_bytes()).hexdigest()==h,n

unpack('/Users/luthier/Downloads/Nexus_100Mesh_Fresh512_Evaluation_20260915.zip',OUT/'phase1',True)
unpack(ROOT/'evaluation_package.zip',OUT/'phase2')
R1=OUT/'phase1/evidence/run';R2=OUT/'phase2/run'
def j(p):return json.loads(p.read_text())
def jl(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def table(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

assert j(R2/'complete.json')['updates']==10000
assert j(OUT/'phase2/runner_exit.json')['train_exit']==0
assert j(R2/'baseline_comparison.json')['exact_match']
updates=jl(R1/'updates.jsonl')+jl(R2/'updates.jsonl')
assert [r['update'] for r in updates]==list(range(1,10001))
data={r['uid']:r for r in csv.DictReader((OUT/'phase1/evidence/overfit100_manifest.csv').open())}
assert len(data)==100
counts={u:0 for u in data}
train=[]
for r in updates:
    assert len(r['meshes'])==4
    for m in r['meshes']:
        counts[m['uid']]+=1;assert counts[m['uid']]==m['participation']
        train.append(dict(update=r['update'],epoch=r['epoch'],uid=m['uid'],participation=m['participation'],
            edge_soft4=m['parts']['edge'],face_soft4=m['parts']['face'],weighted_loss=m['weighted_loss']))
assert set(counts.values())=={400}
assert all(all(x['delta_l2']>0 for x in r['actual_updates'].values()) for r in updates)
for epoch in range(1,401):
    d=j((R1 if epoch<=200 else R2)/f'epoch-order-{epoch:03d}.json')
    assert len(d['uids'])==100 and set(d['uids'])==set(data)
    got=[m['uid'] for r in updates[(epoch-1)*25:epoch*25] for m in r['meshes']]
    assert got==d['uids']
table('per_mesh_training_40000_participations.csv',train)

all_evals={}
for run in [R1,R2]:
    for f in sorted(run.glob('eval-summary-epoch*.json')):
        s=j(f);rows=jl(run/f"eval-epoch{s['epoch']:03d}.jsonl")
        assert len(rows)==100 and {r['uid'] for r in rows}==set(data)
        if s['epoch'] in all_evals:
            old=all_evals[s['epoch']][1]
            assert [{k:v for k,v in r.items() if k!='face_seconds'} for r in old]==[{k:v for k,v in r.items() if k!='face_seconds'} for r in rows]
        all_evals[s['epoch']]=(s,rows)
assert sorted(all_evals)==[0,10,25,50,100,150,200,250,300,350,400]
flat=[];trend=[]
for epoch,(s,rows) in sorted(all_evals.items()):
    for group in ['edge','face']:
        for k in ['fp','fn']:
            vals=[r[group][k] for r in rows]
            total=sum(vals) if all(v is not None for v in vals) else None
            assert total==s['total_'+group+'_'+k]
    assert s['joint_perfect']==sum(r['joint_perfect'] for r in rows)
    t=dict(epoch=epoch,update=s['updates'],edge_soft4=statistics.mean(r['parts']['edge'] for r in rows),
        face_pool_soft4=statistics.mean(r['parts']['face'] for r in rows),joint_perfect=s['joint_perfect'],face_incomplete=len(s['face_incomplete']))
    for group in ['edge','face']:
        tp=sum(r[group]['tp'] for r in rows);fp=s['total_'+group+'_fp'];fn=s['total_'+group+'_fn']
        t.update({group+'_tp':tp,group+'_fp':fp,group+'_fn':fn,
            group+'_micro_precision':tp/(tp+fp) if fp is not None and tp+fp else None,
            group+'_micro_recall':tp/(tp+fn) if tp+fn else None})
    t['missing_gt_face_candidates']=sum(r['missing_gt_face_candidates'] for r in rows)
    t['face_fn_with_candidate']=t['face_fn']-t['missing_gt_face_candidates']
    t['face_fp_outside_training_pool']=sum(r['face']['actual_fp_outside_training_pool'] for r in rows) if not s['face_incomplete'] else None
    trend.append(t)
    for r in rows:
        d=data[r['uid']];v=dict(epoch=epoch,update=s['updates'],uid=r['uid'],vertices=int(d['vertices']),gt_edges=int(d['gt_edges']),gt_faces=int(d['gt_faces']),participations=r['participations'],edge_soft4=r['parts']['edge'],face_pool_soft4=r['parts']['face'])
        for group in ['edge','face']:
            for k in ['tp','fp','fn','f1']:v[group+'_'+k]=r[group][k]
        v.update(joint_perfect=r['joint_perfect'],face_complete=r['face']['complete'],
            missing_gt_face_candidates=r['missing_gt_face_candidates'],face_fn_with_candidate=r['face']['fn']-r['missing_gt_face_candidates'],
            face_fp_outside_training_pool=r['face']['actual_fp_outside_training_pool'])
        v.update({'min_margin_'+k:value for k,value in r['margins'].items()});flat.append(v)
table('evaluation_trend_epoch000-400.csv',trend)
table('per_mesh_all_1100_evaluations.csv',flat)
final=[r for r in flat if r['epoch']==400];table('per_mesh_final_epoch400.csv',final)
base={r['uid']:r for r in flat if r['epoch']==200};comparison=[]
for r in final:
    b=base[r['uid']];d=dict(uid=r['uid'],vertices=r['vertices'])
    for k in ['edge_soft4','face_pool_soft4','edge_fp','edge_fn','face_fp','face_fn','missing_gt_face_candidates','face_fp_outside_training_pool']:
        d[k+'_epoch200']=b[k];d[k+'_epoch400']=r[k];d[k+'_change']=r[k]-b[k]
    d.update(perfect_epoch200=b['joint_perfect'],perfect_epoch400=r['joint_perfect']);comparison.append(d)
table('per_mesh_epoch200_vs400.csv',comparison)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
x=[r['update'] for r in trend]
for k,label in [('edge_soft4','Edge'),('face_pool_soft4','Face: fixed training pool')]:axes[0,0].plot(x,[r[k] for r in trend],'o-',label=label)
for k in ['edge_fp','edge_fn','face_fp','face_fn']:axes[0,1].plot(x,[r[k] if r[k] is not None else float('nan') for r in trend],'o-',label=k)
axes[0,1].set_yscale('symlog')
axes[1,0].plot(x,[r['joint_perfect'] for r in trend],'o-',label='Strict Edge + actual Face')
axes[1,0].set_ylim(-1,101)
for k in ['missing_gt_face_candidates','face_fn_with_candidate']:axes[1,1].plot(x,[r[k] for r in trend],'o-',label=k)
axes[1,1].set_yscale('symlog')
for a,title in zip(axes.flat,['Evaluation Soft4','Actual reconstruction errors','Strict pass /100','Actual Face FN decomposition']):
    a.set_title(title);a.set_xlabel('Cumulative optimizer updates');a.axvline(5000,color='grey',ls='--',alpha=.5);a.grid(alpha=.25);a.legend(fontsize=8)
fig.suptitle('Fixed100 fresh512: first 200 epochs + resumed 200 epochs')
fig.savefig(OUT/'full_training_evaluation.png',dpi=160);plt.close(fig)
last=trend[-1];old=next(t for t in trend if t['epoch']==200)
passed=[r['uid'] for r in final if r['joint_perfect']]
improved={k:sum(r[k+'_change']<0 for r in comparison) for k in ['edge_fp','edge_fn','face_fp','face_fn']}
summary=dict(completed=True,total_updates=10000,additional_updates=5000,total_epochs=400,
    per_mesh_participations=400,unique_evaluations=1100,epoch200_repeated_load_verification=100,
    baseline=old,final=last,final_perfect_uids=passed,meshes_improved_vs_epoch200=improved,
    final_checkpoint=all_evals[400][0]['checkpoint'],final_checkpoint_sha256=all_evals[400][0]['checkpoint_sha256'])
(OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
checkpoint_verification=j(ROOT/'final_checkpoint_hash_verification.json')
assert checkpoint_verification['final_checkpoint_hash_verified'] and checkpoint_verification['sha256']==summary['final_checkpoint_sha256']
shutil.copy2(ROOT/'final_checkpoint_hash_verification.json',OUT/'final_checkpoint_hash_verification.json')
lines=['# 固定100条：400 epochs完整评估包','',
    '两段预算均已完成。第一段5000更新，第二段完整恢复后新增5000更新；累计10000更新、400epochs，每条mesh直接参与400次。',
    '本包基于保存日志和代码做只读汇总，没有新增optimizer update，也没有重新运行模型。','',
    f"最终严格成功：{last['joint_perfect']}/100。成功UID：{', '.join(passed) or '无'}。",
    f"末尾100条中Face枚举未完成：{last['face_incomplete']}条。",
    '', '|阶段|Edge FP/FN|实际Face FP/FN|严格成功|','|---|---|---|---|',
    f"|epoch200|{old['edge_fp']}/{old['edge_fn']}|{old['face_fp']}/{old['face_fn']}|{old['joint_perfect']}/100|",
    f"|epoch400|{last['edge_fp']}/{last['edge_fn']}|{last['face_fp']}/{last['face_fn']}|{last['joint_perfect']}/100|",'',
    f"末尾漏面分解：缺边未入候选{last['missing_gt_face_candidates']}，候选存在但判负{last['face_fn_with_candidate']}。",
    f"末尾实际Face FP中训练pool之外的数量：{last['face_fp_outside_training_pool']}。",'',
    '## 先看这些文件',
    '- full_training_evaluation.png：累计10000步的同口径完整评估趋势。',
    '- evaluation_trend_epoch000-400.csv：11轮去重后的全量指标，Precision/Recall为micro汇总。',
    '- per_mesh_final_epoch400.csv：最终100条逐mesh错误、候选覆盖、margin与严格成功。',
    '- per_mesh_epoch200_vs400.csv：同UID前后对照；change=末尾减起点，负值表示计数减少。',
    '- per_mesh_all_1100_evaluations.csv：全部去重后的逐mesh评估。原始epoch200加载复核仍保留在phase2中。',
    '- per_mesh_training_40000_participations.csv：每次mesh直接参与训练的记录。',
    '- phase1/：第一段完整评估包，原始5000步更新日志在evidence/run/updates.jsonl。',
    '- phase2/：第二段原始5000步日志、5轮完整评估、恢复核验、完成记录、代码/配置/候选清单。',
    '- SUMMARY.json、verification.json、SHA256SUMS.txt：汇总、独立一致性检查、包内哈希。','',
    '## 必须保留的解释边界',
    '训练固定100条，μ路径，KL=0，logvar冻结；不等于已恢复sampling/KL的VAE阶段。有效batch4是4个完整mesh微批loss/4累积后统一更新，不是每条训练10000次。',
    '完整验收在同checkpoint暂停更新后依次前向100条，不是一次packed100。Face从当次预测Edge图枚举。Face Soft4在固定训练pool上计算，不能当作实际候选全集loss。',
    '原epoch0有未完整枚举，FP/F1保留null；统计表空格不是0。严格成功必须同checkpoint所有Edge/实际Face FP=FN=0；不同检查点成功身份不得合并计算。',
    '区分已恢复的GT、仍然大量存在的误报与严格成功；loss下降和高Recall不能替代100/100验收。',
    '包内不含大checkpoint、mesh/pool二进制或未记录的逐候选logit/embedding；它是结果审阅包，不是离线推理部署包。父文件路径/已有SHA见phase1/checkpoint_inventory.json和manifest；第二段验收checkpoint路径/SHA见phase2/run/eval-summary-*.json。',
    f"最终checkpoint：{summary['final_checkpoint']}\n\nSHA256：{summary['final_checkpoint_sha256']}",
    '源代码包括原始快照、实际运行时替换及恢复入口；args.precision历史字符串不能替代manifest.backend与实际math00实现。',
    '当前预算已经停止。没有自动追加训练、调LR、改Face pool或新建其他实验。']
(OUT/'START_HERE_评估说明.md').write_text('\n\n'.join(lines)+'\n')
verification=dict(zip_crc_passed=True,input_file_checksums_passed=True,updates_contiguous_1_to10000=True,
    all400_epoch_uid_orders_verified=True,each_mesh_participations400=True,all_four_groups_updated_each_step=True,
    every_evaluation_uid_set_and_aggregate_verified=True,resume_baseline_exact=True,normal_exit=True,final_server_checkpoint_hash_recomputed=True)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
shutil.copy2(__file__,OUT/'build_final_handoff.py')
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p!=OUT/'SHA256SUMS.txt')
(OUT/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in files))
dest=Path('/Users/luthier/Downloads')/(OUT.name+'.zip')
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,str(p.relative_to(OUT.parent)))
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
print(json.dumps(dict(archive=str(dest),bytes=dest.stat().st_size,summary=summary),indent=2))
