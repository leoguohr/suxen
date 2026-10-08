"""Audit saved outputs and package evidence after the fixed budget; no updates."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import csv,json,zipfile,hashlib
from pathlib import Path
import numpy as np
import torch
from loader import sha
torch.set_num_threads(1)
R=Path(__file__).resolve().parent;run=R/'run'
done=json.loads((R/'complete.json').read_text());cfg=json.loads((R/'config.json').read_text())
assert done['updates']==2000 and done['stopped_at_budget']
logs=[json.loads(x) for x in (run/'updates.jsonl').read_text().splitlines()]
assert [x['update'] for x in logs]==list(range(1,2001))
for x in logs:
    assert [m['uid'] for m in x['meshes']]==cfg['uids'] and x['mesh_count']==50
    assert all(m['participations']==x['update'] for m in x['meshes'])
    assert all(x['lrs'][g]==target*min(x['update']/100,1) for g,target in cfg['lr_target'].items())
    assert all(np.isfinite(v['delta_l2']) for v in x['actual_updates'].values())
evals=[];flat=[];artifacts=[];prediction_count=0
for step in cfg['checkpoints']:
    e=json.loads((run/f'eval-step{step:04d}.json').read_text());assert e['step']==step and len(e['meshes'])==50
    assert [x['uid'] for x in e['meshes']]==cfg['uids'] and sha(e['checkpoint'])==e['checkpoint_sha256']
    cp=torch.load(e['checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==step and set(cp['participation'].values())=={step}
    if step==0:assert not cp['optimizer']['state']
    else:
        assert all(int(s['step'])==step for s in cp['optimizer']['state'].values())
        assert len(cp['optimizer']['state'])==sum(len(g['params']) for g in cp['optimizer']['param_groups'])
    for group in cp['optimizer']['param_groups']:
        assert tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0
    frozen={k:v for k,v in cp['model'].items() if k.startswith('autoencoder.log_variance.')}
    if step==0:initial_logvar={k:v.clone() for k,v in frozen.items()}
    else:assert all(torch.equal(v,initial_logvar[k]) for k,v in frozen.items())
    artifacts.append(dict(path=e['checkpoint'],sha256=e['checkpoint_sha256'],bytes=Path(e['checkpoint']).stat().st_size,step=step))
    for row in e['meshes']:
        p=R/row['prediction_path'];assert sha(p)==row['prediction_sha256']
        with np.load(p) as q:
            n=len(q['vertices']);ge=q['gt_edges'];pe=q['predicted_edge_ids'];gf=q['gt_faces'];fi=q['actual_face_candidate_ids']
            edgekey=lambda a:a[:,0].astype(np.int64)*n+a[:,1]
            facekey=lambda a:(a[:,0].astype(np.int64)*n+a[:,1])*n+a[:,2]
            ek=edgekey(pe);assert len(np.unique(ek))==len(ek) and (q['predicted_edge_logits']>0).all()
            etp=int(np.isin(ek,edgekey(ge)).sum());efp=len(pe)-etp;efn=len(ge)-etp
            assert [etp,efp,efn]==[row['edge'][k] for k in ['tp','fp','fn']]
            fk=facekey(fi);y=np.isin(fk,facekey(gf));pred=q['actual_face_candidate_logits']>0
            assert len(np.unique(fk))==len(fk) and np.array_equal(y,q['actual_face_candidate_gt'])
            ftp=int((pred&y).sum());ffp=int((pred&~y).sum());ffn=len(gf)-ftp
            assert [ftp,ffp,ffn]==[row['face'][k] for k in ['tp','fp','fn']]
            adj=[set() for _ in range(n)]
            for i,j in pe:adj[int(i)].add(int(j))
            assert sum(len(adj[i]&adj[j]) for i in range(n) for j in adj[i])==len(fi)
            for i,j in [(0,1),(0,2),(1,2)]:assert np.isin(fi[:,i].astype(np.int64)*n+fi[:,j],ek).all()
            assert row['joint_perfect']==(efp==efn==ffp==ffn==0)
            assert row['face']['fn_present_but_negative']+row['face']['fn_missing_candidate']==ffn
        prediction_count+=1
        flat.append(dict(step=step,uid=row['uid'],vertices=row['vertices'],**row['parts'],
            **{kind+'_'+k:row[kind][k] for kind in ['edge','face'] for k in ['tp','fp','fn']},
            joint_perfect=row['joint_perfect'],face_fn_missing=row['face']['fn_missing_candidate'],
            face_fn_present=row['face']['fn_present_but_negative'],face_fp_in_pool=row['face']['actual_fp_inside_training_pool'],face_fp_out_pool=row['face']['actual_fp_outside_training_pool']))
    assert e['joint_perfect']==sum(x['joint_perfect'] for x in e['meshes'])
    evals.append(e);del cp
    print('AUDIT',step,flush=True)
assert prediction_count==1050
for p in [R/'initial.pt',Path(done['full_model'])]:artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
assert sha(done['full_model'])==done['full_model_sha256']
frozen=json.loads((R/'protected_mainline.json').read_text())
for p,digest in frozen.items():assert sha(p)==digest,p
audit=dict(actual_updates=2000,total_mesh_participations=100000,per_mesh=2000,prediction_npz_verified=prediction_count,
    complete_predicted_graph_triangle_enumerations_verified=True,all_checkpoint_adam_steps_verified=True,
    logvar_unchanged=True,protected_mainline_unchanged=True,scope='independent CPU file/state/prediction audit, no additional training or network rerun')
recovery=None
if (R/'recovery.json').exists():
    recovery=json.loads((R/'recovery.json').read_text())
    assert recovery['replay_updates_verified']==87 and recovery['durable_restore_update']==200
    old=[json.loads(x) for x in (R/'interruption_20260921/updates.jsonl').read_text().splitlines()]
    assert [x['update'] for x in old]==list(range(1,288))
    for x,y in zip(old,logs):
        assert {k:v for k,v in x.items() if k!='seconds'}=={k:v for k,v in y.items() if k!='seconds'}
    recovered=Path(recovery['replay_checkpoint']);assert sha(recovered)==recovery['replay_checkpoint_sha256']
    artifacts.append(dict(path=str(recovered),bytes=recovered.stat().st_size,sha256=sha(recovered),step=287))
    audit.update(durable_restore_update=200,interrupted_recorded_updates=287,
        recovered_updates=1800,replayed_updates_verified=87,recorded_optimizer_calls_including_replay=2087,
        logical_trajectory_updates=2000,old_log_preserved=True)
(R/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(R/'MODEL_ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
with (R/'per_mesh.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
complexity=json.loads((R/'data_complexity.json').read_text());final=evals[-1]
success=any(e['joint_perfect']==50 for e in evals)
lines=['# Teacher CAD50 × 当前512维AE：2000次全50条更新','',
    ('本轮在同一个checkpoint达到50/50 Edge＋实际Face严格零错误。' if success else f"本轮固定预算内未达到50/50；末尾联合严格成功{final['joint_perfect']}/50。"),'',
    f"最高联合成功{done['best']['joint_perfect']}/50（step{done['best']['step']}）；首次50/50：{done['first50perfect_step']}。2000次完整累积更新已执行并停止，每条直接参与2000次，共100000次完整mesh参与。",'',
    '## 数据身份与难度','',
    '附件是nexus_overfit_data_results_no_code.zip，而非另一个teacher50_for_current_ae.zip。已直接从其实际训练缓存data/point50/training.pt无损导出50份FP32 mesh；未加载老师权重或预测，未修改坐标、编号、面索引或目标。', '',
    '总计2872顶点、8364 GT边、5576 GT面、235741个pair；00保留68点，13保留16点，20是最大274点样本。', '',
    '老师50条顶点数中位数12，原100条为953.5；老师24条为同构的8点12面拓扑，全部50条共23类顶点—面拓扑。原100条的Edge pair总量约为其359.16倍。拓扑同构统计忽略坐标与面朝向，仅作只读难度描述，不改训练目标。', '',
    '这些差异支持两组任务规模和重复度明显不同；不能单凭此统计证明老师速度的唯一原因，也不能将本轮结果自动推广到原100条大mesh。', '',
    '## 实际执行','',
    '- seed0正常随机初始化，fresh Adam，完整Encoder/μ/16块Decoder/两共享线性head训练；logvar冻结，μ路径，KL=0。',
    '- E/μ目标LR=1e-5，D/heads=1e-4；lr(t)=目标LR×min(t/100,1)。每步每条loss先除50，累积50条后统一clip=1、Adam一次。',
    '- math00与原有效评分不变；fully-diff Soft4内部/4保留，membership/分母均有梯度，无额外外层0.25。',
    '- GT建池基础规则：全部非GT三环＋最多1F wedge＋0.5F uniform，唯一负例不足不重复补数；13946个训练候选在开跑前固定，没有模型挖掘或动态刷新。',
    '- 预检没有执行optimizer更新；已核验全部50条重复forward、μ路径、全部可训练参数梯度连通及00/13/20与原有效loss及评分表示梯度逐位一致。',
    '- 训练未缓存可训练网络输出。21个检查点均冻结同一完整网络，逐条执行真实Encoder→μ→Decoder；实际Face完整来自当次预测Edge图。',
    '- 预检两次脚手架检查失败分别是确定性安装顺序、读取split μ视图的梯度；修正的是安装顺序和梯度记录hook，未更换计算公式、未偷跑更新。原始日志已保留。','',
    '## 完整验收轨迹','',
    '| step | Edge loss | Face pool loss | Edge FP/FN | 实际Face FP/FN | Edge micro-F1 | Face micro-F1 | 联合严格成功 |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|']
for e in evals:
    ec=e['counts']['edge'];fc=e['counts']['face']
    lines.append(f"| {e['step']} | {e['losses']['edge']:.8g} | {e['losses']['face']:.8g} | {ec['fp']}/{ec['fn']} | {fc['fp']}/{fc['fn']} | {ec['micro_f1']:.9f} | {fc['micro_f1']:.9f} | {e['joint_perfect']}/50 |")
lines+=['','只声明这些实际验收过的检查点，不将间隔期间写成每一步均成功。严格成功以四项FP/FN均为0判断，不使用四舍五入F1。','',
    '## 末尾未通过样本','',
    '| UID | 顶点 | Edge FP/FN | Face FP/FN | Face缺候选/候选判负 | Face FP池内/池外 |',
    '|---|---:|---:|---:|---:|---:|']
for r in final['meshes']:
    if r['joint_perfect']:continue
    e=r['edge'];f=r['face'];lines.append(f"| {r['uid']} | {r['vertices']} | {e['fp']}/{e['fn']} | {f['fp']}/{f['fn']} | {f['fn_missing_candidate']}/{f['fn_present_but_negative']} | {f['actual_fp_inside_training_pool']}/{f['actual_fp_outside_training_pool']} |")
lines+=['','00、13、20号的每个检查点明细位于evaluation JSON的special_cases及per_mesh.csv，全部保留，没有删样本。','',
    '## 交付与边界','',
    'Review包含数据/训练pool、有效代码与来源、配置、预检、全部更新与逐mesh评价、SHA和本报告。Predictions包含21×50条实际预测Edge以及完整Face候选ID/logit/GT label。',
    '完整权重、Adam和RNG均保存在服务器的checkpoint中；首次50/50（若有）、最高成功及最终checkpoint路径由first50perfect.json、best.json、complete.json标记。大checkpoint未重复打入评估ZIP；文件路径与SHA在MODEL_ARTIFACTS.json。',
    '固定100条主线的保护文件哈希在运行结束再次核对。接口预检、真实完整网络训练、实际重建验收分别留有记录；独立CPU审计未再次运行网络。',
    '有限配置下的失败不证明容量不足；本小数据成功也不代表原100条已解决。']
if recovery:
    lines+=['','## 服务器中断与精确恢复','',
        '原服务器在记录第287步后停止，最近完整checkpoint为step200。新服务器恢复该点完整权重、四组Adam和RNG；50条保存的step200 loss均精确复现。',
        '201—287共87步作确定性恢复重放：除墙钟耗时外，每条原记录的loss、梯度、clip、LR和实际参数更新统计均精确一致。旧287步日志完整归档，恢复代码及差异一同交付。',
        '最终模型轨迹是step1—2000，每条有效参与2000次；因中断重放，已记录的物理optimizer调用总计2087，不能称完全没有额外重算。未自动延长模型的2000步预算，未重置Adam或改变配置。',
        'complete.json的seconds为本次恢复入口的耗时，不是第一次启动以来的总墙钟时间。完整恢复核验见resume_precheck.json、recovery.json及interruption_20260921。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
handoff=R/'NEW_SESSION_PROMPT.md'
if handoff.exists():
    text=handoff.read_text();start='<!-- FINAL_RESULTS_START -->';end='<!-- FINAL_RESULTS_END -->'
    assert text.count(start)==text.count(end)==1
    best=done['best'];remaining=[x['uid'] for x in final['meshes'] if not x['joint_perfect']]
    result=[start,'**CAD50训练及服务器文件审计已完成，已按2000次有效轨迹更新停止。**',
        f"同一checkpoint达到50/50：{'是' if success else '否'}；首次50/50 step：{done['first50perfect_step']}；最高联合严格成功：{best['joint_perfect']}/50（step{best['step']}）；末尾：{final['joint_perfect']}/50（step2000）。",
        f"末尾Edge FP/FN={final['counts']['edge']['fp']}/{final['counts']['edge']['fn']}；实际Face FP/FN={final['counts']['face']['fp']}/{final['counts']['face']['fn']}。",
        '末尾未联合严格成功UID：'+(', '.join(remaining) if remaining else '无')+'。',
        '21个固定checkpoint均实际完整网络评价全部50条，独立CPU审计核对1050条预测、完整三角形枚举、全部checkpoint/Adam和保护模型哈希；文件审计没有另外重跑网络。',
        '仅对已评价的检查点声称成功；不把检查点间隔中的全部更新称为均保持成功。',
        f"最高成功checkpoint：`{best['checkpoint']}`；SHA256 `{best['sha256']}`。",
        f"最终可续训checkpoint：`{final['checkpoint']}`；SHA256 `{final['checkpoint_sha256']}`。",
        f"最终完整推理模型：`{done['full_model']}`；SHA256 `{done['full_model_sha256']}`。"]
    if done['first50perfect_step'] is not None:
        first=json.loads((run/'first50perfect.json').read_text())
        result.append(f"首次50/50 checkpoint：`{first['checkpoint']}`；SHA256 `{first['sha256']}`。")
    if recovery:
        result.append('从step200恢复并精确重放201—287；最终有效轨迹2000步，每条有效参与2000次。含87步中断恢复重算的已记录物理Adam调用合计2087，未自动增加模型轨迹预算。')
    result += [f"服务器评估包：`{R/'TeacherCAD50_Fresh512_2000_Review.zip'}`；预测包：`{R/'TeacherCAD50_Fresh512_2000_Predictions.zip'}`。本段由最终审计生成，随后执行打包；整包SHA以成功生成的package_manifest.json为准。",
        '本地Downloads是否已经同步、通知是否已发送，仍需接手时核对，不在此自动声称完成下载。',
        '这证明或未证明的是当前配置在老师CAD50上的严格拟合；不能自动推论原100条大mesh已解决。规模/拓扑重复事实见data_complexity.json，训练结果和剩余错误见REPORT.md及per_mesh.csv。',end]
    text=text[:text.index(start)]+'\n\n'.join(result)+text[text.index(end)+len(end):]
    handoff.write_text(text)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
xs=[e['step'] for e in evals]
for k in ['edge','face']:
    axes[0].plot(xs,[e['losses'][k] for e in evals],label=k)
    axes[1].plot(xs,[e['counts'][k]['fp'] for e in evals],label=k+' FP')
    axes[1].plot(xs,[e['counts'][k]['fn'] for e in evals],label=k+' FN')
axes[2].plot(xs,[e['joint_perfect'] for e in evals],'o-',label='joint strict /50');axes[2].set_ylim(-1,51)
for ax,title in zip(axes,['Per-mesh mean Soft4','Actual reconstruction errors','Same-checkpoint strict success']):ax.set(title=title,xlabel='Full50 Adam updates');ax.legend();ax.grid(alpha=.2)
fig.savefig(R/'training_trend.png',dpi=170);plt.close(fig)
excluded={'initial.pt','teacher_training_cache.pt'}
items=[];pred=[]
for p in R.rglob('*'):
    if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.suffix in ['.pt','.zip'] or p.name in ['package_manifest.json','execution.lock']:continue
    name=str(p.relative_to(R))
    if p.suffix=='.npz' and 'predictions-step' in name:pred.append((p,name))
    elif p.suffix in ['.py','.sh','.md','.json','.jsonl','.csv','.log','.npz','.png','.txt','.diff']:items.append((p,name))
def package(name,files):
    path=R/name;manifest=[]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p,n in sorted(files,key=lambda x:x[1]):
            manifest.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p)));z.write(p,n)
        z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,files=len(manifest))
packages=dict(review=package('TeacherCAD50_Fresh512_2000_Review.zip',items),predictions=package('TeacherCAD50_Fresh512_2000_Predictions.zip',pred))
(R/'package_manifest.json').write_text(json.dumps(packages,indent=2)+'\n')
print('AUDIT_AND_PACKAGING_COMPLETE',json.dumps(packages),flush=True)
