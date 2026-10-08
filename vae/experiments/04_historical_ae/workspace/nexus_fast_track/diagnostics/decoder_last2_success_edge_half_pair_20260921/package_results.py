"""Evidence-bounded paired training report and downloadable archives."""
import hashlib,json,zipfile,platform
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent/'decoder_last2_joint_fixed100_20260920'
cfg=json.loads((R/'config.json').read_text());done=json.loads((R/'complete.json').read_text())
audit=json.loads((R/'independent_audit.json').read_text());comparison=json.loads((R/'comparison.json').read_text())
assert done['total_branch_optimizer_steps']==1000 and len(audit['prediction_checks'])==16
data={b:{step:json.loads((R/b/'run'/f'actual-new{step:04d}.json').read_text()) for step in cfg['checkpoints']} for b in cfg['branches']}
A=data['A_control'][500];B=data['B_success_edge_half'][500];base=data['A_control'][0]
def summary(e):return e['summary']['all']
aa,bb=summary(A),summary(B)
fe_delta=B['components']['failed_edge']-A['components']['failed_edge']
edge_delta=B['summary']['F']['edge']['fp']-A['summary']['F']['edge']['fp']
if edge_delta>0 and fe_delta<0:
    verdict='困难Edge训练loss降低，但该组实际Edge FP更多，不能称为困难Edge结构恢复改善。'
elif edge_delta<0:
    verdict='该组实际Edge FP减少；还须同时结合FN、旧成功保持和严格覆盖判断收益。'
else:
    verdict='困难Edge实际FP没有减少，需以完整FP/FN和严格覆盖判断，而非加权loss。'
if bb['joint_perfect']<=aa['joint_perfect']:
    verdict+='本次500步预算内，B没有比Control扩大同checkpoint联合严格成功覆盖。'
else:
    verdict+='本次500步预算内，B的同checkpoint联合严格成功数超过Control。'
lines=['# 固定100条：历史成功组Edge降权的同起点配对对照','',
 f"A、B各完成500次全100条更新，均按预算停止。每条在每个分支新增参与500次；两支不能合计为某一模型新增1000次。", '',
 f"相对A末尾，B的历史失败组Edge loss差值为{fe_delta:+.9g}，该组Edge FP差值为{edge_delta:+d}。"
 f"A联合严格成功{aa['joint_perfect']}/100，B为{bb['joint_perfect']}/100；"
 f"原72条保留分别为{len(A['retained_source72'])}、{len(B['retained_source72'])}，"
 f"新增身份分别为{len(A['new_over_source72'])}、{len(B['new_over_source72'])}。", '',
 verdict,'',
 '以上是本次有限预算的实测对照，不是全局根因证明。不能以B自己的加权loss较低判定获胜。'
 '如果只有困难组loss或FP改善而严格覆盖不增加，只能称局部收益；若丢失旧成功，必须与新增身份一起报告。', '',
 '## 协议与执行', '',
 '- 唯一干预：历史S71的Edge系数A=1、B=0.5。S的Face、F的Edge和Face系数均为1；每mesh始终除100，不按组大小或权重和归一化。000520始终属于F。',
 '- 同一原两块72条源模型和Adam分别恢复；第14/15块、最终LayerNorm、原Edge/Face head训练，其余冻结。末端LR=1e-5，两个head各1e-4，clip=1；μ、KL=0、logvar冻结、math00、fully-diff Soft4及750215个固定Face候选不变。',
 '- B实际重算加权目标梯度、clip、Adam一阶/二阶矩，不是用旧梯度分解表减去一行。',
 '- 两支每次更新均完整累积同一顺序的100条后统一更新。预算各500次；原三组Adam由1500到2000，第14块由500到1000。',
 '- 所有16个检查点均真实执行Encoder→μ→Decoder，并对100条核对冻结缓存路径的hidden、Edge/Face embedding逐位一致；全部实际Face从各自预测Edge图完整枚举。',
 '- 源checkpoint、旧74条模型、缓存和全部训练pool只读保留，末尾重新核对哈希。', '',
 '## A首步参考核验与恢复说明', '',
 '原点100条完整复现：Edge FP/FN=102302/1，实际Face FP/FN=5635/338，同一批72条联合成功。'
 'A首步原始梯度、clip及实际FP32位移与上一轮候选逐位一致。', '',
 '首步先后因参数/最小margin的严格逐位比较停下，现场和日志已保留。根因是有限位移参考额外执行了'
 'FP32(θ0 + FP32(θ1−θ0))；这一相减再相加不是可逆操作。115个微小参数与真实Adam θ1不同，最大差2.2737367544323206e-13。'
 '4条mesh的loss末位因此不同，四项汇总最大差约5.22e-10。实际Adam位移重新加回θ0后逐位等于有限位移参考；100条硬计数完全一致。', '',
 '两次均从已保存A第1步参数、Adam与RNG恢复，没有重跑首步、没有清空Adam；实际更新总数仍为A500+B500。'
 '复核标准与原始差值完整记录在resume_first_step.json及A_control/first_step_audit.json，未将这些差异归因于随机训练。', '',
 '| 首步量 | A原目标 | B成功Edge×0.5 |','|---|---:|---:|']
first={b:json.loads((R/b/'first_step_audit.json').read_text()) for b in cfg['branches']}
exact_first={b:json.loads((R/b/'first_step_exact_displacement.json').read_text()) for b in cfg['branches']}
for key in ['global_gradient_norm','clip_coefficient','actual_delta_norm','actual_relative_delta']:
    lines.append(f"| {key} | {first['A_control'][key]:.10g} | {first['B_success_edge_half'][key]:.10g} |")
lines+=['','| 首步分量 | A：g·实际Δθ | A：实测ΔL | B：g·实际Δθ | B：实测ΔL |','|---|---:|---:|---:|---:|']
for part in ['success_edge','success_face','failed_edge','failed_face']:
    ar=first['A_control'];br=first['B_success_edge_half']
    lines.append(f"| {part} | {exact_first['A_control']['gradient_dot_displacement'][part]:+.9g} | {ar['actual_component_changes'][part]:+.9g} | {exact_first['B_success_edge_half']['gradient_dot_displacement'][part]:+.9g} | {br['actual_component_changes'][part]:+.9g} |")
lines+=['','## 同预算完整重建轨迹','','| 新增step | A Edge FP/FN | B Edge FP/FN | A Face FP/FN | B Face FP/FN | A联合 | B联合 |','|---:|---:|---:|---:|---:|---:|---:|']
for step in cfg['checkpoints']:
    x=summary(data['A_control'][step]);y=summary(data['B_success_edge_half'][step])
    lines.append(f"| {step} | {x['edge']['fp']}/{x['edge']['fn']} | {y['edge']['fp']}/{y['edge']['fn']} | {x['face']['fp']}/{x['face']['fn']} | {y['face']['fp']}/{y['face']['fn']} | {x['joint_perfect']} | {y['joint_perfect']} |")
lines+=['','## 末尾四个未加权分量','','| 量 | A step500 | B step500 |','|---|---:|---:|']
for part in ['success_edge','success_face','failed_edge','failed_face']:
    lines.append(f"| {part} | {A['components'][part]:.10g} | {B['components'][part]:.10g} |")
lines += [f"| 共同原目标 | {A['original_objective']:.10g} | {B['original_objective']:.10g} |",
 f"| 各自优化目标（不可直接排名） | {A['optimized_objective']:.10g} | {B['optimized_objective']:.10g} |", '',
 '## 末尾固定分组结构计数','',
 '| 分支/组 | Edge FP/FN | 实际Face FP/FN | Edge成功 | 联合成功 |','|---|---:|---:|---:|---:|']
for name,e in [('A',A),('B',B)]:
    for group in ['all','S','F']:
        s=e['summary'][group]
        lines.append(f"| {name}/{group} | {s['edge']['fp']}/{s['edge']['fn']} | {s['face']['fp']}/{s['face']['fn']} | {s['edge_perfect']}/{s['meshes']} | {s['joint_perfect']}/{s['meshes']} |")
lines += ['',
 '## 成功身份与大mesh','',
 f"A原72条丢失：{', '.join(A['lost_source72']) or '无'}", '',
 f"B原72条丢失：{', '.join(B['lost_source72']) or '无'}", '',
 f"A新增：{', '.join(A['new_over_source72']) or '无'}", '',
 f"B新增：{', '.join(B['new_over_source72']) or '无'}", '',
 f"末尾仅B成功：{', '.join(comparison['B_only_joint_success']) or '无'}", '',
 f"末尾仅A成功：{', '.join(comparison['A_only_joint_success']) or '无'}", '',
 '大mesh按固定顶点数>1500定义，未按结果选择；完整表见paired_large_mesh.csv。', '',
 '| UID | 顶点 | A Edge FP/FN | B Edge FP/FN | A Face FP/FN | B Face FP/FN |','|---|---:|---:|---:|---:|---:|']
for x,y in zip(A['meshes'],B['meshes']):
    if x['vertices']<=1500:continue
    lines.append(f"| {x['uid']} | {x['vertices']} | {x['edge']['fp']}/{x['edge']['fn']} | {y['edge']['fp']}/{y['edge']['fn']} | {x['face']['fp']}/{x['face']['fn']} | {y['face']['fp']}/{y['face']['fn']} |")
lines+=['','## 文件与核验范围','',
 '- actual_trend.csv：四未加权分量、共同原目标、各自优化目标、成功身份数量及分组结构计数。',
 '- actual_per_mesh.csv、paired_per_mesh.csv、paired_large_mesh.csv：各检查点逐mesh与A/B对齐结果。',
 '- A_control、B_success_edge_half：完整500步updates.jsonl、501条训练前向trace、八个完整实际验收JSON、首步审计与恢复核验。',
 '- Predictions.zip：所有1600条实际预测NPZ，含顶点/GT、全部预测Edge ID和logit、实际Face候选全集ID/logit/GT label、GT覆盖标记。判正统一logit>0。',
 '- 独立CPU核对连续更新次数、同一UID顺序、每组Adam计数、所有checkpoint哈希，且从NPZ重新计算硬计数；用独立集合交集重数预测Edge图的全部三角形，确认枚举无遗漏。',
 '- 训练与全部16点真实网络前向确实执行；独立审计没有再次运行网络，只检查保存文件、模型状态与预测。',
 '- 完整模型、全部可续训checkpoint、首步完整梯度和实际位移保存在各自服务器目录，路径/大小/SHA见MODEL_ARTIFACTS.json。大模型/optimizer/cache不重复塞入评估包。',
 '- 原checkpoint仅保存Torch CPU/CUDA RNG。两支严格恢复这些字段，并使用相同捕获的Python/NumPy起始状态；后两者不参与确定性μ训练。新checkpoint保存了全部四类RNG。',
 '- 没有自动续训、改权重、换LR、刷新pool或扩大训练范围。']
(R/'REPORT.md').write_text('\n'.join(lines)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for branch,label in [('A_control','A original'),('B_success_edge_half','B success Edge x0.5')]:
    xs=cfg['checkpoints'];rows=[data[branch][s] for s in xs]
    for ax,ys,title in [(axes[0,0],[r['components']['failed_edge'] for r in rows],'Historical F Edge loss (unweighted)'),
            (axes[0,1],[summary(r)['edge']['fp'] for r in rows],'Actual Edge FP'),
            (axes[1,0],[summary(r)['face']['fp'] for r in rows],'Actual Face FP'),
            (axes[1,1],[summary(r)['joint_perfect'] for r in rows],'Joint strict success / 100')]:
        ax.plot(xs,ys,'o-',label=label);ax.set(title=title,xlabel='New full100 optimizer updates');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(R/'paired_trend.png',dpi=170);plt.close(fig)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
artifacts=[]
for branch in cfg['branches']:
    for p in sorted((R/branch).rglob('*.pt')):
        artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),branch=branch))
(R/'MODEL_ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
(R/'EXCLUDED_FILES.json').write_text(json.dumps(dict(model_and_optimizer_artifacts='MODEL_ARTIFACTS.json',
    source_full=dict(path=cfg['source_full'],sha256=cfg['source_full_sha256']),
    source_adam=dict(path=cfg['source_checkpoint'],sha256=cfg['source_checkpoint_sha256']),
    frozen_cache=dict(path=str(S/'cache'),manifest='source/cache_manifest.json'),
    training_face_pools=dict(path=str(S/'augmented_pools'),hashes='protocol_verification.json'),
    gradients_source=str(R.parent/'decoder_last2_gradient_groups_20260920/gradient_vectors.pt')),indent=2)+'\n')
entries=[]
for p in R.iterdir():
    if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.json','.md','.csv','.log','.txt','.diff','.png'] and p.name!='package_manifest.json':entries.append((p,p.name))
for branch in cfg['branches']:
    for p in (R/branch).rglob('*'):
        if p.is_file() and p.suffix in ['.json','.jsonl','.log','.md']:entries.append((p,str(p.relative_to(R))))
for root,prefix in [(R/'effective_code','effective_code'),(S/'runtime_dependencies','runtime_dependencies')]:
    for p in root.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:entries.append((p,prefix+'/'+str(p.relative_to(root))))
for name in ['READY.json','construction_args.json','overfit100_manifest.csv','data_manifest.csv','selection.json','pool_provenance.json']:
    entries.append((R/name,'source/'+name))
for name in ['cache/manifest.json','source_manifest.json','cache_gradient_verification.json']:
    entries.append((S/name,'source/'+name.replace('/','_')))
pred=[(p,str(p.relative_to(R))) for branch in cfg['branches'] for p in (R/branch/'run').glob('predictions-*/*.npz')]
assert len(pred)==1600
def package(name,items):
    path=R/name;manifest=[]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p,n in sorted(items,key=lambda t:t[1]):
            manifest.append(dict(path=n,bytes=p.stat().st_size,sha256=sha(p)));z.write(p,n)
        z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),files=len(items))
review=package('Nexus_SuccessEdgeHalf_AB500_Review_20260921.zip',entries)
print('REVIEW',review,flush=True)
predictions=package('Nexus_SuccessEdgeHalf_AB500_Predictions_20260921.zip',pred)
(R/'package_manifest.json').write_text(json.dumps(dict(review=review,predictions=predictions),indent=2)+'\n')
print('PACKAGES COMPLETE',flush=True)
