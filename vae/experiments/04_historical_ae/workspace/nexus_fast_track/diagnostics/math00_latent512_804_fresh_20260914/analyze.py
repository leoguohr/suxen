"""Read saved logs only; never load a network or perform optimizer updates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
status=json.loads((r/'status.json').read_text())
raw=(r/'updates.jsonl').read_text()
partial_copy=not raw.endswith('\n')
if partial_copy:
    assert status['state']!='completed', 'Incomplete final log transfer'
    raw=raw.rsplit('\n',1)[0]
updates=[json.loads(x) for x in raw.splitlines() if x.strip()]
evaluations=[json.loads(p.read_text()) for p in sorted(r.glob('eval-update*.json'))]
steps=np.array([x['update'] for x in updates])
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for key,label in [('edge_soft4','Edge'),('face_soft4','Face')]:
    axes[0,0].plot(steps,[x[key] for x in updates],label=label,lw=1)
axes[0,0].set(title='Fully differentiable Soft4 (before update)',ylabel='Loss')
es=[x['update'] for x in evaluations]
for kind in ['edge','face']:
    for metric in ['fp','fn']:
        axes[0,1].plot(es,[x[kind][metric] for x in evaluations],marker='o',label=f'{kind} {metric}')
axes[0,1].set(yscale='symlog',title='Actual full reconstruction (after update)',ylabel='Errors; symlog scale')
axes[1,0].plot(es,[x['gt_face_candidates']/1604 for x in evaluations],marker='o',label='GT face candidate coverage')
for kind in ['edge','face']:
    axes[1,0].plot(es,[x[kind]['f1'] for x in evaluations],marker='o',label=f'{kind} F1')
axes[1,0].set(ylim=(-.03,1.03),title='Full predicted-graph acceptance',ylabel='Fraction')
for kind in ['mu','hidden','edge','face']:
    axes[1,1].plot(steps,[x['scales'][kind]['rms'] for x in updates],label=kind,lw=1)
axes[1,1].set(yscale='log',title='Representation RMS',ylabel='RMS')
for ax in axes.flat:
    ax.set_xlabel('Actual optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle(f'Fresh latent512 / one full804 mesh / math00 / mu only — {len(updates)} updates')
fig.savefig(r/'training_curves.png',dpi=180);plt.close(fig)
summary=dict(status=status['state'],recorded_updates=len(updates),partial_running_copy=partial_copy,contiguous_updates=steps.tolist()==list(range(1,len(updates)+1)),
             evaluations=evaluations,all512_channels_have_gradient=all(x['interface']['mu_gradient_nonzero_channels']==512 for x in updates),
             nonzero_update_counts={g:sum(x['actual_updates'][g]['delta_l2']>0 for x in updates) for g in updates[0]['actual_updates']},
             clipped_updates=sum(x['clip_coefficient']<1 for x in updates),
             median_update_seconds=float(np.median([x['seconds'] for x in updates])),
             latest_loss=dict(edge=updates[-1]['edge_soft4'],face=updates[-1]['face_soft4']),
             strict_perfect_checkpoints=[x['update'] for x in evaluations if x['perfect']])
(r/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# fresh latent512：804点完整重建实验','',f'状态：{status["state"]}；已记录{len(updates)}次实际更新。',
       '', '本轮同时改变latent、初始化和训练设置，不能把结果唯一归因于512维。',
       '', 'μ重建；sampling/KL关闭；latent512、Decoder hidden1024、评分表示各32；fresh Adam；前100步warmup，随后E/μ=1e-5、D/heads=1e-4。',
       '', '| 更新数 | Edge TP/FP/FN | 实际Face TP/FP/FN | GT Face候选覆盖 | Edge/Face Soft4 | 严格全对 |',
       '| --- | --- | --- | --- | --- | --- |']
for x in evaluations:
    counts=lambda k:'/'.join(str(x[k][v]) for v in ['tp','fp','fn'])
    lines.append(f'| {x["update"]} | {counts("edge")} | {counts("face")} | {x["gt_face_candidates"]}/1604 | {x["parts"][0]["edge"]:.6g}/{x["parts"][0]["face"]:.6g} | {x["perfect"]} |')
lines += ['', 'Face从当次预测Edge图完整枚举，缺失GT候选仍计FN。每步Face训练pool计数不能替代上述实际Face计数。',
          '',f'普通更新耗时中位数：{summary["median_update_seconds"]:.3f}秒；发生裁剪：{summary["clipped_updates"]}/{len(updates)}步。',
          f'所有更新512个μ通道均有梯度：{summary["all512_channels_have_gradient"]}；四组非零更新次数：{summary["nonzero_update_counts"]}。',
          '', '完整原始日志：updates.jsonl；实际验收：eval-update*.json；运行配置与源码：manifest.json、train.py、source_archive/。',
          '', '![训练曲线](training_curves.png)']
if status['state']=='completed':
    f=evaluations[-1]
    lines += ['', '本轮结论：3000次更新已完成并停止；预定检查点没有严格全对，未通过后期保持验收。',
              f'最终Face FN={f["face"]["fn"]}，其中{f["missing_gt_face_candidates"]}个因缺边未进入实际候选，另有{f["face"]["fn"]-f["missing_gt_face_candidates"]}个进入候选但判负。',
              '2500到3000步，漏边、漏面和多余边/面均减少；因此存在实际结构学习，但仍有大量FP，不能用loss下降或高召回替代严格重建通过。',
              '本轮未证明512维新建构流程已跑通，也不能据此证明512维表达不了该mesh；不自动增加预算。',
              '', 'verification.json核验了3000条连续日志、全部指定checkpoint哈希、Adam步数与LR调度、logvar专属权重不变，以及启动日志修正前后相同种子的初始权重一致。',
              'loss表使用对应checkpoint更新后的重新forward；updates.jsonl的第3000条loss是第3000次更新之前，两者参数点不同。',
              '', '完整checkpoint仍保存在服务器实验目录；本地日志包不包含大型checkpoint。']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='evaluations'},indent=2))
