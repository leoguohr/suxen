"""Report existing full reconstruction evaluations; no training or forward."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
done=json.loads((r/'completion.json').read_text())
assert done['completed_updates']==4500 and done['additional_updates']==1000
logs=[json.loads(x) for x in (r/'updates.jsonl').read_text().splitlines()]
assert [x['update'] for x in logs]==list(range(1,1001))
checks=[json.loads(p.read_text()) for p in sorted(r.glob('eval-update*.json'))]
assert [x['update'] for x in checks]==[0,200,400,600,800,1000]
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,key in zip(axes[0],['edge_soft4','face_soft4']):
    ax.plot([x['cumulative_update']-1 for x in logs],[x[key] for x in logs],lw=.8)
    ax.set_title(key+' (before update)')
for ax,key in zip(axes[1],['edge','face']):
    for kind in ['fp','fn']:
        ax.plot([x['cumulative_update'] for x in checks],[x[key][kind] for x in checks],marker='o',label=kind.upper())
    ax.set_title('Actual '+key+' errors');ax.legend()
for ax in axes.flat:
    ax.set_xlabel('Cumulative updates');ax.grid(alpha=.2)
fig.suptitle('804 / latent512 / B continuation / unchanged LR / mu / fully-diff Soft4')
fig.savefig(r/'training_curves.png',dpi=170);plt.close(fig)
summary=dict(checks=checks,perfect_checks=[x['update'] for x in checks if x['perfect']],
             last_three_checks_perfect=done['last_three_checks_perfect'],
             median_update_seconds=float(np.median([x['seconds'] for x in logs])),
             clipped_updates=sum(x['clip_coefficient']<1 for x in logs))
(r/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 512维 B分支续训1000步：累计3500至4500','',
       '从B step500完整权重、四组Adam和RNG继续。Encoder/μ LR=3e-6，Decoder/两head LR=3e-5；保持μ路径、KL=0、logvar冻结、math00、fully-diff Soft4、原候选与评分、clip=1。无新warmup或LR缩放。','',
       '| 新增步数 | 累计步数 | Edge TP/FP/FN | 实际Face TP/FP/FN | GT Face候选覆盖 | Edge/Face Soft4 |',
       '| --- | --- | --- | --- | --- | --- |']
for x in checks:
    counts=lambda k:'/'.join(str(x[k][v]) for v in ['tp','fp','fn'])
    lines.append(f'| {x["update"]} | {x["cumulative_update"]} | {counts("edge")} | {counts("face")} | {x["gt_face_candidates"]}/1604 | {x["parts"][0]["edge"]:.6f}/{x["parts"][0]["face"]:.6f} |')
lines+=['','Face从当前预测Edge图完整枚举；GT Face未进入候选仍计FN。训练pool准确率不能替代实际重建。',
        '',f'严格成功检查点（新增步数）：{summary["perfect_checks"]}；末三个检查点全部严格成功：{summary["last_three_checks_perfect"]}。',
        f'普通更新耗时中位数：{summary["median_update_seconds"]:.3f}秒；发生裁剪：{summary["clipped_updates"]}/1000步。',
        '恰好完成1000次新增更新并停止，未自动延长。起点/Adam/RNG/固定logvar及各组实际更新核验见verification.json。',
        '', '![训练曲线](training_curves.png)']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
