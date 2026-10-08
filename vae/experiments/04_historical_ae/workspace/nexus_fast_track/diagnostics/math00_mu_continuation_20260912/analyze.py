"""Summarize the bounded 200-update math00 recovery, without extending it."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
accept=read('step0_acceptance.json');done=read('complete.json')
updates=[json.loads(line) for line in (ROOT/'updates.jsonl').read_text().splitlines()]
checks=[read(f'mu_step{step:04d}.json') for step in [0,1,10,20,50,100,200]]
assert [r['update'] for r in updates]==list(range(1,201))
assert done['updates']==200 and done['logvar_unchanged']
group_stats={}
for name in updates[0]['actual_updates']:
    changes=np.array([r['actual_updates'][name]['relative_l2'] for r in updates])
    grad=np.array([r['gradient_norms_preclip'][name] for r in updates])
    group_stats[name]=dict(nonzero_update_steps=int((changes>0).sum()),relative_update_min=float(changes.min()),relative_update_median=float(np.median(changes)),relative_update_max=float(changes.max()),gradient_norm_median=float(np.median(grad)),cumulative=done['cumulative_parameter_change'][name])
clip=np.array([r['clip_coefficient'] for r in updates])
summary=dict(stage=done['stage'],initial=checks[0],final=checks[-1],group_stats=group_stats,clip=dict(min=float(clip.min()),median=float(np.median(clip)),max=float(clip.max()),clipped_steps=int((clip<1).sum())),perfect_evaluation_steps=done['perfect_evaluation_steps'],seconds=done['seconds'],fixed_epsilon_step0=accept['fixed_epsilon'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# B16100 → math00：200步 μ 恢复','',
       '同一 warm start，Encoder/μ=1e-8，Decoder/heads=1e-7，fresh Adam 默认 betas/eps，wd=0，global clip=1。logvar冻结，KL=0，fully-diff Edge/Face Soft4、原candidate pool及归约不变。实际Face从各评估点预测Edge图枚举。',
       '', '## Step0：TP / FP / FN','', '|模式|mesh|Edge|Face|','|---|---|---|---|']
def count(r,kind):return '/'.join(str(r[kind][k]) for k in ['tp','fp','fn'])
for mode,key in [('mu','mu'),('fixed epsilon','fixed_epsilon')]:
    for label,r in zip(['small','large'],accept[key]['rec']):lines.append(f'|{mode}|{label}|{count(r,"edge")}|{count(r,"face")}|')
lines+=['','μ尚未四项全对，因此本轮为恢复训练。固定ε baseline 与既有00的全部保存数组、loss、四项loss精确匹配；μ step0重复forward完全一致。', '', '## μ 检查点','', '|更新次数|总loss|Small Edge FP/FN|Small Face FP/FN|Large Edge FP/FN|Large Face FP/FN|','|---:|---:|---|---|---|---|']
for r in checks:
    cells=[f'{m[k]["fp"]}/{m[k]["fn"]}' for m in r['rec'] for k in ['edge','face']]
    lines.append(f'|{r["step"]}|{r["loss"]:.10g}|'+'|'.join(cells)+'|')
lines+=['','## 实际更新','', '|参数组|非零更新步数|每步relative L2中位数|200步累计relative L2|','|---|---:|---:|---:|']
for name,s in group_stats.items():lines.append(f'|{name}|{s["nonzero_update_steps"]}/200|{s["relative_update_median"]:.6g}|{s["cumulative"]["relative_l2"]:.6g}|')
lines+=['',f'Clip系数 min/median/max：{clip.min():.6g} / {np.median(clip):.6g} / {clip.max():.6g}。',f'完整验收四项全对的已保存更新点：{done["perfect_evaluation_steps"]}。只在约定的7个点完整枚举Face，不能把它们之间的训练步都算作已验收。',f'200次更新及检查用时 {done["seconds"]:.1f} 秒。', '', '四项loss和各组梯度/实际更新量逐步记录在updates.jsonl；硬重建和margin记录在mu_stepXXXX.json。服务器同名目录保留step0/200完整权重及Adam和检查点logit NPZ，本地取回统计日志。没有自动延长训练或进入sampling阶段。', '', '## 本轮判断', '', '00路径已通过这次有限预算的μ恢复验收：正常E+D+heads联合更新可修复起点错误，并在后续约定检查点保持四项全对。Encoder全部200步有非零FP32更新，排除了仅凭权重未动而保持成功的情况。这里建立的是00数值路径自己的μ成功工作点；尚未验证当前工作点的sampling训练或KL，不把结果外推为全VAE训练问题已经解决。', '', '![训练曲线](curves.png)']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
x=np.arange(201)
for i,label in enumerate(['small','large']):
    for kind in ['edge','face']:
        vals=[r['parts_before_update'][i][kind] for r in updates]+[checks[-1]['parts'][i][kind]]
        axes[0,0].plot(x,vals,label=f'{label} {kind}')
axes[0,0].set(title='Fully-differentiable Soft4 parts',xlabel='Completed updates',yscale='log');axes[0,0].legend()
for i,label in enumerate(['small','large']):
    for kind in ['edge','face']:
        axes[0,1].plot([r['step'] for r in checks],[r['rec'][i][kind]['fp']+r['rec'][i][kind]['fn'] for r in checks],'-o',label=f'{label} {kind}')
axes[0,1].set(title='Actual reconstruction errors at checks',xlabel='Completed updates',ylabel='FP + FN');axes[0,1].legend()
for name in group_stats:
    axes[1,0].plot(np.arange(1,201),[r['actual_updates'][name]['relative_l2'] for r in updates],label=name)
axes[1,0].set(title='Actual FP32 parameter displacement',xlabel='Update',yscale='log',ylabel='Relative L2');axes[1,0].legend()
axes[1,1].plot(np.arange(1,201),clip);axes[1,1].set(title='Global gradient clipping coefficient',xlabel='Update',yscale='log')
for ax in axes.flat:ax.grid(alpha=.25)
fig.savefig(ROOT/'curves.png',dpi=160);plt.close(fig)
print(json.dumps(dict(loss0=checks[0]['loss'],loss200=checks[-1]['loss'],perfect_checks=done['perfect_evaluation_steps'],group_stats=group_stats,clip=summary['clip']),indent=2))
