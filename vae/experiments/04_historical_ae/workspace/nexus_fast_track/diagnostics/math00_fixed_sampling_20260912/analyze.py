"""Report fixed-epsilon continuation and the paired held-out noise evaluation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads((ROOT/p).read_text())
done=read('complete.json');start=read('step0_acceptance.json')
trace=[json.loads(s) for s in (ROOT/'updates.jsonl').read_text().splitlines()]
assert [r['update'] for r in trace]==list(range(1,201))
check_steps=[0,1,10,20,50,100,200]
checks={mode:[read(f'{mode}_step{s:04d}.json') for s in check_steps] for mode in ['mu','sample']}
noises={s:[json.loads(t) for t in (ROOT/f'independent_step{s:04d}.jsonl').read_text().splitlines()] for s in [0,200]}
assert len(noises[0])==len(noises[200])==50
assert [r['seeds'] for r in noises[0]]==[r['seeds'] for r in noises[200]]
stats={}
for group in trace[0]['actual_updates']:
    update=np.array([r['actual_updates'][group]['relative_l2'] for r in trace]);grad=np.array([r['gradient_norms_preclip'][group] for r in trace])
    stats[group]=dict(nonzero_update_steps=int((update>0).sum()),relative_update_median=float(np.median(update)),gradient_norm_median=float(np.median(grad)),cumulative=done['cumulative_parameter_change'][group])
noise_stats={}
for step,rows in noises.items():
    noise_stats[step]=dict(simultaneous_perfect=sum(r['four_way_perfect'] for r in rows),total=50,by_mesh={})
    for i,label in enumerate(['small','large']):
        noise_stats[step]['by_mesh'][label]={}
        for kind in ['edge','face']:
            f1=np.array([r['rec'][i][kind]['f1'] for r in rows])
            noise_stats[step]['by_mesh'][label][kind]=dict(perfect_count=sum(r['rec'][i][kind]['fp']==r['rec'][i][kind]['fn']==0 for r in rows),f1_mean=float(f1.mean()),f1_min=float(f1.min()),f1_p5=float(np.quantile(f1,.05)),max_fp=max(r['rec'][i][kind]['fp'] for r in rows),max_fn=max(r['rec'][i][kind]['fn'] for r in rows))
clip=np.array([r['clip_coefficient'] for r in trace])
summary=dict(start=start,final=done['final'],group_stats=stats,noise_stats=noise_stats,perfect_mu_check_steps=done['perfect_mu_check_steps'],perfect_sample_check_steps=done['perfect_sample_check_steps'],clip=dict(min=float(clip.min()),median=float(np.median(clip)),max=float(clip.max()),clipped_updates=int((clip<1).sum())),training_seconds=done['training_seconds'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# math00_mu_step200 → 固定ε sampling，200更新','',
       '从上一轮math00 μ第200步权重和Adam出发。E/μ=1e-8，D/heads=1e-7，新增logvar组=1e-4；原四组Adam完整保留，logvar组从空状态开始。全网络可训练，KL=0，global clip=1，wd=0，只有sampling fully-diff Edge+Face Soft4参与优化。μ仅验收。',
       '', 'Backend00、candidate pool、归约和clamp[-20,10]未变；不重新初始化logvar。训练固定ε970000/970001，但每次重新计算当前μ、logvar和z。50组独立验收噪声预先固定，前后使用同一集合，每次均为两mesh packed forward。',
       '', '## Step0与最终实际重建','', '|新增更新|路径|mesh|Edge TP/FP/FN|Face TP/FP/FN|','|---:|---|---|---|---|']
def counts(row,kind):return '/'.join(str(row[kind][k]) for k in ['tp','fp','fn'])
for step in [0,200]:
    for mode in ['mu','sample']:
        for label,row in zip(['small','large'],checks[mode][0 if step==0 else -1]['rec']):lines.append(f'|{step}|{mode}|{label}|{counts(row,"edge")}|{counts(row,"face")}|')
lines+=['','## 全部完整检查点','', '|更新|μ loss|固定ε loss|μ四项全对|固定ε四项全对|','|---:|---:|---:|---|---|']
for mu,sample in zip(checks['mu'],checks['sample']):lines.append(f'|{mu["step"]}|{mu["loss"]:.10g}|{sample["loss"]:.10g}|{mu["four_way_perfect"]}|{sample["four_way_perfect"]}|')
lines+=['', 'μ与sampling是不同前向目标，不能用它们在同一步的loss差当作训练退化。前后趋势应分别与各自step0比较。', '', '## 同一50组独立噪声','', '|新增更新|两mesh Edge+Face同时严格全对|','|---:|---:|']
for step in [0,200]:lines.append(f'|{step}|{noise_stats[step]["simultaneous_perfect"]}/50|')
lines+=['', '|末尾独立噪声|严格全对次数|最大FP|最大FN|','|---|---:|---:|---:|']
for label in ['small','large']:
    for kind in ['edge','face']:
        ns=noise_stats[200]['by_mesh'][label][kind]
        lines.append(f'|{label} {kind}|{ns["perfect_count"]}/50|{ns["max_fp"]}|{ns["max_fn"]}|')
lines+=['','## 当前后验与实际扰动','', '|更新|mesh|σ median|σ p95|σ max|clamp下限占比|clamp上限占比|实际z−μ RMS|实际z−μ max abs|','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for step in [0,200]:
    sample=checks['sample'][0 if step==0 else -1]
    for label,p in zip(['small','large'],sample['posterior']):lines.append(f'|{step}|{label}|{p["std"]["median"]:.8g}|{p["std"]["p95"]:.8g}|{p["std"]["max"]:.8g}|{p["lower_clamp_fraction"]:.6g}|{p["upper_clamp_fraction"]:.6g}|{p["actual_noise_rms"]:.8g}|{p["actual_noise_max_abs"]:.8g}|')
lines+=['','## 更新与梯度','', '|组|非零更新步数|每步relative L2中位数|累计relative L2|梯度norm中位数（clip前）|','|---|---:|---:|---:|---:|']
for name,s in stats.items():lines.append(f'|{name}|{s["nonzero_update_steps"]}/200|{s["relative_update_median"]:.6g}|{s["cumulative"]["relative_l2"]:.6g}|{s["gradient_norm_median"]:.6g}|')
lines+=['',f'全局clip系数 min/median/max = {clip.min():.6g}/{np.median(clip):.6g}/{clip.max():.6g}。全部五组共同裁剪；{int((clip<1).sum())}/200步发生裁剪。', '', '## 判定', '', f'已完成200次固定ε更新。相同50组独立噪声的同时严格成功数为 {noise_stats[0]["simultaneous_perfect"]}/50 → {noise_stats[200]["simultaneous_perfect"]}/50。应将固定ε的训练验收与这个独立噪声结果分别判断，不能以μ或固定ε全对代替噪声集合验收。后验尾部σ的变化也必须与中位数一起看。', '', '本轮固定ε续训通过约定检查点验收，但独立噪声下严格重建的保持未通过。μ和固定ε在全部7个检查点都四项全对；μ loss在后半程回升，最终高于自身step0，不能因为μ硬指标全对而忽略这一变化。独立噪声错误主要集中在large Face，其F1仍很高，但严格全对次数明显减少。σ尾部扩大与独立噪声验收变差同时出现，本轮没有单独隔离σ与reconstruction mapping变化各自的贡献。', '', '这是固定ε目标下的200步预算，不能代替每步新ε训练稳定性。也未加入KL或额外μ loss。完整硬重建只在约定的7个点验收，不将中间未验收步计为严格成功。', '', '原始checkpoint、Adam、各次实际candidate/logit NPZ在服务器同名目录；本地保留全部统计JSON/JSONL及代码。math00_mu_step200.json记录独立成功基线的位置与SHA256。checkpoint_verification.json确认原四组Adam从200续至400，logvar新组从无状态训练至200。', '', '![曲线](curves.png)']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axes[0,0].plot(range(201),[r['loss_before_update'] for r in trace]+[checks['sample'][-1]['loss']],label='fixed epsilon objective')
axes[0,0].plot(check_steps,[r['loss'] for r in checks['mu']],'-o',label='mu eval only')
axes[0,0].set(title='Compare each path with its own step0',xlabel='New updates',yscale='log');axes[0,0].legend()
bars=axes[0,1].bar(['Before training','After 200 updates'],[noise_stats[s]['simultaneous_perfect'] for s in [0,200]],color=['#4477aa','#cc6677'])
axes[0,1].bar_label(bars,labels=[f'{noise_stats[s]["simultaneous_perfect"]}/50' for s in [0,200]],padding=4)
axes[0,1].set(title='Same held-out noise pairs: simultaneous perfect',ylabel='Both meshes, Edge + Face all correct',ylim=(0,55))
for i,label in enumerate(['small','large']):
    for q in ['median','p95']:
        vals=[checks['sample'][0]['posterior'][i]['std'][q]]+[r['posterior_after_update'][i]['std'][q] for r in trace]
        axes[1,0].plot(range(201),vals,label=f'{label} sigma {q}')
axes[1,0].axhline(np.exp(-10),color='grey',ls=':',label='clamp sigma floor');axes[1,0].set(title='Measured posterior sigma',xlabel='New updates',yscale='log');axes[1,0].legend()
for group in stats:axes[1,1].plot(range(1,201),[r['gradient_norms_preclip'][group] for r in trace],label=group)
axes[1,1].set(title='Group gradient norms before global clip',xlabel='New updates',yscale='log');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.25)
fig.savefig(ROOT/'curves.png',dpi=160);plt.close(fig)
print(json.dumps(dict(noise=noise_stats,group_stats=stats,clip=summary['clip'],sample_loss=[checks['sample'][0]['loss'],checks['sample'][-1]['loss']],mu_loss=[checks['mu'][0]['loss'],checks['mu'][-1]['loss']]),indent=2))
