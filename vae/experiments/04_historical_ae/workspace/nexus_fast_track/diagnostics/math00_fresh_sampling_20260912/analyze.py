"""Finite-budget acceptance using common monitoring noise and held-back final noise."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
done=read('complete.json');manifest=read('manifest.json')
trace=[json.loads(s) for s in (ROOT/'updates.jsonl').read_text().splitlines()]
assert len(trace)==200 and [r['update'] for r in trace]==list(range(1,201))
checks=[0,1,10,20,50,100,200];monitor=[0,20,50,100,200]
mu=[read(f'mu_step{s:04d}.json') for s in checks]
fixed=[read(f'sample_step{s:04d}.json') for s in checks]
noise_stats={}
for step in monitor:
    values=[json.loads(s) for s in (ROOT/f'independent_step{step:04d}.jsonl').read_text().splitlines()]
    assert len(values)==50 and [x['seeds'] for x in values]==manifest['evaluation_seeds']
    summary=read(f'independent_summary_step{step:04d}.json')
    assert summary['simultaneous_perfect']==sum(all(r[k]['fp']==r[k]['fn']==0 for r in v['rec'] for k in ['edge','face']) for v in values)
    summary['errors']={}
    for i,label in enumerate(['small','large']):
        for kind in ['edge','face']:
            errors=np.array([[v['rec'][i][kind]['fp'],v['rec'][i][kind]['fn']] for v in values])
            unique,counts=np.unique(errors,axis=0,return_counts=True)
            summary['errors'][label+'_'+kind]=dict(histogram=[dict(fp=int(e[0]),fn=int(e[1]),count=int(n)) for e,n in zip(unique,counts)],mean_fp=float(errors[:,0].mean()),mean_fn=float(errors[:,1].mean()),max_fp=int(errors[:,0].max()),max_fn=int(errors[:,1].max()))
    noise_stats[step]=summary
group_stats={}
for group in manifest['groups']:
    updates=np.array([r['actual_updates'][group]['relative_l2'] for r in trace])
    grads=np.array([r['gradient_norms_preclip'][group] for r in trace])
    group_stats[group]=dict(nonzero_updates=int((updates>0).sum()),relative_update_median=float(np.median(updates)),gradient_norm_median=float(np.median(grads)),cumulative=done['cumulative_parameter_change'][group])
assert len({h for r in trace for h in r['epsilon_sha256']})==400
clip=np.array([r['clip_coefficient'] for r in trace])
control={}
for step in [0,200]:
    values=[json.loads(s) for s in (ROOT.parent/'math00_fixed_sampling_20260912'/f'independent_step{step:04d}.jsonl').read_text().splitlines()]
    assert [x['seeds'] for x in values]==manifest['evaluation_seeds']
    control[step]=dict(simultaneous_perfect=sum(v['four_way_perfect'] for v in values),mean_soft4=float(np.mean([v['loss'] for v in values])))
summary=dict(acceptance=done['acceptance'],monitoring=noise_stats,previous_fixed_control=control,final_unseen=done['final_unseen'],groups=group_stats,clip=dict(min=float(clip.min()),median=float(np.median(clip)),clipped_updates=int((clip<1).sum())),mu_loss=[mu[0]['loss'],mu[-1]['loss']],fixed_diagnostic_loss=[fixed[0]['loss'],fixed[-1]['loss']],posterior_initial=fixed[0]['posterior'],posterior_final=fixed[-1]['posterior'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# math00：每步新ε，200次更新','',
       '起点为math00_mu_step200完整权重及对应Adam。原四组状态从200继续；logvar组从空状态开始。E/μ=1e-8，D/heads=1e-7，logvar=1e-4。全网络sampling fully-diff Edge+Face Soft4，KL=0，global clip=1，wd=0；backend00、clamp[-20,10]、candidate和归约不变。',
       '', '唯一训练变量是固定ε改为持续前进的独立CUDA RNG。每次更新前各mesh只生成一份ε，backward/recompute和验收不推进训练RNG；200步共400份独立张量均已保存。μ和原固定ε仅作诊断，不参与训练loss。',
       '', '## 主验收：同一50组监控噪声','', '|新增更新|同时四项全对|Soft4均值|Soft4标准差|','|---:|---:|---:|---:|']
for step,s in noise_stats.items():lines.append(f'|{step}|{s["simultaneous_perfect"]}/50|{s["mean_soft4"]:.10g}|{s["std_soft4"]:.6g}|')
lines+=['','同一成功起点与同一监控集合，对照如下：','', '|训练ε方式|step0同时全对|step200同时全对|step200验收Soft4均值|','|---|---:|---:|---:|',f'|上一轮固定ε|{control[0]["simultaneous_perfect"]}/50|{control[200]["simultaneous_perfect"]}/50|{control[200]["mean_soft4"]:.10g}|',f'|本轮每步新ε|{noise_stats[0]["simultaneous_perfect"]}/50|{noise_stats[200]["simultaneous_perfect"]}/50|{noise_stats[200]["mean_soft4"]:.10g}|']
lines+=['','## μ与原固定ε诊断','', '|更新|μ全对|原固定ε全对|μ loss|原固定ε loss|','|---:|---|---|---:|---:|']
for a,b in zip(mu,fixed):lines.append(f'|{a["step"]}|{a["four_way_perfect"]}|{b["four_way_perfect"]}|{a["loss"]:.10g}|{b["loss"]:.10g}|')
lines+=['','## 最终实际重建：TP/FP/FN','', '|路径|mesh|Edge|Face|','|---|---|---|---|']
for label,d in [('mu',mu[-1]),('fixed diagnostic',fixed[-1])]:
    for mesh,row in zip(['small','large'],d['rec']):
        counts=['/'.join(str(row[k][v]) for v in ['tp','fp','fn']) for k in ['edge','face']]
        lines.append(f'|{label}|{mesh}|{counts[0]}|{counts[1]}|')
lines+=['','## 后验与非零扰动','', '|更新|mesh|σ median|σ p95|σ max|clamp下限占比|clamp上限占比|固定诊断ε实际扰动RMS|','|---:|---|---:|---:|---:|---:|---:|---:|']
for d in [fixed[0],fixed[-1]]:
    for label,p in zip(['small','large'],d['posterior']):lines.append(f'|{d["step"]}|{label}|{p["std"]["median"]:.8g}|{p["std"]["p95"]:.8g}|{p["std"]["max"]:.8g}|{p["lower_clamp_fraction"]:.6g}|{p["upper_clamp_fraction"]:.6g}|{p["actual_noise_rms"]:.8g}|')
lines+=['','## 参数实际更新','', '|组|非零更新次数|relative L2中位数|累计relative L2|clip前梯度norm中位数|','|---|---:|---:|---:|---:|']
for name,s in group_stats.items():lines.append(f'|{name}|{s["nonzero_updates"]}/200|{s["relative_update_median"]:.6g}|{s["cumulative"]["relative_l2"]:.6g}|{s["gradient_norm_median"]:.6g}|')
lines+=['',f'裁剪发生 {int((clip<1).sum())}/200 步；系数 min/median={clip.min():.6g}/{np.median(clip):.6g}。', '', '## 预先规定的判定', '', f'有限预算通过：**{done["acceptance"]["budget_pass"]}**。判据为step50/100/200 μ严格全对、相同监控集合50/50，且五组持续有非零实际更新、采样扰动非零。']
if done['final_unseen'] is not None:
    lines+=['',f'因此执行额外50组此前未参与训练或监控的噪声：同时严格全对 **{done["final_unseen"]["simultaneous_perfect"]}/50**，Soft4均值 {done["final_unseen"]["mean_soft4"]:.10g}。']
else:lines+=['','未执行额外50组：'+done['final_unseen_skipped_reason']]
lines+=['', '这是两mesh、当前小σ和KL=0下的有限预算结果，不构成对全部高斯噪声的保证。监控集合没有用于反向传播、改LR、选择训练ε或提前终止。误差分布见summary.json的monitoring/errors及逐组JSONL。', '', '服务器同名目录保存checkpoint、完整Adam、训练RNG状态和200步实际ε；本地保存代码、完整统计与图。监控集省略NPZ序列化，其余metric代码逐字复用；step0逐组结果与固定ε对照完全相同。evaluation_seeds.json中的旧training字段仅保存原固定ε诊断值；实际训练使用独立的training_rng_seed=2026091203并持续前进，详见manifest及每步RNG记录。', '', '![曲线](curves.png)']
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axes[0,0].plot(range(200),[r['loss_before_update'] for r in trace],alpha=.45,label='new training epsilon each update')
axes[0,0].errorbar(monitor,[noise_stats[s]['mean_soft4'] for s in monitor],yerr=[noise_stats[s]['std_soft4'] for s in monitor],fmt='o-',label='same 50 eval noise pairs')
axes[0,0].set(title='Sampling Soft4: compare monitored expectation',xlabel='Completed updates');axes[0,0].legend()
axes[0,1].plot(monitor,[noise_stats[s]['simultaneous_perfect'] for s in monitor],'-o',label='fresh-epsilon training')
axes[0,1].plot([0,200],[control[s]['simultaneous_perfect'] for s in [0,200]],'x',markersize=9,label='fixed-epsilon control (0/200 only)')
axes[0,1].set(title='Same monitoring set: simultaneous perfect',xlabel='Completed updates',ylabel='Pairs out of 50',ylim=(-1,53));axes[0,1].legend()
posterior=[r['posterior_before_update'] for r in trace]+[fixed[-1]['posterior']]
for i,label in enumerate(['small','large']):
    for q in ['median','p95']:axes[1,0].plot(range(201),[r[i]['std'][q] for r in posterior],label=f'{label} sigma {q}')
axes[1,0].axhline(np.exp(-10),color='grey',ls=':',label='sigma clamp floor');axes[1,0].set(title='Measured posterior',xlabel='Completed updates',yscale='log');axes[1,0].legend()
for name in group_stats:axes[1,1].plot(range(1,201),[r['actual_updates'][name]['relative_l2'] for r in trace],label=name)
axes[1,1].set(title='Actual FP32 parameter updates',xlabel='Update',yscale='log',ylabel='Relative L2');axes[1,1].legend()
for ax in axes.flat:ax.grid(alpha=.25)
fig.savefig(ROOT/'curves.png',dpi=160);plt.close(fig)
print(json.dumps(dict(acceptance=done['acceptance'],monitoring={s:(r['simultaneous_perfect'],r['mean_soft4']) for s,r in noise_stats.items()},final_unseen=done['final_unseen'],clip=summary['clip'],groups=group_stats),indent=2))
