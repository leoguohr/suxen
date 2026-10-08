"""Verify resumed training and distinguish reaching perfect edges from staying perfect."""
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'soft4'
OLD = ROOT.parent/'soft4_small_20260910/soft4'


def read(path):
    return json.loads(path.read_text())


prior = [json.loads(x) for x in (OLD/'trace.jsonl').read_text().splitlines()]
trace = [json.loads(x) for x in (OUT/'trace.jsonl').read_text().splitlines()]
complete, meta = read(OUT/'complete.json'), read(OUT/'provenance.json')
assert [r['step'] for r in prior] == list(range(1201))
assert [r['step'] for r in trace] == list(range(1200,3001))
assert complete['final'] == trace[-1] and complete['optimizer_final_step'] == 3000
assert meta['optimizer_state_restored_exactly'] and meta['additional_updates'] == 1800
assert meta['script_sha256'] == hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert meta['soft4_tau'] == 1 and meta['soft4_epsilon'] == 1e-8
first, count, streak, longest = None, 0, 0, 0
for i,x in enumerate(trace):
    r=x['rows'][0]
    assert len(x['rows']) == 1 and r['tp']+r['fn'] == 1152 and r['tn']+r['fp'] == 73153
    assert r['edge_f1'] == 2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
    assert x['is_perfect'] == r['is_perfect'] == (r['fp'] == r['fn'] == 0)
    assert x['encoder_lr'] == 1e-5 and x['decoder_lr'] == 1e-4
    assert x['rng_unchanged'] and x['frozen_unchanged']
    assert x['objective'] == r['soft4_loss']
    if x['is_perfect'] and first is None: first=x['step']
    count += int(x['is_perfect'])
    streak=streak+1 if x['is_perfect'] else 0
    longest=max(longest,streak)
    assert (x['first_perfect_step'],x['perfect_evaluations'],x['perfect_streak'],x['longest_perfect_streak']) == (first,count,streak,longest)
    if i:
        assert np.isclose(r['mu_relative_delta'],r['mu_delta_l2']/(trace[i-1]['rows'][0]['mu_l2']+1e-12))
assert complete['first_perfect_step'] == first and complete['perfect_evaluations'] == count
assert not any(r['rows'][0]['fp']==r['rows'][0]['fn']==0 for r in prior)
joined=prior[:1200]+trace  # One step1200 entry, from the verified resume forward.
assert [r['step'] for r in joined] == list(range(3001))
perfect_steps=[x['step'] for x in trace if x['is_perfect']]
intervals=[]
for s in perfect_steps:
    if intervals and s == intervals[-1][1]+1: intervals[-1][1]=s
    else: intervals.append([s,s])
windows={}
for n in [200,500]:
    rows=[x['rows'][0] for x in trace[-n:]]
    windows[str(n)]=dict(step_from=3001-n, step_to=3000,
        f1_min_median_max=np.quantile([r['edge_f1'] for r in rows],[0,.5,1]).tolist(),
        perfect_count=sum(r['is_perfect'] for r in rows),
        fp_min_median_max=np.quantile([r['fp'] for r in rows],[0,.5,1]).tolist(),
        fn_min_median_max=np.quantile([r['fn'] for r in rows],[0,.5,1]).tolist())
result=dict(final=trace[-1], first_perfect_step=first, perfect_count=count, longest_perfect_streak=longest,
    perfect_intervals=intervals, tail_windows=windows, resume_verification=read(OUT/'resume_verification.json'),
    seconds=complete['seconds'], additional_updates=1800, config_and_resume_checks_passed=True)
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (ROOT/'combined_trace.jsonl').open('w') as f:
    for x in joined:
        r=x['rows'][0]
        f.write(json.dumps(dict(x,is_perfect=(r['fp']==r['fn']==0)))+'\n')

fig,axes=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
steps=[x['step'] for x in joined]
axes[0].plot(steps,[x['rows'][0]['edge_f1']*100 for x in joined],color='#178D78',lw=.8)
if perfect_steps:
    axes[0].scatter(perfect_steps,[100]*len(perfect_steps),s=5,color='#E19C27',label='FP=FN=0')
    axes[0].legend()
axes[0].set(ylabel='Hard Edge F1 (%)',ylim=(100*min(x['rows'][0]['edge_f1'] for x in trace)-.5,100.3))
for key,color in [('fp','#C23B43'),('fn','#476FBB')]:
    axes[1].plot(steps,[x['rows'][0][key] for x in joined],label=key.upper(),color=color,lw=.8)
axes[1].set(ylabel='Wrong / missing edges');axes[1].set_yscale('symlog',linthresh=1);axes[1].legend()
axes[1].set_ylim(-.1,1.3*max(max(x['rows'][0]['fp'],x['rows'][0]['fn']) for x in trace))
axes[2].plot(steps,[x['objective'] for x in joined],label='Soft4 objective',color='#178D78',lw=.8)
axes[2].plot(steps,[x['rows'][0]['balanced_loss'] for x in joined],label='GT-balanced BCE',color='#7060B0',lw=.8)
axes[2].set(ylabel='Loss',xlabel='Completed updates',yscale='log');axes[2].legend()
for ax in axes:
    ax.set_xlim(1200,3000)
    ax.axvline(1200,color='#888888',ls='--',lw=1)
    ax.grid(alpha=.2)
fig.suptitle('Small-only Soft4: resume model + Adam from 1,200 to 3,000 updates\nUnchanged E LR=1e-5, D LR=1e-4 | tau=1, detached membership | hard threshold=0')
fig.savefig(ROOT/'training_curve.png',dpi=160);plt.close(fig)
f=trace[-1]['rows'][0]
lines=['# Small-only Soft4：原配置续训到3000步','',
       '恢复step1200模型及Adam全部状态（step、exp_avg、exp_avg_sq、参数组），新增1800次更新。E LR=1e-5、D LR=1e-4、τ=1、detach membership、FP32组归约、ε=1e-8、μ mode、Face=KL=wd=0、clip=1，后端不变。', '',
       f"最终：TP={f['tp']}、FP={f['fp']}、FN={f['fn']}、TN={f['tn']}；F1={100*f['edge_f1']:.6f}%；Soft4={f['soft4_loss']:.8g}，GT-balanced BCE={f['balanced_loss']:.8g}。", '',
       f'首次严格100%步数：{first}；续训评估中严格100%次数：{count}；最长连续严格100%：{longest}步。没有提前停止或调整LR。','',
       '|末尾窗口|F1 最小/中位/最大|FP=FN=0次数|','|---|---|---:|']
for n,w in windows.items():
    lines.append(f"|末{n}步|{' / '.join(f'{100*v:.5f}%' for v in w['f1_min_median_max'])}|{w['perfect_count']}/{n}|")
lines += ['', '首次出现100%只说明当前配置到达过完全重建，是否稳定需要结合连续保持长度和末尾窗口。完整100%区间保存在summary.json。', '',
          '恢复时模型参数、Adam状态逐张量相等；step1200重新前向与旧评估的差异记录在soft4/resume_verification.json。沿用Flash/CUDA后端，不声称逐位数值确定性。', '',
          '服务器每200步保存包含Adam状态的checkpoint；若出现首次100%，另保存该步checkpoint与embedding。原始step1200和旧日志保留。combined_trace.jsonl按0–3000对齐，step1200采用恢复后的前向记录。','',
          '![训练曲线](training_curve.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in result.items() if k not in ['resume_verification','perfect_intervals']},indent=2))
