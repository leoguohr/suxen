"""Large-only Soft4: verify all 3000 updates and summarize strict edge recovery."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'soft4'


def read(path):
    return json.loads(path.read_text())


trace = [json.loads(x) for x in (OUT/'trace.jsonl').read_text().splitlines()]
complete, meta = read(OUT/'complete.json'), read(OUT/'provenance.json')
small_meta=read(ROOT.parent/'soft4_small_20260910/soft4/provenance.json')
assert [r['step'] for r in trace] == list(range(3001))
assert complete['final'] == trace[-1] and complete['optimizer_final_step'] == 3000
assert meta['initial_weights_verified_tensor_equal'] and meta['fresh_adam_verified_empty']
assert meta['initial_sha256'] == small_meta['initial_sha256']
assert meta['selected_uids'] == ['nexus_2k_001849']
assert meta['script_sha256'] == hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
for key in ['encoder_lr','decoder_lr','soft4_tau','soft4_epsilon','soft4_membership','soft4_group_reduction','threshold','mode','optimizer']:
    assert meta[key] == small_meta[key], key
first, count, streak, longest = None, 0, 0, 0
for i,x in enumerate(trace):
    r=x['rows'][0]
    assert len(x['rows']) == 1 and r['tp']+r['fn'] == 7719 and r['tn']+r['fp'] == 3306306
    assert r['edge_f1'] == 2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
    assert x['is_perfect'] == r['is_perfect'] == (r['fp'] == r['fn'] == 0)
    assert x['encoder_lr'] == 1e-5 and x['decoder_lr'] == 1e-4
    assert x['rng_unchanged'] and x['frozen_unchanged'] and x['objective'] == r['soft4_loss']
    assert np.isclose(r['soft4_loss'],np.mean([g['mean'] for g in r['soft4_groups'].values()]))
    for g in r['soft4_groups'].values():
        assert np.isclose(g['mean'],g['numerator']/(g['mass']+1e-8),rtol=2e-6)
    if x['is_perfect'] and first is None: first=x['step']
    count += int(x['is_perfect'])
    streak=streak+1 if x['is_perfect'] else 0
    longest=max(longest,streak)
    assert (x['first_perfect_step'],x['perfect_evaluations'],x['perfect_streak'],x['longest_perfect_streak']) == (first,count,streak,longest)
    if i:
        assert np.isclose(r['mu_relative_delta'],r['mu_delta_l2']/(trace[i-1]['rows'][0]['mu_l2']+1e-12))
assert complete['first_perfect_step'] == first and complete['perfect_evaluations'] == count
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
        fn_min_median_max=np.quantile([r['fn'] for r in rows],[0,.5,1]).tolist(),
        soft4_min_median_max=np.quantile([r['soft4_loss'] for r in rows],[0,.5,1]).tolist(),
        balanced_min_median_max=np.quantile([r['balanced_loss'] for r in rows],[0,.5,1]).tolist())
best=max(trace,key=lambda x:x['rows'][0]['edge_f1'])
result=dict(final=trace[-1], first_perfect_step=first, perfect_count=count, longest_perfect_streak=longest,
    perfect_intervals=intervals, tail_windows=windows, best_step=best['step'], best=best['rows'][0],
    seconds=complete['seconds'], initial_sha256=meta['initial_sha256'], all_config_and_trace_checks_passed=True)
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')

fig,axes=plt.subplots(3,2,figsize=(15,11),constrained_layout=True)
for col,rows in enumerate([trace,trace[-500:]]):
    steps=[x['step'] for x in rows]
    f1=[x['rows'][0]['edge_f1']*100 for x in rows]
    axes[0,col].plot(steps,f1,color='#178D78',lw=.8)
    perfect=[x['step'] for x in rows if x['is_perfect']]
    if perfect:
        axes[0,col].scatter(perfect,[100]*len(perfect),s=6,color='#E19C27',label='FP=FN=0');axes[0,col].legend()
    axes[0,col].set(ylabel='Hard Edge F1 (%)',title='All 3,000 updates' if col==0 else 'Last 500 updates',
                    ylim=(-1,101) if col==0 else (max(0,min(f1)-.5),100.2))
    for key,color in [('fp','#C23B43'),('fn','#476FBB')]:
        axes[1,col].plot(steps,[x['rows'][0][key] for x in rows],label=key.upper(),color=color,lw=.8)
    axes[1,col].set(ylabel='Wrong / missing edges');axes[1,col].set_yscale('symlog',linthresh=1);axes[1,col].legend()
    axes[2,col].plot(steps,[x['objective'] for x in rows],label='Soft4 objective',color='#178D78',lw=.8)
    axes[2,col].plot(steps,[x['rows'][0]['balanced_loss'] for x in rows],label='GT-balanced BCE',color='#7060B0',lw=.8)
    axes[2,col].set(ylabel='Loss',xlabel='Completed updates',yscale='log');axes[2,col].legend()
for ax in axes.flat: ax.grid(alpha=.2)
fig.suptitle('Large-only Soft4 from the same random initial.pt | 2,575 vertices, 7,719 GT edges\nE LR=1e-5, D LR=1e-4 | tau=1, detached membership, FP32 reduction | hard threshold=0')
fig.savefig(ROOT/'training_curve.png',dpi=160);plt.close(fig)
f=trace[-1]['rows'][0]
lines=['# Large-only Soft4：同一随机初始化，3000步','',
       '从一直使用的initial.pt开始，逐参数核对模型权重相等；fresh Adam状态为空。未加载small训练权重。E LR=1e-5、D LR=1e-4、τ=1、detach membership、FP32组归约、ε=1e-8、μ mode、Face=KL=wd=0、clip=1，相同Flash后端，全程未调LR。', '',
       f"最终：TP={f['tp']}、FP={f['fp']}、FN={f['fn']}、TN={f['tn']}；F1={100*f['edge_f1']:.6f}%；Soft4={f['soft4_loss']:.8g}，GT-balanced BCE={f['balanced_loss']:.8g}。", '',
       f'首次严格100%：{first if first is not None else "未出现"}；总次数：{count}；最长连续保持：{longest}步。', '',
       f"最佳F1={100*best['rows'][0]['edge_f1']:.6f}%，首次达到该最佳值在step{best['step']}。", '',
       '|末尾窗口|F1 最小/中位/最大|FP=FN=0次数|','|---|---|---:|']
for n,w in windows.items():
    lines.append(f"|末{n}步|{' / '.join(f'{100*v:.5f}%' for v in w['f1_min_median_max'])}|{w['perfect_count']}/{n}|")
lines += ['', '首次达到100%与稳定保持100%分开判断；未达到100%也不能仅凭3000步推断更长训练永远无法达到。', '',
          '所有3001次评估均使用完整3314025个无序顶点对、固定threshold=0。summary.json包含完整100%区间和末200/500步的错边、漏边、两种loss分布；soft4/trace.jsonl包含每一步全部指标与μ变化。', '',
          '每200步checkpoint包含模型和Adam状态；首次100%若发生会另存该步。完整checkpoint和embedding在服务器本目录，本地取回完整日志。旧实验源码及initial.pt哈希前后未变；保留原Flash后端，不声称逐位数值确定性。', '',
          '![训练曲线](training_curve.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
