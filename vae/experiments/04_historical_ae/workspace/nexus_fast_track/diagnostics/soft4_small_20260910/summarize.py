"""Matched small-only metrics and complete TP-birth/return event audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'single_mesh_four_20260910/small'
CASES = {'original': OLD, 'fixed4': ROOT/'fixed4', 'soft4': ROOT/'soft4', 'balanced': ROOT/'balanced'}
LABELS = {'original': 'Original four-group', 'fixed4': 'Fixed4', 'soft4': 'Soft4', 'balanced': 'GT-balanced'}
COLORS = {'original': '#C23B43', 'fixed4': '#D38B21', 'soft4': '#178D78', 'balanced': '#7060B0'}
FIELDS = {'original': 'four_loss', 'fixed4': 'fixed4_loss', 'soft4': 'soft4_loss', 'balanced': 'balanced_loss'}
traces, summary, events, metas = {}, {}, {}, {}


def read(path):
    return json.loads(path.read_text())


def point(t, i):
    r=t[i]['rows'][0]
    return dict(step=i, objective=t[i]['objective'], tp=r['tp'], fp=r['fp'], fn=r['fn'], tn=r['tn'],
                edge_f1=r['edge_f1'], balanced_loss=r['balanced_loss'])


for case, folder in CASES.items():
    t=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    meta, completed=read(folder/'provenance.json'), read(folder/'complete.json')
    assert [x['step'] for x in t] == list(range(1201))
    assert completed['final'] == t[-1] and completed['steps'] == 1200
    assert meta['selected_uids'] == ['nexus_2k_000387']
    assert meta['encoder_lr'] == 1e-5 and meta['decoder_lr'] == 1e-4
    if case != 'original':
        assert meta['script_sha256'] == hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
    for i,x in enumerate(t):
        assert len(x['rows']) == 1 and x['frozen_unchanged'] and x['rng_unchanged']
        r=x['rows'][0]
        assert r['tp']+r['fn'] == 1152 and r['tn']+r['fp'] == 73153
        assert r['edge_f1'] == 2*r['tp']/max(2*r['tp']+r['fp']+r['fn'],1)
        assert np.isclose(x['objective'],r[FIELDS[case]])
        if case != 'original':
            assert np.isclose(r['fixed4_loss'],r['four_loss']*sum(r[g]>0 for g in ['tp','tn','fp','fn'])/4)
            groups=r['soft4_groups']
            for g in groups.values():
                assert np.isclose(g['mean'],g['numerator']/(g['mass']+1e-8),rtol=2e-6)
            assert np.isclose(r['soft4_loss'],np.mean([g['mean'] for g in groups.values()]))
        if i:
            assert np.isclose(r['mu_relative_delta'],r['mu_delta_l2']/(t[i-1]['rows'][0]['mu_l2']+1e-12))
    rows=[x['rows'][0] for x in t]; tail=rows[-200:]
    best=max(range(len(t)),key=lambda i: rows[i]['edge_f1'])
    perfect=[i for i,r in enumerate(rows) if r['fp']==r['fn']==0]
    summary[case]=dict(final=rows[-1], final_objective=t[-1]['objective'], best_step=best,best_f1=rows[best]['edge_f1'],
        first_perfect_step=min(perfect) if perfect else None,
        last200_perfect_count=sum(r['fp']==r['fn']==0 for r in tail),
        last200_f1_min_median_max=np.quantile([r['edge_f1'] for r in tail],[0,.5,1]).tolist(),
        last200_mu_drift_median_p95_max=np.quantile([r['mu_relative_delta'] for r in tail],[.5,.95,1]).tolist(),
        tp_zero_evaluations=sum(r['tp']==0 for r in rows), seconds=completed['seconds'])
    births=[i for i in range(1,len(t)) if rows[i-1]['tp']==0 and rows[i]['tp']>0]
    ev=[]
    for i in births:
        end=next((j for j in range(i+1,len(t)) if rows[j]['tp']==0),None)
        ev.append(dict(birth=i,return_to_zero=end, before=point(t,i-1), at_birth=point(t,i),
            birth_objective_delta=t[i]['objective']-t[i-1]['objective'],
            before_return=point(t,end-1) if end else None, at_return=point(t,end) if end else None,
            return_objective_delta=t[end]['objective']-t[end-1]['objective'] if end else None,
            peak_tp=max(r['tp'] for r in rows[i:end] if r) if end else max(r['tp'] for r in rows[i:]),
            window=[point(t,j) for j in range(max(0,i-3),min(len(t),(end if end else i)+4))]))
    events[case]=ev;traces[case]=t;metas[case]=meta
assert len({m['initial_sha256'] for m in metas.values()})==1
assert all(m['source_sha256']==metas['original']['source_sha256'] for m in metas.values())

# Three complete birth/return episodes with the largest upward birth changes per hard objective.
selected={case:sorted(sorted([e for e in events[case] if e['return_to_zero'] is not None],
                           key=lambda e:e['birth_objective_delta'],reverse=True)[:3],key=lambda e:e['birth'])
          for case in ['original','fixed4']}
for case, es in selected.items():
    for e in es:
        e['soft4_same_steps']=[point(traces['soft4'],p['step']) for p in e['window']]
        b,d=e['birth'],e['return_to_zero']
        e['soft4_delta_at_same_birth_step']=traces['soft4'][b]['objective']-traces['soft4'][b-1]['objective']
        e['soft4_delta_at_same_return_step']=traces['soft4'][d]['objective']-traces['soft4'][d-1]['objective']

fig,axes=plt.subplots(3,1,figsize=(12,12),constrained_layout=True)
for case,t in traces.items():
    rows=[x['rows'][0] for x in t]
    axes[0].plot([r['edge_f1']*100 for r in rows],label=LABELS[case],color=COLORS[case],lw=1)
    axes[1].plot([r['balanced_loss'] for r in rows],color=COLORS[case],lw=1)
    axes[2].plot([x['objective'] for x in t],color=COLORS[case],lw=1)
axes[0].set(ylabel='Hard Edge F1 (%)',ylim=(-2,102));axes[0].legend(ncol=2)
axes[1].set(ylabel='Same GT-balanced BCE',yscale='log')
axes[2].set(ylabel='Training objective (different definitions)',xlabel='Completed updates')
for ax in axes: ax.grid(alpha=.2)
fig.suptitle('Small-only: 386 vertices, same initial weights, 1,200 updates\nE LR=1e-5 | D LR=1e-4 | mu | Face=KL=0 | hard threshold=0',fontsize=14)
fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(17,8),constrained_layout=True)
for row,case in enumerate(['original','fixed4']):
    for col in range(3):
        ax=axes[row,col]
        if col>=len(selected[case]): ax.set_visible(False);continue
        e=selected[case][col];steps=[p['step'] for p in e['window']]
        for c in [case,'soft4']:
            ax.plot(steps,[traces[c][i]['objective'] for i in steps],label=LABELS[c]+' objective',color=COLORS[c],marker='.',ms=3)
        twin=ax.twinx()
        twin.step(steps,[traces[case][i]['rows'][0]['tp'] for i in steps],where='mid',color='#555555',alpha=.5,label='Hard TP (source)',ls=':')
        twin.set_ylabel('TP (hard-loss run)',color='#555555')
        ax.axvline(e['birth'],color='#999999',lw=.8,ls='--');ax.axvline(e['return_to_zero'],color='#999999',lw=.8,ls='--')
        ax.set(title=f"{LABELS[case]}: birth {e['birth']}, return {e['return_to_zero']}\nBirth objective delta = {e['birth_objective_delta']:+.5f}",xlabel='Completed updates',ylabel='Objective')
        ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.suptitle('TP=0 -> TP>0 -> TP=0: three largest upward birth changes per hard loss\nSoft4 shown at the same step numbers, not the same model parameters',fontsize=14)
fig.savefig(ROOT/'tp_events.png',dpi=150);plt.close(fig)

(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'tp_events.json').write_text(json.dumps(dict(all_events=events,selected=selected,
    selection='top 3 complete episodes by positive birth objective delta, then chronological',
    warning='Across-update deltas include parameter changes. Cross-run same steps are not the same parameters.'),indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps(dict(initial_sha256=metas['original']['initial_sha256'],
    four_small_only_runs=True,trace_rows_per_case=1201,counts_f1_objective_mu_and_soft_reduction_verified=True,
    original_reused=True,fixed4_and_balanced_added_because_previous_runs_were_two_mesh=True),indent=2)+'\n')
lines=['# Small-only：Original / Fixed4 / Soft4 / GT-balanced','',
       'Original直接引用上轮small-only；旧Fixed4和GT-balanced实际是双mesh，本次补齐small-only，未重跑旧实验。四组来自同一initial.pt，E LR=1e-5、D LR=1e-4、μ mode、Face=KL=wd=0、clip=1、fresh Adam，保留相同Flash后端。', '',
       'Soft4：τ=1、ε=1e-8、sigmoid membership detach、四组分子分母均用FP32全顶点对求和，再分别归一化，固定除4。所有组每步用logit>0计算同一硬指标。', '',
       '|训练|最终F1|最后200步F1 最小/中位/最大|最终GT-balanced BCE|训练objective|TP/FP/FN|','|---|---:|---|---:|---:|---|']
for c,s in summary.items():
    r=s['final'];q=s['last200_f1_min_median_max']
    lines.append(f"|{LABELS[c]}|{r['edge_f1']*100:.4f}%|{' / '.join(f'{x*100:.3f}%' for x in q)}|{r['balanced_loss']:.7g}|{s['final_objective']:.7g}|{r['tp']}/{r['fp']}/{r['fn']}|")
lines += ['', '训练objective定义不同，不能按其绝对数值直接排名；统一BCE和硬F1才是横向比较口径。', '', '## TP出现又消失的事件', '',
          '下表从每条硬loss轨迹的完整事件中选取birth objective上升最大的3次，完整事件与窗口在tp_events.json。Soft4列是相同步数的变化，不是相同参数下的反事实。','',
          '|训练|TP birth → return|birth objective变化|return objective变化|Soft4相同步数birth变化|','|---|---|---:|---:|---:|']
for c,es in selected.items():
    for e in es: lines.append(f"|{LABELS[c]}|{e['birth']} → {e['return_to_zero']}|{e['birth_objective_delta']:+.7f}|{e['return_objective_delta']:+.7f}|{e['soft4_delta_at_same_birth_step']:+.7f}|")
lines += ['', 'Soft4自身的所有TP出现事件：', '', '|birth|出现的TP数|前一步objective|本步objective|变化|再次TP=0|','|---:|---:|---:|---:|---:|---|']
for e in events['soft4']:
    lines.append(f"|{e['birth']}|{e['at_birth']['tp']}|{e['before']['objective']:.8f}|{e['at_birth']['objective']:.8f}|{e['birth_objective_delta']:+.8f}|{e['return_to_zero'] if e['return_to_zero'] is not None else '之后未出现'}|")
lines += ['', '各组自身TP从0出现的次数：'+', '.join(f'{LABELS[c]}={len(es)}' for c,es in events.items())+'。', '',
          'Soft4没有硬预测分组开关，因此不存在由logit恰好跨0触发的离散分组重定义。但相邻训练步仍可能有较大变化：参数更新、相对很小的soft组质量以及stop-gradient重算权重都需区分。不能把时间序列的每个跳变都归于分组开关，也不能认为连续就必然稳定收敛。', '',
          'stop-gradient意味着反传把本次membership视为常量；下一次forward仍会重算membership。其反传方向不是把membership也求导后的完整标量函数梯度。', '',
          '## 结论边界', '',
          '同一初始化和网络下，Original/Fixed4最终为空边图，Soft4最终达到99.13%硬F1，说明修改训练loss足以显著改变当前重建结果；不支持把失败简单归结为encoder或decoder容量不足。Soft4最后200步F1仍在93.04%–99.34%之间，没有一次严格达到100%，尚未完全overfit。', '',
          'GT-balanced最终BCE更低（0.02282 vs Soft4的0.03192），却有351条错边，Soft4只有1条错边。这再次说明较低的平均BCE不自动等于更好的零阈值重建。', '',
          'Soft4既取消硬组出现/消失，也改变了组内样本权重，因此本实验没有单独隔离“不连续性”的全部因果贡献。证据强烈指向loss分组/加权设计是当前训练的重要障碍，不能宣布已证明唯一根因。', '',
          '每200步checkpoint和embedding保留在服务器本目录；本地保存完整逐步日志、对照图、事件窗口与核验记录。','',
          '![四组对照](comparison.png)','', '![TP事件窗口](tp_events.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
