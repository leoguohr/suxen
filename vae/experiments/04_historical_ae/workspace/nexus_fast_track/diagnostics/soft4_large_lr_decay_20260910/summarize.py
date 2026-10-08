"""Verify and compare two LR branches from the same large Soft4 step3000 state."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
CASES={'control':(1e-5,1e-4),'decay':(1e-6,1e-5)}
LABELS={'control':'A: original LR','decay':'B: LR x0.1'}
COLORS={'control':'#BD4149','decay':'#16887A'}


def read(path):
    return json.loads(path.read_text())


traces,metas,summary={},{},{}
for case,rates in CASES.items():
    p=ROOT/case
    t=[json.loads(line) for line in (p/'trace.jsonl').read_text().splitlines()]
    meta,done=read(p/'provenance.json'),read(p/'complete.json')
    assert [x['step'] for x in t] == list(range(3000,4001))
    assert done['final']==t[-1] and done['optimizer_final_step']==4000
    assert meta['optimizer_restored_exactly'] and meta['resume_parameters_verified_tensor_equal']
    assert meta['only_optimizer_group_lr_changed'] and meta['additional_updates']==1000
    assert meta['start_step']==3000 and meta['selected_uids']==['nexus_2k_001849']
    assert meta['objective']=='soft4' and meta['script_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
    first,count,streak,longest=None,0,0,0
    for i,x in enumerate(t):
        r=x['rows'][0]
        assert len(x['rows'])==1 and r['tp']+r['fn']==7719 and r['tn']+r['fp']==3306306
        assert r['edge_f1']==2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
        assert x['is_perfect']==r['is_perfect']==(r['fp']==r['fn']==0)
        assert (x['encoder_lr'],x['decoder_lr'])==rates
        assert x['objective']==r['soft4_loss'] and x['rng_unchanged'] and x['frozen_unchanged']
        assert np.isclose(r['soft4_loss'],np.mean([g['mean'] for g in r['soft4_groups'].values()]))
        if x['is_perfect'] and first is None:first=x['step']
        count+=int(x['is_perfect']);streak=streak+1 if x['is_perfect'] else 0;longest=max(longest,streak)
        assert (x['first_perfect_step'],x['perfect_evaluations'],x['perfect_streak'],x['longest_perfect_streak'])==(first,count,streak,longest)
        if i:assert np.isclose(r['mu_relative_delta'],r['mu_delta_l2']/(t[i-1]['rows'][0]['mu_l2']+1e-12))
    assert done['first_perfect_step']==first and done['perfect_evaluations']==count
    tails={}
    for n in [200,500]:
        rows=[x['rows'][0] for x in t[-n:]]
        tails[str(n)]={key:np.quantile([r[key] for r in rows],[0,.5,1]).tolist()
                      for key in ['edge_f1','tp','fp','fn','soft4_loss','balanced_loss','mu_relative_delta']}
        tails[str(n)]['perfect_count']=sum(r['is_perfect'] for r in rows)
    best=max(t,key=lambda x:x['rows'][0]['edge_f1'])
    intervals=[]
    for x in t:
        if x['is_perfect']:
            if intervals and x['step']==intervals[-1][1]+1:intervals[-1][1]=x['step']
            else:intervals.append([x['step'],x['step']])
    summary[case]=dict(final=t[-1]['rows'][0],first_perfect_step=first,perfect_count=count,
        longest_perfect_streak=longest,perfect_intervals=intervals,tail_windows=tails,
        best_step=best['step'],best=best['rows'][0],seconds=done['seconds'])
    traces[case]=t;metas[case]=meta
for key in ['resume_checkpoint','resume_sha256','source_sha256','loss_script_sha256','soft4_tau','soft4_epsilon',
            'soft4_membership','soft4_group_reduction','threshold','mode','optimizer','gpu']:
    assert metas['control'][key]==metas['decay'][key],key
npzs={c:np.load(ROOT/c/'step3000_nexus_2k_001849.npz') for c in CASES}
start_diff={k:float(np.linalg.norm(npzs['decay'][k].astype(float)-npzs['control'][k])/np.linalg.norm(npzs['control'][k].astype(float))) for k in ['mu','edge']}
verification=dict(same_start_checkpoint_sha256=metas['control']['resume_sha256'],model_and_adam_restored_both=True,
    fixed_lr_per_branch_verified=True,all_1001_evaluations_per_branch_verified=True,
    start_forward_B_vs_A_relative_l2=start_diff,
    start_counts={c:{k:traces[c][0]['rows'][0][k] for k in ['tp','fp','fn','edge_f1']} for c in CASES})
(ROOT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

fig,axes=plt.subplots(4,1,figsize=(12,13),constrained_layout=True)
for case,t in traces.items():
    steps=[x['step'] for x in t];rows=[x['rows'][0] for x in t];color=COLORS[case]
    axes[0].plot(steps,[r['edge_f1']*100 for r in rows],color=color,label=LABELS[case],lw=.9)
    perfect=[x['step'] for x in t if x['is_perfect']]
    if perfect:axes[0].scatter(perfect,[100]*len(perfect),color=color,s=8)
    for key,style in [('fp','-'),('fn','--')]:
        axes[1].plot(steps,[r[key] for r in rows],color=color,ls=style,label=LABELS[case]+' '+key.upper(),lw=.8)
    axes[2].plot(steps,[r['soft4_loss'] for r in rows],color=color,lw=.8)
    axes[3].plot(steps,[r['balanced_loss'] for r in rows],color=color,lw=.8)
axes[0].set(ylabel='Hard Edge F1 (%)',ylim=(100*min(x['rows'][0]['edge_f1'] for t in traces.values() for x in t)-.2,100.1));axes[0].legend()
axes[1].set(ylabel='Wrong / missing edges');axes[1].set_yscale('symlog',linthresh=1);axes[1].legend(ncol=2)
axes[2].set(ylabel='Soft4 objective',yscale='log')
axes[3].set(ylabel='GT-balanced BCE',yscale='log',xlabel='Completed updates')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Large-only Soft4 LR decay: same step3000 weights and Adam state\nA: E=1e-5, D=1e-4 | B: E=1e-6, D=1e-5 | 1,000 additional updates each')
fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
lines=['# Large-only Soft4：原LR vs LR×0.1','',
       '两个分支独立恢复同一个step3000模型和Adam全部状态。A：E=1e-5、D=1e-4；B：E=1e-6、D=1e-5。各新增1000次更新，到step4000。只改变B的参数组LR，其余Adam字段、Soft4、τ=1、detach membership、FP32组归约、μ mode、Face=KL=wd=0、clip=1和后端相同。两组在同一张A100上依次运行。','',
       '|分支|最终F1|TP / FP / FN|Soft4|GT-balanced BCE|首次100%|100%次数|最长连续|','|---|---:|---|---:|---:|---|---:|---:|']
for c,s in summary.items():
    r=s['final']
    lines.append(f"|{LABELS[c]}|{100*r['edge_f1']:.5f}%|{r['tp']} / {r['fp']} / {r['fn']}|{r['soft4_loss']:.7g}|{r['balanced_loss']:.7g}|{s['first_perfect_step'] if s['first_perfect_step'] else '未出现'}|{s['perfect_count']}|{s['longest_perfect_streak']}|")
lines+=['','|分支/窗口|F1 最小 / 中位 / 最大|100%次数|','|---|---|---:|']
for c,s in summary.items():
    for n,w in s['tail_windows'].items():lines.append(f"|{LABELS[c]} / 末{n}步|{' / '.join(f'{100*v:.5f}%' for v in w['edge_f1'])}|{w['perfect_count']}/{n}|")
lines+=['','## 结果解释','',
        '同一个step3000起点，原LR分支始终未达到100%；降低E/D学习率到原来的0.1倍后，新增171步首次达到100%，1000步续训中共544次完全重建。最后200步F1中位数从A的99.447%提高到B的100%，且B的两种loss明显更低。该对照支持学习率是这个训练阶段的重要限制因素。','',
        'B仍未始终保持100%：最长连续39步，最后200步139次完全重建，最后一步仍有1条错边、2条漏边。不能将“已经找到过完全重建参数”表述为“完全稳定收敛”。','',
        '起点权重和Adam状态逐张量核对相等，两个分支checkpoint SHA相同。初始前向B相对A的L2差异：`'+json.dumps(start_diff)+'`；保留原Flash后端，不声称逐位确定性。','',
        '不能把一次100%等同于稳定保持，也不能根据一次LR对照推断所有初始化或更长训练的上限。完整末200/500步TP/FP/FN、loss、μ变化统计见summary.json。','',
        '每200步保存模型与Adam checkpoint；首次100%若出现，另存该步。完整日志已取回本地，checkpoint与embedding保留在服务器对应目录。','',
        '![LR对照曲线](comparison.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
