"""Verify resumed two-mesh metrics, early stopping, and generate the result curves."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'soft4_two_mesh_recipe_20260910'
OUT = ROOT/'continue'
UIDS = ['nexus_2k_000387', 'nexus_2k_001849']


def read(path):
    return json.loads(path.read_text())


def stats(t):
    streak = longest = 0
    for x in t:
        streak = streak+1 if x['I_t'] else 0
        longest = max(streak, longest)
    return dict(start=t[0]['step'], end=t[-1]['step'], observations=len(t),
                first_both_step=next((x['step'] for x in t if x['I_t']), None),
                both_count=sum(x['I_t'] for x in t), longest_both_streak=longest,
                meshes={uid: dict(perfect_count=sum(x['rows'][i]['is_perfect'] for x in t),
                    **{k:np.quantile([x['rows'][i][k] for x in t], [0,.5,1]).tolist()
                       for k in ['edge_f1','tp','fp','fn','soft4_loss','balanced_loss']})
                        for i,uid in enumerate(UIDS)})


t = [json.loads(x) for x in (OUT/'trace.jsonl').read_text().splitlines()]
old = [json.loads(x) for x in (PREVIOUS/'combined_trace.jsonl').read_text().splitlines()]
meta, done = read(OUT/'provenance.json'), read(OUT/'complete.json')
assert [x['step'] for x in t] == list(range(4000,done['steps']+1))
assert done['steps'] <= 6000 and done['optimizer_final_step'] == done['steps']
assert done['additional_updates'] == done['steps']-4000 and done['final'] == t[-1]
assert meta['optimizer_restored_exactly'] and meta['optimizer_fields_unchanged']
assert meta['loaded_weights_verified_tensor_equal'] and meta['simultaneous_perfect_target'] == 20
assert meta['script_sha256'] == hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert done['original_sources_unchanged'] and done['frozen_and_rng_checks_passed']
for i,x in enumerate(t):
    assert len(x['rows']) == 2 and (x['encoder_lr'],x['decoder_lr']) == (1e-6,1e-5)
    assert x['frozen_unchanged'] and x['rng_unchanged']
    assert np.isclose(x['objective'],sum(r['soft4_loss'] for r in x['rows'])/2)
    for j,r in enumerate(x['rows']):
        assert r['uid'] == UIDS[j]
        assert r['tp']+r['fn'] == [1152,7719][j] and r['tn']+r['fp'] == [73153,3306306][j]
        assert r['is_perfect'] == (r['fp'] == r['fn'] == 0)
        assert r['edge_f1'] == 2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
        assert np.isclose(r['soft4_loss'],np.mean([g['mean'] for g in r['soft4_groups'].values()]))
        if i:
            assert np.isclose(r['mu_relative_delta'],r['mu_delta_l2']/(t[i-1]['rows'][j]['mu_l2']+1e-12))
    assert x['I_t'] == int(all(r['is_perfect'] for r in x['rows']))
    assert x['early_stop'] == (x['both_count'] >= 20)
assert not any(x['early_stop'] for x in t[:-1])
assert done['early_stop'] == t[-1]['early_stop']
assert (done['early_stop'] and done['both_count'] == 20) or done['steps'] == 6000
assert 'next_update_preclip_grad_l2' not in t[-1]
combined = old[:-1]+t
assert [x['step'] for x in combined] == list(range(done['steps']+1))
first, count, streak, longest = None,0,0,0
for x in combined:
    if x['I_t'] and first is None:
        first = x['step']
    count += x['I_t']; streak = streak+1 if x['I_t'] else 0; longest = max(streak,longest)
    assert (first,count,streak,longest) == (x['first_both_step'],x['both_count'],x['both_streak'],x['longest_both_streak'])
summary = dict(overall=stats(combined), continuation=stats(t),
               tail_windows={str(n):stats(combined[-n:]) for n in [200,500]},
               early_stop=done['early_stop'],stop_reason=done['stop_reason'],
               final=t[-1],seconds=done['seconds'],additional_updates=done['additional_updates'],
               first_both_checkpoint=next((x.get('saved_checkpoint') for x in t if x['I_t']),None),
               best_observation=max(combined,key=lambda x:min(r['edge_f1'] for r in x['rows'])),
               resume_verification=read(OUT/'resume_verification.json'))
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'combined_trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in combined))
(ROOT/'verification.json').write_text(json.dumps(dict(all_steps_and_metrics_verified=True,
    early_stop_checked_before_next_update=True,adam_restored_all_fields=True,
    start_checkpoint_sha256=meta['start_checkpoint_sha256'],
    boundary_counting='Previous0..3999 + resumed4000..stop; no duplicated step4000'),indent=2)+'\n')

fig,axes = plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
steps=[x['step'] for x in t]
for i,(name,color) in enumerate([('Small','#16887a'),('Large','#bd4149')]):
    rows=[x['rows'][i] for x in t]
    axes[0,0].plot(steps,[100*r['edge_f1'] for r in rows],color=color,label=name,lw=.8)
    for k,style in [('fp','-'),('fn','--')]:
        axes[0,1].plot(steps,[r[k] for r in rows],color=color,ls=style,label=name+' '+k.upper(),lw=.8)
    axes[1,0].plot(steps,[r['soft4_loss'] for r in rows],color=color,lw=.8)
    axes[1,1].plot(steps,[r['balanced_loss'] for r in rows],color=color,lw=.8)
    axes[2,0].plot(steps,[r['mu_relative_delta'] for r in rows],color=color,lw=.8)
axes[0,0].set(ylabel='Hard Edge F1 (%)');axes[0,0].legend()
axes[0,1].set(ylabel='FP / FN',yscale='symlog');axes[0,1].legend(ncol=2)
axes[1,0].set(ylabel='Soft4',yscale='log');axes[1,1].set(ylabel='GT-balanced BCE',yscale='log')
axes[2,0].set(ylabel='Relative change of mu',yscale='log')
axes[2,1].plot(steps,[x['both_count'] for x in t],color='#5c477d',label='Cumulative both-perfect count')
axes[2,1].axhline(20,ls='--',color='gray',label='Acceptance / early stop: 20')
axes[2,1].set(ylabel='Simultaneous perfect observations',ylim=(-.5,21));axes[2,1].legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.2);ax.set_xlabel('Completed updates')
fig.suptitle('Two-mesh Soft4 continuation: same step4000 model and Adam state\nE LR=1e-6, D LR=1e-5; early stop at20 simultaneous-perfect observations, limit6000')
fig.savefig(ROOT/'continuation_curve.png',dpi=150);plt.close(fig)

lines=['# Two-mesh Soft4：step4000续训', '',
       f"从step4000恢复模型和Adam全部状态，E/D LR保持1e-6/1e-5。实际新增{done['additional_updates']}步，结束于step{done['steps']}；停止原因：{done['stop_reason']}。", '',
       '提前停止标准：同一次前向两条mesh均FP=FN=0，累计出现20次；无需连续。达到标准时保存checkpoint，不再执行下一次更新。其余配置、数据、Soft4定义与之前相同。', '',
       f'首次同时100%：{first}；累计次数：{count}；最长连续次数：{longest}。None表示未出现。', '',
       '|最终mesh|TP / FP / FN|F1|Soft4|GT-balanced BCE|','|---|---|---:|---:|---:|']
for name,r in zip(['Small','Large'],t[-1]['rows']):
    lines.append(f"|{name}|{r['tp']} / {r['fp']} / {r['fn']}|{100*r['edge_f1']:.6f}%|{r['soft4_loss']:.8g}|{r['balanced_loss']:.8g}|")
lines += ['', '|窗口|同时100%次数|Small F1 min / median / max|Large F1 min / median / max|','|---|---:|---|---|']
for n,w in summary['tail_windows'].items():
    f=[' / '.join(f'{100*v:.6f}%' for v in w['meshes'][uid]['edge_f1']) for uid in UIDS]
    lines.append(f"|最后{n}步|{w['both_count']}/{n}|{f[0]}|{f[1]}|")
lines += ['', '累计20次是本次sanity-check验收门槛，并不等于之后每一步都保持完美重建。Face与KL没有训练；本次验收只针对这两条训练mesh的边。', '',
          '继续使用原Flash/CUDA后端；没有随机采样不意味着数值逐位确定。step4000前后两次前向均保留，合并统计只计算续训重放的一次。权重与Adam状态已逐张量验证。', '',
          f"最终checkpoint（服务器）：`{t[-1].get('saved_checkpoint')}`。", '',
          '![续训曲线](continuation_curve.png)', '']
best = summary['best_observation']
lines += [f"两mesh中较低F1最高的一次观测：step{best['step']}，small/large F1="+
          ' / '.join(f"{100*r['edge_f1']:.6f}%" for r in best['rows'])+
          '。该项为日志中的最好观测，不保证该步恰逢checkpoint保存。', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(first=first,count=count,longest=longest,end=done['steps'],early_stop=done['early_stop']),indent=2))
