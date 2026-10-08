"""Verify500 updates per branch and compare large-only training with small monitoring."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
UIDS = ['nexus_2k_000387','nexus_2k_001849']
RATES = dict(A=(1e-6,1e-5),B=(3e-7,3e-6),C=(1e-7,1e-6))
COLORS = dict(A='#bd4149',B='#16887a',C='#6654a1')


def read(path):
    return json.loads(path.read_text())


def stats(t,index=None):
    flags=[x['I_t'] if index is None else x['rows'][index]['is_perfect'] for x in t]
    streak=longest=0
    for flag in flags:
        streak=streak+1 if flag else 0;longest=max(longest,streak)
    result=dict(first_step=next((x['step'] for x,f in zip(t,flags) if f),None),
                count=sum(flags),longest=longest,streak=streak)
    if index is not None:
        rows=[x['rows'][index] for x in t]
        result.update({k:np.quantile([r[k] for r in rows],[0,.5,1]).tolist()
                       for k in ['edge_f1','tp','fp','fn','soft4_loss','balanced_loss']})
    return result


traces,metas,summaries={},{},{}
for case,rates in RATES.items():
    folder=ROOT/case
    t=[json.loads(line) for line in (folder/'trace.jsonl').read_text().splitlines()]
    meta,done=read(folder/'provenance.json'),read(folder/'complete.json')
    assert [x['step'] for x in t] == list(range(6000,6501))
    assert done['steps']==done['optimizer_final_step']==6500 and done['additional_updates']==500
    assert not done['early_stop'] and done['final']==t[-1]
    assert meta['start_step']==6000 and meta['updates']==500
    assert meta['optimizer_restored_exactly'] and meta['optimizer_fields_except_lr_unchanged']
    assert meta['loaded_weights_verified_tensor_equal'] and meta['selected_uids']==UIDS
    assert meta['script_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
    assert done['frozen_and_rng_checks_passed'] and done['original_sources_unchanged']
    for j,x in enumerate(t):
        assert len(x['rows'])==2 and not x['small_loss_in_backward']
        assert (x['encoder_lr'],x['decoder_lr'])==rates
        assert x['objective']==x['rows'][1]['soft4_loss']
        assert x['rng_unchanged'] and x['frozen_unchanged']
        for i,r in enumerate(x['rows']):
            assert r['uid']==UIDS[i]
            assert r['tp']+r['fn']==[1152,7719][i] and r['tn']+r['fp']==[73153,3306306][i]
            assert r['edge_f1']==2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
            assert r['is_perfect']==(r['fp']==r['fn']==0)
            assert math.isclose(r['soft4_loss'],np.mean([g['mean'] for g in r['soft4_groups'].values()]),rel_tol=1e-6)
            if j:
                assert math.isclose(r['mu_relative_delta'],r['mu_delta_l2']/(t[j-1]['rows'][i]['mu_l2']+1e-12),rel_tol=1e-10)
                s=stats(t[1:j+1],i)
                assert {k:s[k] for k in ['first_step','count','streak','longest']}==x['per_mesh_perfect'][r['uid']]
        assert x['I_t']==int(all(r['is_perfect'] for r in x['rows']))
        if j:
            s=stats(t[1:j+1])
            assert (x['first_both_step'],x['both_count'],x['both_streak'],x['longest_both_streak'])==(s['first_step'],s['count'],s['streak'],s['longest'])
    assert 'next_update_preclip_grad_l2' not in t[-1]
    updates=t[1:]
    summaries[case]=dict(rates=rates,start=t[0]['rows'],final=t[-1]['rows'],
        small=stats(updates,0),large=stats(updates,1),both=stats(updates),seconds=done['seconds'],
        tail200=dict(small=stats(updates[-200:],0),large=stats(updates[-200:],1),both=stats(updates[-200:])),
        final_checkpoint=t[-1]['saved_checkpoint'])
    traces[case],metas[case]=t,meta
for case in ['B','C']:
    for key in ['start_checkpoint_sha256','source_sha256','loss_script_sha256','initial_sha256','soft4_tau',
                'soft4_epsilon','soft4_membership','soft4_group_reduction','threshold','mode','clip_norm','weight_decay',
                'loss_reduction','hardware','torch_version','cuda_version','trainable_encoder_parameters','trainable_decoder_parameters']:
        assert metas[case][key]==metas['A'][key],key
(ROOT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps(dict(all_500_updates_per_case_verified=True,
    same_step6000_sha256=metas['A']['start_checkpoint_sha256'],model_and_adam_restored_all_cases=True,
    only_large_loss_used_every_update=True,hard_metrics_counts_and_lr_verified=True,
    perfect_counting='500 new updates6001..6500 only; baseline6000 excluded'),indent=2)+'\n')

fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
for case,t in traces.items():
    steps=[x['step']-6000 for x in t];color=COLORS[case]
    for col,index in [(0,1),(1,0)]:
        axes[0,col].plot(steps,[100*x['rows'][index]['edge_f1'] for x in t],color=color,lw=.8,label=case)
        for key,style in [('fp','-'),('fn','--')]:
            axes[1,col].plot(steps,[x['rows'][index][key] for x in t],color=color,ls=style,lw=.8,label=case+' '+key.upper())
    axes[2,0].plot(steps,[x['per_mesh_perfect'][UIDS[1]]['count'] for x in t],color=color,label=case)
    axes[2,1].plot(steps,[x['both_count'] for x in t],color=color,label=case)
axes[0,0].set(ylabel='Large Edge F1 (%)',title='Large: trained')
axes[0,1].set(ylabel='Small Edge F1 (%)',title='Small: forward monitoring only')
for col in [0,1]:
    axes[0,col].legend();axes[1,col].set(ylabel='FP / FN',yscale='symlog');axes[1,col].legend(ncol=3,fontsize=8)
if all(x['rows'][0]['is_perfect'] for t in traces.values() for x in t):
    axes[0,1].set(ylim=(99.9,100.005),yticks=[99.9,99.95,100.0])
    axes[1,1].set(yscale='linear',ylim=(-.05,1.05),yticks=[0,1])
    axes[1,1].text(.5,.5,'All three runs: small FP = FN = 0 at every step',
                   transform=axes[1,1].transAxes,ha='center',fontsize=9)
axes[2,0].set(ylabel='Cumulative large-perfect count');axes[2,1].set(ylabel='Cumulative both-perfect count')
for ax in axes.flat:
    ax.grid(alpha=.2);ax.set_xlabel('Additional updates from step6000')
fig.suptitle('Large-only local LR test: same step6000 model + Adam, E and D trainable\nA: 1e-6 / 1e-5 | B: 3e-7 / 3e-6 | C: 1e-7 / 1e-6 |500 updates each')
fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)

lines=['# Step6000后期局部实验：只训练large，small逐步检查','',
       '三支均独立恢复同一个two-mesh step6000 checkpoint和Adam全部状态，E+D可训练。A/B/C只改变学习率。每步两条完整mesh共同前向；训练objective=1×Soft4_large，small的loss不反传。μ mode、τ=1、detach membership、FP32组归约、Face=KL=wd=0、clip=1、原Flash后端不变。各跑满500步，不提前停止。','',
       '100%次数统计新增500步（6001～6500），不包含6000起点。','',
       '|分支|E / D LR|large最终FP / FN|large最终F1|large首次100%|large100%次数|最长连续|small最终FP / FN|small100%次数|同时100%次数|',
       '|---|---|---|---:|---|---:|---:|---|---:|---:|']
for case,s in summaries.items():
    small,large=s['final'];lp=s['large']
    lines.append(f"|{case}|{s['rates'][0]:g} / {s['rates'][1]:g}|{large['fp']} / {large['fn']}|{100*large['edge_f1']:.6f}%|{lp['first_step']}|{lp['count']}/500|{lp['longest']}|{small['fp']} / {small['fn']}|{s['small']['count']}/500|{s['both']['count']}/500|")
lines += ['','None表示未出现100%。','',
          '|分支|large末200步F1 min / median / max|large末200步100%次数|最终large Soft4 / GT-balanced BCE|','|---|---|---:|---|']
for case,s in summaries.items():
    w=s['tail200']['large'];r=s['final'][1]
    lines.append(f"|{case}|{' / '.join(f'{100*v:.6f}%' for v in w['edge_f1'])}|{w['count']}/200|{r['soft4_loss']:.8g} / {r['balanced_loss']:.8g}|")
lines += ['','起点模型与Adam状态逐张量核对相等，只有LR按分支覆盖。原Flash/CUDA数值非逐位确定；三个起点重新前向的结果均保留，不能将微小重放差异称为不同checkpoint。','',
          '|起点重放|small FP / FN|large FP / FN|','|---|---|---|']
for case,s in summaries.items():
    a,b=s['start'];lines.append(f"|{case}|{a['fp']} / {a['fn']}|{b['fp']} / {b['fn']}|")
lines += ['','当前loss是单独large的完整Soft4，而之前two-mesh训练是0.5×small+0.5×large，且继承之前的Adam状态。三个分支之间比较口径一致；不能将与旧two-mesh轨迹的差异全部归因于移除small。','',
          '每200步及最终保存模型和Adam；首次large或两条同时100%也保存。达到过100%、100%出现频率和持续稳定性需要分别判断；本实验不检验face重建或泛化。','',
          '![三支对照曲线](comparison.png)','']
check=read(ROOT/'checkpoint_verification.json')
assert check['optimizer_steps_in_verification']==0
assert all(x['adam_step']==6500 and x['metrics_match_trace'] for x in check['final_checkpoints'].values())
lines += ['结果支持：在这个后期起点、large-only训练条件下，A的学习率仍偏大；进一步降低LR明显提高了完全重建的频率和连续性。C最后235步两条同时100%，最后200步全部通过。','',
          'C最终checkpoint已重新加载，仅前向重放：'+
          '；'.join(f"{x['uid']} TP={x['tp']}, FP={x['fp']}, FN={x['fn']}" for x in check['C_fresh_load_forward'])+'。','',
          '因此已保存并验证同一个模型同时完全重建两条mesh边的参数。该结论限于当前边重建任务，不等同于face或KL恢复后也能保持，也不证明任意初始化都会达到。','',
          'C最终checkpoint（服务器）：`'+summaries['C']['final_checkpoint']+'`。','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({c:{k:s[k] for k in ['small','large','both']} for c,s in summaries.items()},indent=2))
