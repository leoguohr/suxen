"""Validate all 600 updates in each branch and compare actual reconstructions."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def read(p):
    return json.loads(p.read_text())
def q(v):
    return np.quantile(v,[0,.5,1]).tolist()

traces={};fulls={};metas={};summary={}
script_sha=hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
for branch in ['A','B']:
    out=ROOT/branch
    t=[json.loads(x) for x in (out/'trace.jsonl').read_text().splitlines()]
    full=read(out/'full_reconstruction.json');meta=read(out/'provenance.json');done=read(out/'complete.json')
    assert [x['step'] for x in t]==list(range(10500,11101))
    assert [x['step'] for x in full]==list(range(10500,11101,50))
    assert done['additional_updates']==600 and done['optimizer_final_step']==11100 and done['face_head_optimizer_step']==4600
    assert done['final']==t[-1] and done['full_reconstruction_final']==full[-1]
    assert done['original_sources_unchanged'] and done['frozen_and_rng_checks_passed']
    assert meta['script_sha256']==script_sha and meta['optimizer_restored_exactly'] and meta['loaded_weights_verified_tensor_equal']
    assert meta['start_checkpoint_sha256']=='b12e8f031e2bf355c2ee5628937fc0beab9b2dc1b5b48552fea000b14c5ec0f2'
    lrs=[1e-7,1e-6] if branch=='A' else [1e-8,1e-7]
    assert [meta['encoder_lr'],meta['decoder_lr']]==lrs and meta['face_head_lr']==lrs[1]
    assert meta['soft4_tau']==1 and meta['soft4_epsilon']==1e-8 and meta['face_weight']==1
    for x in t:
        assert [x['encoder_lr'],x['decoder_lr']]==lrs
        assert x['both_meshes_in_backward'] and x['frozen_unchanged'] and x['rng_unchanged']
        assert x['I_t']==int(all(r['fp']==r['fn']==0 for r in x['rows']))
        assert math.isclose(x['objective'],np.mean([r['soft4_loss']+r['face_training']['loss'] for r in x['rows']]),rel_tol=1e-6)
        for i,r in enumerate(x['rows']):
            assert r['tp']+r['fn']==[1152,7719][i] and r['fp']+r['tn']==[73153,3306306][i]
            assert r['edge_f1']==2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
            f=r['face_training']
            assert f['tp']+f['fn']==[768,5146][i] and f['fp']+f['tn']==[1152,7892][i]
            assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
            assert math.isclose(f['loss'],np.mean([g['mean'] for g in f['soft_groups'].values()]),rel_tol=1e-6)
    for x in full:
        assert x==t[x['step']-10500]['full_reconstruction']
        for i,r in enumerate(x['rows']):
            assert r['recovery_complete']
            f=r['face'];assert f['tp']+f['fn']==[768,5146][i]
            assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
            assert r['joint_perfect']==(r['edge']['fp']==r['edge']['fn']==f['fp']==f['fn']==0)
        assert x['both_edge_and_face_perfect']==all(r['joint_perfect'] for r in x['rows'])
    assert done['both_count']==sum(x['I_t'] for x in t[1:])
    perfect_steps=[x['step'] for x in full[1:] if x['both_edge_and_face_perfect']]
    summary[branch]=dict(encoder_lr=lrs[0],decoder_and_face_lr=lrs[1],updates=600,seconds=done['seconds'],
        start=full[0],final=full[-1],training_start=t[0]['rows'],training_final=t[-1]['rows'],
        both_edges_perfect_count=done['both_count'],both_edges_longest_streak=done['longest_both_streak'],
        both_edge_face_perfect_evaluation_steps=perfect_steps,
        per_mesh={})
    for i,name in enumerate(['Small','Large']):
        summary[branch]['per_mesh'][name]=dict(
            edge_perfect_update_count=sum(x['rows'][i]['is_perfect'] for x in t[1:]),
            face_training_perfect_update_count=sum(x['rows'][i]['face_training']['fp']==x['rows'][i]['face_training']['fn']==0 for x in t[1:]),
            face_reconstruction_perfect_evaluation_steps=[x['step'] for x in full[1:] if x['rows'][i]['face']['fp']==x['rows'][i]['face']['fn']==0],
            best_reconstruction_evaluation=max(full[1:],key=lambda x:x['rows'][i]['face']['f1']),
            tail200_edge_f1=q([x['rows'][i]['edge_f1'] for x in t[-200:]]),
            tail200_face_training_f1=q([x['rows'][i]['face_training']['f1'] for x in t[-200:]]),
            tail200_face_reconstruction_f1_at_4_evaluations=q([x['rows'][i]['face']['f1'] for x in full[-4:]]))
    vpath=out/'checkpoint_verification.json'
    if vpath.exists():
        v=read(vpath)
        assert v['checkpoint_matches_final_metrics'] and v['parameters_unchanged']
        assert v['adam_steps_by_group']==[[11100.],[11100.],[4600.]]
        summary[branch]['checkpoint_replay']=v
    traces[branch]=t;fulls[branch]=full;metas[branch]=meta

for key in ['start_checkpoint_sha256','initial_sha256','source_sha256','loss_script_sha256','script_sha256',
            'soft4_tau','soft4_epsilon','soft4_membership','soft4_group_reduction','face_loss','face_weight',
            'face_negatives','loss_reduction','mode','clip_norm','weight_decay','selected_uids','full_face_evaluation_period']:
    assert metas['A'][key]==metas['B'][key],key
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps(dict(both600_updates_verified=True,all26_full_evaluations_verified=True,
    same_start_checkpoint_and_three_adam_states=True,only_learning_rates_differ=True,
    losses_and_hard_metrics_checked=True),indent=2)+'\n')

fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
for i,name in enumerate(['Small','Large']):
    for branch,color in [('A','#cc7040'),('B','#177b9b')]:
        t=traces[branch];full=fulls[branch];steps=[x['step']-10500 for x in t];es=[x['step']-10500 for x in full]
        rows=[x['rows'][i] for x in t];faces=[x['rows'][i]['face'] for x in full]
        axes[0,i].plot(steps,[100*r['edge_f1'] for r in rows],ls=':',lw=.8,color=color,label=branch+' Edge')
        axes[0,i].plot(es,[100*f['f1'] for f in faces],marker='.',color=color,label=branch+' reconstructed Face')
        axes[1,i].plot(steps,[r['face_training']['loss'] for r in rows],lw=.8,color=color,label=branch+' Face Soft4')
        axes[1,i].plot(steps,[r['face_training']['balanced_bce'] for r in rows],ls='--',lw=.8,color=color,label=branch+' Face GT-balanced BCE')
        for k,ls in [('fp','-'),('fn','--')]:
            axes[2,i].plot(es,[f[k] for f in faces],ls=ls,marker='.',color=color,label=branch+' Face '+k.upper())
    axes[0,i].set(title=name,ylabel='F1 (%)')
    axes[1,i].set(ylabel='Face loss',yscale='log')
    axes[2,i].set(ylabel='Actual reconstruction errors')
for ax in axes.flat:
    ax.grid(alpha=.2);ax.set_xlabel('Updates after step10500');ax.legend(fontsize=8)
fig.suptitle('600 updates each from identical step10500 + Adam | Edge Soft4 + Face Soft4\nA: E=1e-7, D/Face=1e-6 | B: E=1e-8, D/Face=1e-7 | Threshold remains 0')
fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)

lines=['# step10500：Edge+Face Soft4，学习率A/B各600步','',
    'A：E=1e-7，Decoder与Face head=1e-6；B：E=1e-8，Decoder与Face head=1e-7。均独立从相同step10500 checkpoint及全部三组Adam状态恢复，先核对状态张量完全相同，再设置分支LR。两条mesh每步完整参与，Edge Soft4 + Face Soft4，每条mesh等权，Face权重1。', '',
    'τ=1，ε=1e-8，membership detach，FP32组归约，μ路径，KL=wd=0，clip=1，同一后端、数据顺序和固定mixed面负例，threshold=0。GPU0依次跑A、B，各600步，不提前停止。每200步保存模型和Adam；每步记录Edge与固定训练候选Face，每50步完整评估实际面重建。', '',
    '|分支 / 阶段|Mesh|Edge F1|实际Face F1|Face TP / FP / FN|','|---|---|---:|---:|---|']
for branch,z in summary.items():
    for stage in ['start','final']:
        for name,r in zip(['Small','Large'],z[stage]['rows']):
            f=r['face'];lines.append(f"|{branch} / {z[stage]['step']}|{name}|{100*r['edge']['f1']:.6f}%|{100*f['f1']:.6f}%|{f['tp']} / {f['fp']} / {f['fn']}|")
lines+=['','|分支|两条Edge同时100%的更新次数|两条Edge+Face同时100%的完整评估点|耗时|','|---|---:|---|---:|']
for branch,z in summary.items():
    lines.append(f"|{branch}|{z['both_edges_perfect_count']}/600|{z['both_edge_face_perfect_evaluation_steps']}（共12次更新后评估）|{z['seconds']/60:.2f}分钟|")
lines+=['','|分支 / Mesh|Face Soft4 起点 → 终点|Face GT-balanced BCE 起点 → 终点|最后200步实际Face F1 min / median / max（4次完整评估）|','|---|---:|---:|---|']
for branch,z in summary.items():
    for i,name in enumerate(['Small','Large']):
        a=z['training_start'][i]['face_training'];b=z['training_final'][i]['face_training']
        tail=z['per_mesh'][name]['tail200_face_reconstruction_f1_at_4_evaluations']
        lines.append(f"|{branch} / {name}|{a['loss']:.7g} → {b['loss']:.7g}|{a['balanced_bce']:.7g} → {b['balanced_bce']:.7g}|"+' / '.join(f'{100*v:.6f}%' for v in tail)+'|')
lines+=['','每次实际Face评估复用当步训练forward的embedding，从当前预测边图枚举三角形、按Face logit>0筛选；GT面未进入边图候选也计FN。训练候选Face F1与实际重建不同，完整Face成功次数不能表述为逐步成功次数。', '',
    '相同起点指参数和Adam状态张量经逐项验证相同；既有Flash/CUDA数值路径不保证重新forward逐位一致。因此A/B起点重放分别保存，与原step10500日志分开。这里只改变LR，没有用阈值扫描最优值进行验收。', '',
    '最终checkpoint（服务器）：','',
    '- `/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_lr600_20260910/A/checkpoint-11100.pt`',
    '- `/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_lr600_20260910/B/checkpoint-11100.pt`','',
    '已有E/D参数Adam step11100，Face head Adam step4600。原始checkpoint和旧实验代码均未修改。启动前曾修正调度脚本与Python标准库queue的命名冲突，失败启动未生成训练trace、未做参数更新。','']
for branch,z in summary.items():
    if 'checkpoint_replay' in z:
        v=z['checkpoint_replay'];lines += [f"{branch}最终checkpoint重新加载、仅forward复查（参数未更新），SHA256：`{v['checkpoint_sha256']}`。",'']
        for name,r in zip(['Small','Large'],v['replayed_full_reconstruction']['rows']):
            f=r['face'];lines.append(f"- {name}：Edge FP/FN={r['edge']['fp']}/{r['edge']['fn']}，Face FP/FN={f['fp']}/{f['fn']}，Face F1={100*f['f1']:.6f}%。")
        lines.append('')
lines+=['本轮判断：B的两条Edge同时100%次数为581/600，优于A的456/600；但Face并未表现出同样明确的优势。B终点少2个漏面，A最后200步的4次完整评估Face F1中位数略高，且A最好的评估点为5个漏面，B为7个。两个分支均未出现完整边+面同时100%。', '',
    'A的large Face Soft4和GT-balanced BCE终值更低，说明继续原LR仍在改善目标；B降低LR主要观察到边稳定性改善。仅凭此次600步实验，不能认定降低十倍LR已经解决Face残余错误，也不能由终点2个面的差距宣称B在Face上明确胜出。', '',
    '最终checkpoint重新forward后large分别为A：0 FP/8 FN，B：0 FP/7 FN，与训练末步的10/8个FN有所差异。已有数值敏感性仍存在，报告同时保留末步与重放计数，不能把这次重新forward当作额外训练带来的进步。', '',
    '![对照曲线](comparison.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:dict(final=v['final'],both_edge_face_perfect_evaluation_steps=v['both_edge_face_perfect_evaluation_steps']) for k,v in summary.items()},indent=2))
