"""Verify the Face Soft4 continuation and compare common reconstruction metrics."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'continue'
def read(p):
    return json.loads(p.read_text())
t=[json.loads(x) for x in (OUT/'trace.jsonl').read_text().splitlines()]
full=read(OUT/'full_reconstruction.json')
meta,done=read(OUT/'provenance.json'),read(OUT/'complete.json')
previous=read(ROOT.parent/'soft4_edge_face_lr600_20260910/B/complete.json')
assert [x['step'] for x in t]==list(range(11100,12101))
assert [x['step'] for x in full]==list(range(11100,12101,50))
assert done['final']==t[-1] and done['full_reconstruction_final']==full[-1]
assert done['optimizer_final_step']==12100 and done['face_head_optimizer_step']==5600
assert done['original_sources_unchanged'] and done['frozen_and_rng_checks_passed']
assert meta['script_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert (meta['encoder_lr'],meta['decoder_lr'],meta['face_head_lr'])==(1e-8,1e-7,1e-7)
assert meta['optimizer_restored_exactly'] and meta['loaded_weights_verified_tensor_equal']
assert meta['soft4_tau']==1 and meta['soft4_epsilon']==1e-8 and meta['face_weight']==1
for x in t:
    assert x['frozen_unchanged'] and x['rng_unchanged'] and x['both_meshes_in_backward']
    assert math.isclose(x['objective'],np.mean([r['soft4_loss']+r['face_training']['loss'] for r in x['rows']]),rel_tol=1e-6)
    assert x['I_t']==int(all(r['fp']==r['fn']==0 for r in x['rows']))
    for i,r in enumerate(x['rows']):
        assert r['tp']+r['fn']==[1152,7719][i] and r['tn']+r['fp']==[73153,3306306][i]
        assert r['edge_f1']==2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
        f=r['face_training']
        assert f['tp']+f['fn']==[768,5146][i] and f['tn']+f['fp']==[1152,7892][i]
        assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
        assert math.isclose(f['loss'],np.mean([g['mean'] for g in f['soft_groups'].values()]),rel_tol=1e-6)
        for g in f['soft_groups'].values():
            assert math.isclose(g['mean'],g['numerator']/(g['mass']+1e-8),rel_tol=1e-6)
for x in full:
    assert x==t[x['step']-11100]['full_reconstruction']
    for i,r in enumerate(x['rows']):
        assert r['recovery_complete']
        f=r['face']
        assert f['tp']+f['fn']==[768,5146][i]
        assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
        assert r['joint_perfect']==(r['edge']['fp']==r['edge']['fn']==f['fp']==f['fn']==0)
    assert x['both_edge_and_face_perfect']==all(r['joint_perfect'] for r in x['rows'])
summary=dict(start=full[0],final=full[-1],previous_B_final=previous['full_reconstruction_final'],
    both_edges_perfect_updates=sum(x['I_t'] for x in t[1:]),
    both_edge_face_perfect_evaluation_steps=[x['step'] for x in full[1:] if x['both_edge_and_face_perfect']],
    seconds=done['seconds'],meshes={})
for i,name in enumerate(['Small','Large']):
    summary['meshes'][name]=dict(
        face_bce_start=t[0]['rows'][i]['face_training']['balanced_bce'],
        face_bce_final=t[-1]['rows'][i]['face_training']['balanced_bce'],
        face_soft4_start=t[0]['rows'][i]['face_training']['loss'],
        face_soft4_final=t[-1]['rows'][i]['face_training']['loss'],
        best_actual_face_evaluation=max(full[1:],key=lambda x:x['rows'][i]['face']['f1'])['step'],
        tail200_face_reconstruction_f1_min_median_max=np.quantile([x['rows'][i]['face']['f1'] for x in full[-4:]],[0,.5,1]).tolist(),
        tail200_edge_f1_min_median_max=np.quantile([x['rows'][i]['edge_f1'] for x in t[-200:]],[0,.5,1]).tolist(),
        tail200_face_training_f1_min_median_max=np.quantile([x['rows'][i]['face_training']['f1'] for x in t[-200:]],[0,.5,1]).tolist())
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps(dict(all1000_updates_verified=True,all21_full_evaluations_verified=True,
    face_soft4_formula_and_counts_verified=True,model_and_three_adam_groups_restored=True),indent=2)+'\n')

fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
steps=[x['step'] for x in t];es=[x['step'] for x in full]
for i,name in enumerate(['Small','Large']):
    rows=[x['rows'][i] for x in t];faces=[x['rows'][i]['face'] for x in full]
    axes[0,i].plot(steps,[100*r['edge_f1'] for r in rows],label='Edge',lw=.8)
    axes[0,i].plot(steps,[100*r['face_training']['f1'] for r in rows],label='Face: fixed training candidates',lw=.8)
    axes[0,i].plot(es,[100*f['f1'] for f in faces],label='Face: reconstructed mesh',marker='.',lw=1)
    axes[0,i].set(title=name,ylabel='F1 (%)');axes[0,i].legend(fontsize=8)
    axes[1,i].plot(steps,[r['soft4_loss'] for r in rows],label='Edge Soft4',lw=.8)
    axes[1,i].plot(steps,[r['face_training']['loss'] for r in rows],label='Face Soft4',lw=.8)
    axes[1,i].plot(steps,[r['face_training']['balanced_bce'] for r in rows],label='Face GT-balanced BCE',lw=.8)
    axes[1,i].set(ylabel='Loss',yscale='log');axes[1,i].legend(fontsize=8)
    for k,ls in [('fp','-'),('fn','--')]:
        axes[2,i].plot(es,[f[k] for f in faces],ls=ls,marker='.',label='Face '+k.upper())
    axes[2,i].set(ylabel='Reconstructed face errors');axes[2,i].legend()
for ax in axes.flat:
    ax.grid(alpha=.2);ax.set_xlabel('Global update')
fig.suptitle('Step11100 + Adam continuation: Edge Soft4 + Face Soft4\nDetached membership, tau=1, eps=1e-8, FP32 reduction | E=1e-8, D=1e-7')
fig.savefig(ROOT/'edge_face_soft4_curve.png',dpi=150);plt.close(fig)


lines=['# B支低LR续训1000步：两条mesh Edge/Face验收','',
    '从B支step11100模型和完整三组Adam状态恢复，续训1000步至step12100。E LR=1e-8，Decoder及Face head LR=1e-7，保持Edge Soft4+Face Soft4，Face权重1，两条mesh等权完整参与。', '',
    '保持τ=1、ε=1e-8、detach membership、FP32组归约、μ模式、KL=wd=0、clip=1、原Flash后端及固定mixed面负例。判定阈值仍为0。每200步保存模型及Adam，每步记录Edge与固定训练候选Face指标，每50步完整枚举预测边图中的三角形并评分。', '',
    '|阶段|Mesh|Edge F1|实际Face F1|Face TP / FP / FN|', '|---|---|---:|---:|---|']
for stage,x in [('B原末步',previous['full_reconstruction_final']),('起点重放',full[0]),('1000步后',full[-1])]:
    for name,r in zip(['Small','Large'],x['rows']):
        f=r['face'];lines.append(f"|{stage} / {x['step']}|{name}|{100*r['edge']['f1']:.6f}%|{100*f['f1']:.6f}%|{f['tp']} / {f['fp']} / {f['fn']}|")
lines+=['',f"两条Edge同时100%：{summary['both_edges_perfect_updates']}/1000次更新。两条Edge+Face同时100%的更新后完整评估点：{summary['both_edge_face_perfect_evaluation_steps']}（共20次完整评估）。", '',
    '|Mesh|Face Soft4 起点→结束|Face GT-balanced BCE 起点→结束|最后200步实际Face F1 min/median/max（4次完整评估）|','|---|---|---|---|']
for name,r in summary['meshes'].items():
    tail=' / '.join(f'{100*v:.6f}%' for v in r['tail200_face_reconstruction_f1_min_median_max'])
    lines.append(f"|{name}|{r['face_soft4_start']:.7g} → {r['face_soft4_final']:.7g}|{r['face_bce_start']:.7g} → {r['face_bce_final']:.7g}|{tail}|")
lines+=['','实际Face验收先由预测边图枚举三角形，再以Face logit>0筛选；因漏边未进入候选的GT面也计FN。固定训练候选Face F1不能代替实际面重建。完整评估仅每50步一次，成功次数不冒充每一步的计数。', '',
    '记录原B末步、本次起点重放及最终checkpoint重放，保留原数值后端的前向差异。loss下降不能代替两条mesh同时FP=FN=0的验收。', '',
    '最终checkpoint（服务器）：`'+t[-1]['saved_checkpoint']+'`。E/D参数Adam step12100、Face head Adam step5600。旧checkpoint和实验代码未修改。','']
vpath=OUT/'checkpoint_verification.json'
if vpath.exists():
    v=read(vpath)
    assert v['checkpoint_matches_final_metrics'] and v['parameters_unchanged']
    assert v['adam_steps_by_group']==[[12100.],[12100.],[5600.]]
    lines+=['最终checkpoint仅forward重新核验，SHA256：`'+v['checkpoint_sha256']+'`。','']
    for name,r in zip(['Small','Large'],v['replayed_full_reconstruction']['rows']):
        f=r['face'];lines.append(f"- {name}：Edge FP/FN={r['edge']['fp']}/{r['edge']['fn']}，Face FP/FN={f['fp']}/{f['fn']}，Face F1={100*f['f1']:.6f}%。")
lines+=['','![训练及重建曲线](edge_face_soft4_curve.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(final=full[-1],both_edge_face_perfect_evaluation_steps=summary['both_edge_face_perfect_evaluation_steps']),indent=2))
