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
OUT=ROOT/'joint_soft4'
def read(p):
    return json.loads(p.read_text())
t=[json.loads(x) for x in (OUT/'trace.jsonl').read_text().splitlines()]
full=read(OUT/'full_reconstruction.json')
meta,done=read(OUT/'provenance.json'),read(OUT/'complete.json')
previous=read(ROOT.parent/'soft4_restore_face_20260910/joint_resume/complete.json')
assert [x['step'] for x in t]==list(range(9500,10501))
assert [x['step'] for x in full]==list(range(9500,10501,50))
assert done['final']==t[-1] and done['full_reconstruction_final']==full[-1]
assert done['optimizer_final_step']==10500 and done['face_head_optimizer_step']==4000
assert done['original_sources_unchanged'] and done['frozen_and_rng_checks_passed']
assert meta['script_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert (meta['encoder_lr'],meta['decoder_lr'],meta['face_head_lr'])==(1e-7,1e-6,1e-6)
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
    assert x==t[x['step']-9500]['full_reconstruction']
    for i,r in enumerate(x['rows']):
        assert r['recovery_complete']
        f=r['face']
        assert f['tp']+f['fn']==[768,5146][i]
        assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
        assert r['joint_perfect']==(r['edge']['fp']==r['edge']['fn']==f['fp']==f['fn']==0)
    assert x['both_edge_and_face_perfect']==all(r['joint_perfect'] for r in x['rows'])
summary=dict(start=full[0],final=full[-1],previous_hard_face_final=previous['full_reconstruction_final'],
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
fig.suptitle('Step9500 + Adam continuation: Edge Soft4 + Face Soft4\nDetached membership, tau=1, eps=1e-8, FP32 reduction | E=1e-7, D=1e-6')
fig.savefig(ROOT/'edge_face_soft4_curve.png',dpi=150);plt.close(fig)

lines=['# Face改为Soft4：step9500续训1000步','',
    '完整恢复step9500模型和三组Adam状态，只将Face训练项由原hard four-group改为与Edge同一函数实现的Soft4。τ=1、ε=1e-8、membership detach、FP32组归约、固定除4。Face权重1，两条mesh等权参与。', '',
    'E LR=1e-7，Decoder及Face head LR=1e-6；μ路径、KL=wd=0、clip=1、原Flash后端、原固定mixed面负例均保持。最终E/D Adam step10500，Face head Adam step4000。', '',
    '|阶段|Mesh|Edge F1|实际Face F1|Face TP / FP / FN|', '|---|---|---:|---:|---|']
for stage,x in [('9500原日志',previous['full_reconstruction_final']),('9500同权重重放',full[0]),('10500结束',full[-1])]:
    for name,r in zip(['Small','Large'],x['rows']):
        f=r['face'];lines.append(f"|{stage}|{name}|{100*r['edge']['f1']:.6f}%|{100*f['f1']:.6f}%|{f['tp']} / {f['fp']} / {f['fn']}|")
lines+=['', '|Mesh|Face GT-balanced BCE：起点 → 结束|Face Soft4：起点 → 结束|','|---|---:|---:|']
for name,r in summary['meshes'].items():
    lines.append(f"|{name}|{r['face_bce_start']:.7g} → {r['face_bce_final']:.7g}|{r['face_soft4_start']:.7g} → {r['face_soft4_final']:.7g}|")
lines+=['',f"两条Edge同时100%：{summary['both_edges_perfect_updates']}/1000次更新。每50步完整评估一次，两条Edge+Face同时100%的更新后评估点：{summary['both_edge_face_perfect_evaluation_steps']}。",'',
    '实际Face指标先从预测边图枚举三角形，再由Face logit>0筛选；未进入候选的GT面也计入FN。固定训练候选上的F1不能代替此指标。完整Face并非每步枚举，因此只能报告已观察到的完整评估结果。', '',
    '同权重起点保留旧日志与本次重放，Flash/CUDA数值差异并不等同于权重或优化器重置。Face Soft4下降说明其目标在改善，但不能单独证明实际面重建改善；旧hard-four与新Soft4数值不直接比较，采用共同的GT-balanced BCE及hard reconstruction作比较。', '',
    '此次是有历史hard-Face训练的续训实验，不是Soft4从随机初始化的容量测试，也没有同步旧loss续训对照；不能据此归因所有残余错误。', '',
    'checkpoint在服务器：`'+t[-1]['saved_checkpoint']+'`。逐步日志、公式梯度校验、完整面评估、配置与hash清单已保存本地。', '',
    '本轮结果：small的实际面恢复在全部完整评估点均为100%；large换loss后先下降、随后恢复，最终1个FP、17个FN，尚未达到100%。相较本次起点重放4个FP、14个FN，总错误数仍为18；因此不能称为最终hard重建改善。最后全部GT面均进入候选，残余错误来自Face筛选而非漏边。', '',
    '![训练与重建曲线](edge_face_soft4_curve.png)','']
vpath=OUT/'checkpoint_verification.json'
if vpath.exists():
    v=read(vpath)
    assert v['checkpoint_matches_final_metrics'] and v['parameters_unchanged']
    assert v['adam_steps_by_group']==[[10500.],[10500.],[4000.]]
    for saved,replayed in zip(v['saved_full_reconstruction']['rows'],v['replayed_full_reconstruction']['rows']):
        assert saved['face']==replayed['face']
        assert all(saved['edge'][k]==replayed['edge'][k] for k in ['tp','fp','fn'])
    lines+=['最终checkpoint已重新加载、仅forward核验，两条mesh的Edge/Face TP、FP、FN均与训练末步一致；参数和文件hash保持不变。', '',
        '最终checkpoint SHA256：`'+v['checkpoint_sha256']+'`。','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k not in ['start','final','previous_hard_face_final']},indent=2))
