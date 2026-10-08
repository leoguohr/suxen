"""Check training components against actual edge-gated face reconstruction."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'joint'
UIDS=['nexus_2k_000387','nexus_2k_001849']
def read(p):
    return json.loads(p.read_text())

old_t=[json.loads(line) for line in (OUT/'trace.jsonl').read_text().splitlines()]
old_full=read(OUT/'full_reconstruction.json')
old_meta,old_done=read(OUT/'provenance.json'),read(OUT/'complete.json')
new_out=ROOT/'joint_resume'
new_t=[json.loads(line) for line in (new_out/'trace.jsonl').read_text().splitlines()]
new_full=read(new_out/'full_reconstruction.json')
meta,done=read(new_out/'provenance.json'),read(new_out/'complete.json')
assert [x['step'] for x in old_t]==list(range(6500,7501))
assert [x['step'] for x in new_t]==list(range(7500,9501))
assert old_done['final']==old_t[-1] and done['final']==new_t[-1]
assert old_done['optimizer_final_step']==7500 and old_done['face_head_optimizer_step']==1000
assert done['optimizer_final_step']==9500 and done['face_head_optimizer_step']==3000
assert old_meta['script_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert meta['script_sha256']==hashlib.sha256((ROOT/'resume.py').read_bytes()).hexdigest()
for m,d in [(old_meta,old_done),(meta,done)]:
    assert m['face_weight']==1 and (m['encoder_lr'],m['decoder_lr'],m['face_head_lr'])==(1e-7,1e-6,1e-6)
    assert m['optimizer_restored_exactly'] and m['loaded_weights_verified_tensor_equal']
    assert d['original_sources_unchanged'] and d['frozen_and_rng_checks_passed']
for k in ['face_loss','face_weight','face_negatives','loss_reduction','soft4_tau','soft4_membership','soft4_group_reduction',
          'source_sha256','threshold','mode','clip_norm','weight_decay','encoder_lr','decoder_lr','face_head_lr']:
    assert old_meta[k]==meta[k],k
t=old_t[:-1]+new_t
full=old_full[:-1]+new_full
assert [x['step'] for x in t]==list(range(6500,9501))
assert [x['step'] for x in full]==list(range(6500,9501,50))
assert done['full_reconstruction_final']==full[-1]
for x in t:
    assert x['both_meshes_in_backward'] and x['frozen_unchanged'] and x['rng_unchanged']
    assert math.isclose(x['objective'],np.mean([r['soft4_loss']+r['face_training']['loss'] for r in x['rows']]),rel_tol=1e-6)
    for i,r in enumerate(x['rows']):
        assert r['uid']==UIDS[i]
        assert r['tp']+r['fn']==[1152,7719][i] and r['fp']+r['tn']==[73153,3306306][i]
        assert r['edge_f1']==2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
        f=r['face_training']
        assert f['tp']+f['fn']==[768,5146][i] and f['fp']+f['tn']==[1152,7892][i]
        assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
        assert math.isclose(f['loss'],np.mean([g['mean_bce'] for g in f['groups'].values() if g['count']]),rel_tol=1e-6)
for x in full:
    assert x==t[x['step']-6500]['full_reconstruction']
    for i,r in enumerate(x['rows']):
        if r['recovery_complete']:
            f=r['face'];assert f['tp']+f['fn']==[768,5146][i]
            assert f['f1']==2*f['tp']/(2*f['tp']+f['fp']+f['fn'])
            assert r['joint_perfect']==(r['edge']['fp']==r['edge']['fn']==f['fp']==f['fn']==0)
    assert x['both_edge_and_face_perfect']==all(r['joint_perfect'] for r in x['rows'])
updates=t[1:]
summary=dict(start=full[0],final=full[-1],training_final=t[-1],
    both_edge_and_face_perfect_evaluation_steps=[x['step'] for x in full[1:] if x['both_edge_and_face_perfect']],
    both_edges_perfect_update_count=sum(x['I_t'] for x in updates),
    tail200={uid:dict(edge_f1=np.quantile([x['rows'][i]['edge_f1'] for x in t[-200:]],[0,.5,1]).tolist(),
                     face_training_f1=np.quantile([x['rows'][i]['face_training']['f1'] for x in t[-200:]],[0,.5,1]).tolist())
             for i,uid in enumerate(UIDS)},seconds=old_done['seconds']+done['seconds'],
    old_step7500=old_full[-1],replayed_step7500=new_full[0])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'combined_trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in t))
(ROOT/'verification.json').write_text(json.dumps(dict(all3000_updates_verified=True,
    all61_full_reconstruction_evaluations_verified=True,old_adam_state_restored_new_face_state_added=True,
    face_loss_is_original_hard_four_group_not_soft4=True,loss_and_metric_counts_verified=True),indent=2)+'\n')

fig,axes=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
steps=[x['step']-6500 for x in t];eval_steps=[x['step']-6500 for x in full]
for i,name in enumerate(['Small','Large']):
    rows=[x['rows'][i] for x in t]
    axes[0,i].plot(steps,[100*r['edge_f1'] for r in rows],label='Edge',color='#16887a',lw=.8)
    axes[0,i].plot(steps,[100*r['face_training']['f1'] for r in rows],label='Face: fixed training candidates',color='#ca9733',lw=.8)
    faces=[x['rows'][i].get('face') for x in full]
    axes[0,i].plot(eval_steps,[100*f['f1'] if f else np.nan for f in faces],label='Face: actual reconstructed mesh',color='#bd4149',marker='.',lw=1)
    axes[0,i].set(title=name,ylabel='F1 (%)',ylim=(-1,101));axes[0,i].legend(fontsize=8)
    axes[1,i].plot(steps,[r['soft4_loss'] for r in rows],label='Edge Soft4',color='#16887a',lw=.8)
    axes[1,i].plot(steps,[r['face_training']['loss'] for r in rows],label='Original Face four-group',color='#bd4149',lw=.8)
    axes[1,i].plot(steps,[r['face_training']['balanced_bce'] for r in rows],label='Face GT-balanced BCE (diagnostic)',color='#ca9733',lw=.8)
    axes[1,i].set(ylabel='Loss',yscale='log');axes[1,i].legend(fontsize=8)
    for k,style in [('fp','-'),('fn','--')]:
        axes[2,i].plot(eval_steps,[f[k] if f else np.nan for f in faces],ls=style,marker='.',label='Face '+k.upper())
    axes[2,i].set(ylabel='Actual reconstruction FP / FN',yscale='symlog');axes[2,i].legend()
for ax in axes.flat:
    ax.grid(alpha=.2);ax.set_xlabel('Updates after restoring Face loss')
fig.suptitle('Restore Face from perfect-edge checkpoint: deterministic mu, KL=0\nEdge Soft4 + original hard four-group Face | E=1e-7, D and face head=1e-6')
fig.savefig(ROOT/'edge_face_curve.png',dpi=150);plt.close(fig)

lines=['# 恢复Face loss：Edge Soft4 + 原Face四组损失','',
       '起点为C分支step6500，两条edge均已完全重建。继续deterministic μ、KL=wd=0、clip=1、E LR=1e-7、D LR=1e-6。两条mesh共同反传，L=mean_mesh(EdgeSoft4+Face_original)，Face权重1。', '',
       '**本轮Soft4仅用于Edge；恢复的Face项仍是原hard TP/TN/FP/FN非空组均值，并未改成Face Soft4。** 使用旧edge+face诊断的固定mixed负例：small1152个、large7892个；所有GT faces均作为正例，不重新抽样。', '',
       '旧参数恢复Adam动量与step6500；此前冻结的face head解除冻结，保留原权重、只新建该head的Adam状态，学习率同Decoder。先训练1000步，再完整恢复三组Adam状态、相同配置续训2000步；最终旧组Adam step9500、face head step3000。', '',
       '|阶段 / mesh|Edge F1|实际Face F1|Face TP / FP / FN|边候选图漏掉的GT面|','|---|---:|---:|---|---:|']
for stage,x in [('起点',full[0]),('Face训练1000步后',old_full[-1]),('Face训练3000步后',full[-1])]:
    for name,r in zip(['Small','Large'],x['rows']):
        f=r.get('face')
        if f:lines.append(f"|{stage} / {name}|{100*r['edge']['f1']:.6f}%|{100*f['f1']:.6f}%|{f['tp']} / {f['fp']} / {f['fn']}|{f['gt_faces_missing_from_edge_candidates']}|")
        else:lines.append(f"|{stage} / {name}|{100*r['edge']['f1']:.6f}%|评估未完成|—|—|")
lines += ['',f"新增3000步中两条Edge同时100%的次数：{summary['both_edges_perfect_update_count']}。每50步检查完整edge+face恢复，成功的更新后评估点：{summary['both_edge_and_face_perfect_evaluation_steps']}。", '',
          '|最终训练候选指标|Edge Soft4|Face原四组loss|Face GT-balanced BCE|候选Face F1|',
          '|---|---:|---:|---:|---:|']
for name,r in zip(['Small','Large'],t[-1]['rows']):
    f=r['face_training'];lines.append(f"|{name}|{r['soft4_loss']:.8g}|{f['loss']:.8g}|{f['balanced_bce']:.8g}|{100*f['f1']:.6f}%|")
lines += ['', '实际Face评估从当前预测边图枚举三角形候选，再按face logit>0筛选；GT面若因漏边未进入候选，也计入FN。训练候选上的Face F1单独报告，不能代替最终mesh的面重建。每次完整评估复用该步训练前向的embedding，未另采样latent。', '',
          ('本轮完整评估已出现两条edge+face同时完全重建；达到过与持续保持仍应区分。' if summary['both_edge_and_face_perfect_evaluation_steps'] else '3000步内的完整评估尚未通过联合验收，不等于证明模型没有面重建容量；原Face损失、固定负例覆盖、新启用face head的优化和共享参数变化仍需分别判断。'), '',
          '原代码与旧checkpoint不修改。每200步保存模型及Adam，完整日志和评估记录已取回本地。', '',
          '最终checkpoint（服务器）：`'+new_t[-1]['saved_checkpoint']+'`。已有E/D参数Adam step9500，face head Adam step3000。', '',
          '最终两条Edge正确，small的Face正确，large仍有4个错面和12个漏面；最终所有GT面均进入边图候选，剩余错误来自Face筛选。训练过程中两条Edge同时100%的比例为2861/3000，不能表述为加入Face后边完全没有受到扰动。', '',
          '![Edge与Face曲线](edge_face_curve.png)', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary,indent=2))
