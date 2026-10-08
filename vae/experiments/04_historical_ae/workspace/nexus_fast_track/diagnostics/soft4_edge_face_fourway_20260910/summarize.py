"""Audit cumulative four-way acceptance, update boundaries and full reconstructions."""
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
full=read(OUT/'full_reconstruction.json');done=read(OUT/'complete.json');meta=read(OUT/'provenance.json')
previous=read(ROOT.parent/'soft4_edge_face_low_lr1000_20260910/continue/complete.json')
end=done['steps'];start=12100
assert start<end<=14100 and done['max_steps']==14100
assert [x['step'] for x in t]==list(range(start,end+1))
assert [x['step'] for x in full]==list(range(start,end+1,50))
assert done['final']==t[-1] and done['full_reconstruction_final']==full[-1]
assert done['optimizer_final_step']==end and done['face_head_optimizer_step']==end-6500
assert done['additional_updates']==end-start
assert done['original_sources_unchanged'] and done['frozen_and_rng_checks_passed']
assert meta['script_sha256']==hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
assert meta['start_checkpoint_sha256']=='fb5dc4d642a545bb468393905bfd09726967c6d55e9a03e23629ec00d6322c06'
assert (meta['encoder_lr'],meta['decoder_lr'],meta['face_head_lr'])==(1e-8,1e-7,1e-7)
assert meta['optimizer_restored_exactly'] and meta['loaded_weights_verified_tensor_equal']
assert meta['four_way_target']==done['four_way_target']==5
accepted=[]
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
    if 'full_reconstruction' in x:
        f=x['full_reconstruction'];flags=[]
        for i,r in enumerate(f['rows']):
            assert r['recovery_complete'] and r['face']['tp']+r['face']['fn']==[768,5146][i]
            z=r['face'];assert z['f1']==2*z['tp']/(2*z['tp']+z['fp']+z['fn'])
            flag=r['edge']['fp']==r['edge']['fn']==z['fp']==z['fn']==0
            assert r['joint_perfect']==flag;flags.append(flag)
        assert f['both_edge_and_face_perfect']==all(flags)
        event=x['step']>start and all(flags)
        assert x['four_way_perfect_evaluation']==event
        if event:
            accepted.append(x['step']);assert 'saved_checkpoint' in x
    else:
        assert x['four_way_perfect_evaluation'] is None
    assert x['four_way_perfect_steps']==accepted and x['four_way_perfect_count']==len(accepted)
    assert x['stop_criterion_reached']==(len(accepted)>=5)
    if len(accepted)>=5:
        assert x['step']==end and 'next_update_preclip_grad_l2' not in x
for x in full:
    assert x==t[x['step']-start]['full_reconstruction']
assert accepted==done['four_way_perfect_steps'] and len(accepted)==done['four_way_perfect_count']
assert done['stop_criterion_reached']==(len(accepted)==5)
assert end==14100 or (len(accepted)==5 and end==accepted[-1])
assert done['early_stop']==(end<14100)
summary=dict(start=full[0],final=full[-1],previous_final=previous['full_reconstruction_final'],
    additional_updates=end-start,stop_reason=done['stop_reason'],early_stop=done['early_stop'],
    four_way_perfect_steps=accepted,four_way_perfect_count=len(accepted),full_evaluation_count=len(full)-1,
    both_edges_perfect_updates=sum(x['I_t'] for x in t[1:]),seconds=done['seconds'],meshes={})
for i,name in enumerate(['Small','Large']):
    r0=t[0]['rows'][i];r1=t[-1]['rows'][i]
    summary['meshes'][name]=dict(face_soft4_start=r0['face_training']['loss'],face_soft4_final=r1['face_training']['loss'],
        face_bce_start=r0['face_training']['balanced_bce'],face_bce_final=r1['face_training']['balanced_bce'],
        best_face_full_evaluation=max(full[1:],key=lambda x:x['rows'][i]['face']['f1']))
    for window in [200,500]:
        tail=[x for x in full[1:] if x['step']>end-window]
        summary['meshes'][name][f'tail{window}']=dict(full_evaluations=len(tail),
            actual_face_f1_min_median_max=np.quantile([x['rows'][i]['face']['f1'] for x in tail],[0,.5,1]).tolist(),
            four_way_count=sum(x['both_edge_and_face_perfect'] for x in tail))
vpath=OUT/'checkpoint_verification.json'
if vpath.exists():
    v=read(vpath)
    assert sorted(x['step'] for x in v)==sorted(set(accepted+[end]))
    assert all(x['checkpoint_matches_saved_metrics'] and x['parameters_unchanged'] for x in v)
    summary['checkpoint_replays']=v
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'verification.json').write_text(json.dumps(dict(actual_update_count=end-start,full_evaluations=len(full),
    four_way_counts_and_stop_boundary_verified=True,loss_and_hard_reconstruction_counts_verified=True,
    no_update_after_fifth_acceptance=True if len(accepted)==5 else 'not_triggered'),indent=2)+'\n')

fig,axes=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
steps=[x['step'] for x in t];es=[x['step'] for x in full]
for i,name in enumerate(['Small','Large']):
    rows=[x['rows'][i] for x in t];faces=[x['rows'][i]['face'] for x in full]
    axes[0,i].plot(steps,[100*r['edge_f1'] for r in rows],label='Edge',lw=.8)
    axes[0,i].plot(es,[100*f['f1'] for f in faces],label='Actual Face reconstruction',marker='.',lw=1)
    axes[0,i].set(title=name,ylabel='F1 (%)')
    if i==0:axes[0,i].set(ylim=(99.9,100.01),yticks=[99.9,99.95,100])
    axes[1,i].plot(steps,[r['soft4_loss'] for r in rows],label='Edge Soft4',lw=.8)
    axes[1,i].plot(steps,[r['face_training']['loss'] for r in rows],label='Face Soft4',lw=.8)
    axes[1,i].plot(steps,[r['face_training']['balanced_bce'] for r in rows],label='Face GT-balanced BCE',lw=.8)
    axes[1,i].set(ylabel='Loss',yscale='log')
    for k,ls in [('fp','-'),('fn','--')]:
        axes[2,i].plot(es,[f[k] for f in faces],ls=ls,marker='.',label='Actual Face '+k.upper())
    axes[2,i].set(ylabel='Actual Face errors')
for ax in axes.flat:
    for s in accepted:ax.axvline(s,color='#208f40',alpha=.25,lw=1)
    ax.grid(alpha=.2);ax.set_xlabel('Global update');ax.legend(fontsize=8)
fig.suptitle(f'Step12100 + Adam | E=1e-8, D/Face=1e-7 | Edge + Face Soft4\nFour-way perfect: {len(accepted)}/5 at {len(full)-1} complete evaluations | Threshold=0')
fig.savefig(ROOT/'fourway_curve.png',dpi=150);plt.close(fig)

lines=['# B支续训：累计5次four-way perfect才提前停止','',
    '从step12100模型及完整Adam状态恢复，E LR=1e-8、Decoder/Face head LR=1e-7，Edge Soft4+Face Soft4；τ=1、ε=1e-8、membership detach、FP32归约，μ路径，KL=wd=0，clip=1及固定负例均不变。最大新增2000步至14100。', '',
    '每50步完整评估同一forward：两条mesh的Edge与Face全部FP=FN=0才计一次。累计达到5次就保存并停止，不要求连续；起点重放不计数，非完整评估步也不计数。每200步保存，并额外保存每个成功完整评估点的模型、Adam和embedding。', '',
    f"实际新增{end-start}步，结束于step{end}；停止原因：{done['stop_reason']}。four-way perfect累计{len(accepted)}/5，成功评估点：{accepted}；更新后完整评估共{len(full)-1}次。", '',
    '|阶段|Mesh|Edge FP/FN|Edge F1|Face FP/FN|Face F1|','|---|---|---|---:|---|---:|']
for stage,x in [('前次末步',previous['full_reconstruction_final']),('起点重放',full[0]),('本轮末步',full[-1])]:
    for name,r in zip(['Small','Large'],x['rows']):
        e=r['edge'];f=r['face'];lines.append(f"|{stage} {x['step']}|{name}|{e['fp']}/{e['fn']}|{100*e['f1']:.6f}%|{f['fp']}/{f['fn']}|{100*f['f1']:.6f}%|")
lines+=['',f"两条Edge同时100%：{summary['both_edges_perfect_updates']}/{end-start}次更新。它与four-way perfect分开统计，不能替代面重建验收。", '',
    '|Mesh|Face Soft4起点→结束|Face GT-balanced BCE起点→结束|最后200步实际Face F1 min/median/max|','|---|---|---|---|']
for name,r in summary['meshes'].items():
    tail=' / '.join(f'{100*v:.6f}%' for v in r['tail200']['actual_face_f1_min_median_max'])
    lines.append(f"|{name}|{r['face_soft4_start']:.7g} → {r['face_soft4_final']:.7g}|{r['face_bce_start']:.7g} → {r['face_bce_final']:.7g}|{tail}（{r['tail200']['full_evaluations']}次评估）|")
lines+=['','实际Face指标从当前预测边图枚举三角形，再按Face logit>0筛选；因漏边未进入候选的GT面也算FN。完整Face仅每50步评估，成功次数不表述为每一步成功次数。', '',
    '最终checkpoint（服务器）：`'+t[-1]['saved_checkpoint']+'`。', '',
    '原参数Adam step='+str(end)+'，Face head Adam step='+str(end-6500)+'；旧checkpoint和实验代码未修改。','']
if 'checkpoint_replays' in summary:
    lines+=['|Checkpoint重放（仅forward）|Mesh|Edge FP/FN|Face FP/FN|','|---|---|---|---|']
    for v in summary['checkpoint_replays']:
        for name,r in zip(['Small','Large'],v['replayed_full_reconstruction']['rows']):
            lines.append(f"|{v['step']}|{name}|{r['edge']['fp']}/{r['edge']['fn']}|{r['face']['fp']}/{r['face']['fn']}|")
    lines+=['','重新forward的计数与保存时计数分别保留；原Flash/CUDA数值路径的敏感性没有在本轮通过换后端消除。若计数改变，不能表述为额外训练带来的变化。','']
lines+=['本轮判断：Face Soft4与GT-balanced BCE继续下降，但在40次完整评估中，large始终有1～4个漏面，没有出现four-way perfect。最终两条Edge正确、small Face正确，large剩2个漏面；最终checkpoint重新forward仍为同样的FP/FN计数。累计5次的门槛未通过，因此按2000步预算上限结束。这个结论仅限本轮训练与评估，不能表述为已经完全过拟合，也不能据此证明模型容量不足。', '',
    '![训练与重建曲线](fourway_curve.png)','']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:summary[k] for k in ['additional_updates','stop_reason','four_way_perfect_steps','both_edges_perfect_updates','final']},indent=2))
