"""Compare saved full-reconstruction checks; no forward or optimizer updates."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
branches={}
for name in ['A_hold','B_lr03']:
    d=r/name
    assert json.loads((d/'completion.json').read_text())['additional_updates']==500
    logs=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
    assert [x['update'] for x in logs]==list(range(1,501))
    checks=[json.loads(p.read_text()) for p in sorted(d.glob('eval-update*.json'))]
    assert [x['update'] for x in checks]==[0,100,200,300,400,500]
    branches[name]=dict(logs=logs,checks=checks)
a,b=[branches[x] for x in ['A_hold','B_lr03']]
for key in ['parts','edge','face','min_margin','gt_face_candidates','scales']:
    assert a['checks'][0][key]==b['checks'][0][key]
assert a['logs'][0]['gradient_norms']==b['logs'][0]['gradient_norms']
ratio={g:b['logs'][0]['actual_updates'][g]['delta_l2']/a['logs'][0]['actual_updates'][g]['delta_l2'] for g in a['logs'][0]['actual_updates']}
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for name,data in branches.items():
    label='Hold LR' if name=='A_hold' else 'LR x0.3'
    logs,checks=data['logs'],data['checks']
    x=[t['update'] for t in logs];cx=[t['update'] for t in checks]
    for ax,key in zip(axes[0],['edge_soft4','face_soft4']):ax.plot(x,[t[key] for t in logs],label=label,lw=1)
    for ax,kind in zip(axes[1],['edge','face']):
        ax.plot(cx,[t[kind]['fp'] for t in checks],marker='o',label=label+' FP')
        ax.plot(cx,[t[kind]['fn'] for t in checks],marker='x',ls='--',label=label+' FN')
for ax,title in zip(axes.flat,['Edge Soft4 (before update)','Face Soft4 (before update)',
                              'Actual Edge errors (after update)','Actual Face errors (after update)']):
    ax.set_title(title);ax.set_xlabel('Additional updates from step3000');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Same latent512 checkpoint + Adam / mu / Soft4 / one full804 mesh')
fig.savefig(r/'comparison_curves.png',dpi=180);plt.close(fig)
result=dict(parent_sha256='06dc42726c816b69f06db353d4de505e9d8bceda864294c8bb75c7907ba0bdd6',
            first_update_gradient_norms_equal=True,first_actual_update_ratio_B_over_A=ratio,branches={})
result['B_all_four_error_counts_lower_at_all_followup_checks']=all(
    y[k][m]<x[k][m] for x,y in zip(a['checks'][1:],b['checks'][1:]) for k in ['edge','face'] for m in ['fp','fn'])
for name,data in branches.items():
    result['branches'][name]=dict(checks=data['checks'],
        strict_perfect_checks=[x['update'] for x in data['checks'] if x['perfect']],
        median_update_seconds=float(np.median([x['seconds'] for x in data['logs']])),
        clipped_updates=sum(x['clip_coefficient']<1 for x in data['logs']),
        all_groups_update_each_step=all(all(y['delta_l2']>0 for y in x['actual_updates'].values()) for x in data['logs']),
        all512_channels_have_gradient=all(x['interface']['mu_gradient_nonzero_channels']==512 for x in data['logs']))
(r/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# 512维 step3000 后期学习率对照','',
       '两支各新增500次更新，均从同一step3000权重、四组Adam和RNG出发。A保持原LR；B四个重建组LR均乘0.3。无warmup、sampling、KL或其他训练改动。',
       '', '| 新增更新 | 分支 | Edge TP/FP/FN | 实际Face TP/FP/FN | GT Face候选覆盖 | Edge/Face Soft4 |',
       '| --- | --- | --- | --- | --- | --- |']
for i in range(6):
    for name,data in branches.items():
        t=data['checks'][i]
        counts=lambda key:'/'.join(str(t[key][x]) for x in ['tp','fp','fn'])
        lines.append(f'| {t["update"]} | {name} | {counts("edge")} | {counts("face")} | {t["gt_face_candidates"]}/1604 | {t["parts"][0]["edge"]:.6f}/{t["parts"][0]["face"]:.6f} |')
lines+=['','Face由各次预测Edge图完整枚举；未入候选的GT Face仍计FN。训练Face pool准确率不替代实际Face验收。',
        '',f'首步两支梯度范数完全相同。四组实际参数位移B/A：{ratio}。',
        '', '这是固定500步预算下的LR对照，不能将短程结果推广为长期最优LR；没有自动续训。',
        '', '完整checkpoint位于服务器同名实验目录下的A_hold和B_lr03；本地日志包不包含大型checkpoint。',
        '', '![对照曲线](comparison_curves.png)']
ae=sum(a['checks'][-1][k][m] for k in ['edge','face'] for m in ['fp','fn'])
be=sum(b['checks'][-1][k][m] for k in ['edge','face'] for m in ['fp','fn'])
lines+=['',f'结果：本轮所有新增100/200/300/400/500检查点，B的Edge FP、Edge FN、Face FP、Face FN均低于A：{result["B_all_four_error_counts_lower_at_all_followup_checks"]}。',
        f'末尾边/面FP+FN合计：A={ae}，B={be}，B少{(1-be/ae)*100:.2f}%。主要收益来自减少多余边和多余面。',
        '两支均没有严格全对检查点。本次结果支持当前工作点上将重建LR乘0.3更有效地清除错误，但不证明后续必然收敛或该LR长期最优。',
        'B的Face loss仍有偶发尖峰，最差margin也并非全部改善；不将更少的错误概括为完全无波动或所有候选同时改善。',
        '每支500次新增更新已完成并停止；没有加权、loss、sampling/KL、架构或optimizer-reset的额外干预。']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({name:data['checks'][-1] for name,data in branches.items()},indent=2))
