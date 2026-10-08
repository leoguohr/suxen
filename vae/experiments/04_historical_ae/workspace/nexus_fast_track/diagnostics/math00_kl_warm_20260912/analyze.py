"""Summarize the bounded paired KL experiment from recorded evaluations."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
names=['math00_kl_control_20260912','math00_kl_warm_20260912']
summary={};runs={}
for name in names:
    path=BASE/name
    read=lambda f:json.loads((path/f).read_text())
    rows=[json.loads(s) for s in (path/'updates.jsonl').read_text().splitlines()]
    complete=read('complete.json')
    checks=[read(f'independent_summary_step{s:04d}.json') for s in [0,20,50,100,200]]
    mu=[read(f'mu_step{s:04d}.json') for s in [0,1,10,20,50,100,200]]
    fixed=[read(f'sample_step{s:04d}.json') for s in [0,1,10,20,50,100,200]]
    groups=list(rows[0]['actual_updates'])
    initial,final=mu[0],mu[-1]
    summary[name]=dict(acceptance=complete['acceptance'],monitor=[dict(step=x['step'],perfect=x['simultaneous_perfect'],rec=x['mean_soft4'],kl=x['mean_kl']) for x in checks],final_unseen=complete['final_unseen'],mu_all_checks_perfect=all(x['four_way_perfect'] for x in mu),fixed_all_checks_perfect=all(x['four_way_perfect'] for x in fixed),initial_kl=initial['kl'],final_kl=final['kl'],kl_change={k:final['kl'][k]-initial['kl'][k] for k in ['total','mu','sigma']},initial_posterior=initial['posterior'],final_posterior=final['posterior'],gradient_probes=[dict(state=r['parameter_state_for_training_forward'],probe=r['independent_gradient_probe']) for r in rows if r['independent_gradient_probe'] is not None],clip_count=sum(r['clip_coefficient']<1 for r in rows),median_clip=float(np.median([r['clip_coefficient'] for r in rows])),nonzero_updates_by_group={g:sum(r['actual_updates'][g]['delta_l2']>0 for r in rows) for g in groups},cumulative_parameter_change=complete['cumulative_parameter_change'],final_counts=[dict(uid=r['uid'],edge=r['edge'],face=r['face']) for r in final['rec']])
    runs[name]=(rows,checks,mu)
a,b=[summary[n] for n in names]
summary['KL_minus_control_final']={k:b['final_kl'][k]-a['final_kl'][k] for k in ['total','mu','sigma']}
summary['rec_KL_minus_control_final']=b['monitor'][-1]['rec']-a['monitor'][-1]['rec']
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(2,3,figsize=(14,8))
for name,label,color in zip(names,['Control beta=0','KL warmup to 1e-4'],['#276FBF','#DE7B27']):
    rows,checks,mu=runs[name];steps=[x['step'] for x in checks]
    axes[0,0].plot(steps,[x['mean_soft4'] for x in checks],'-o',label=label,color=color)
    axes[0,1].plot([x['step'] for x in mu],[x['kl']['mu'] for x in mu],'-o',color=color)
    axes[0,2].plot([x['step'] for x in mu],[x['kl']['sigma'] for x in mu],'-o',color=color)
    for i,style in [(0,'-'),(1,'--')]:
        axes[1,0].plot([x['step'] for x in mu],[x['posterior'][i]['std']['p95'] for x in mu],style,color=color,label=f'{label}: {"small" if i==0 else "large"}')
        axes[1,1].plot([x['step'] for x in mu],[100*x['posterior'][i]['lower_clamp_fraction'] for x in mu],style,color=color)
    axes[1,2].plot(steps,[x['simultaneous_perfect'] for x in checks],'-o',color=color)
for ax,title in zip(axes.flat,['Monitoring noise: mean reconstruction Soft4','K_mu','K_sigma','Posterior sigma p95 (solid small / dashed large)','Logvar at lower clamp (%)','Simultaneous perfect / 50 monitoring draws']):
    ax.set_title(title);ax.set_xlabel('Additional updates');ax.grid(alpha=.2)
axes[0,0].legend();axes[1,2].set_ylim(0,52);axes[1,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160)
print(json.dumps({n:{k:summary[n][k] for k in ['acceptance','monitor','initial_kl','final_kl','kl_change','clip_count','median_clip']} for n in names},indent=2))
