from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
read=lambda name:json.loads((ROOT/name).read_text())
manifest=read('manifest.json');uids=manifest['selected_uids']
checks=[read(f'mu_step{s:04d}.json') for s in manifest['checks']]
fixed=[read(f'sample_step{s:04d}.json') for s in manifest['checks']]
updates=[json.loads(s) for s in (ROOT/'updates.jsonl').read_text().splitlines()]
noise=[];error_details={}
for step in manifest['noise_check_steps']:
 rows=[json.loads(s) for s in (ROOT/f'independent_step{step:04d}.jsonl').read_text().splitlines()]
 summary=read(f'independent_summary_step{step:04d}.json')
 summary.update(old_perfect=sum(r['old_meshes_perfect'] for r in rows),new_perfect=sum(r['new_meshes_perfect'] for r in rows),per_mesh_perfect={uid:sum(all(r['rec'][i][kind]['fp']==r['rec'][i][kind]['fn']==0 for kind in ['edge','face']) for r in rows) for i,uid in enumerate(uids)})
 noise.append(summary)
 error_details[str(step)]=[{k:v for k,v in r.items() if k in ['pair','seeds','rec','old_meshes_perfect','new_meshes_perfect','all_meshes_perfect']} for r in rows]
summary=dict(uids=uids,initial=checks[0],final=checks[-1],fixed_initial=fixed[0],fixed_final=fixed[-1],noise=noise,complete=read('complete.json'),per_mesh_loss_change=[dict(uid=uid,initial=checks[0]['parts'][i],final=checks[-1]['parts'][i]) for i,uid in enumerate(uids)],median_group_gradients={g:float(np.median([x['gradient_norms_preclip'][g] for x in updates])) for g in updates[0]['gradient_norms_preclip']},median_clip=float(np.median([x['clip_coefficient'] for x in updates])),clipped_steps=sum(x['clip_coefficient']<1 for x in updates),nonzero_update_counts={g:sum(x['actual_updates'][g]['delta_l2']>0 for x in updates) for g in updates[0]['actual_updates']})
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'noise_error_details.json').write_text(json.dumps(error_details,indent=2))
fig,axes=plt.subplots(2,3,figsize=(15,8))
colors=['#1769aa','#07815e','#e07a17','#a0409e'];steps=[x['step'] for x in checks]
for i,(uid,color) in enumerate(zip(uids,colors)):
 label=('old ' if i<2 else 'new ')+uid[-6:];style='-' if i<2 else '--'
 for j,kind in enumerate(['edge','face']):axes[0,j].plot(steps,[100*x['rec'][i][kind]['f1'] for x in checks],style+'o',color=color,label=label)
 axes[0,2].plot([x['update'] for x in updates],[sum(x['parts_before_update'][i].values()) for x in updates],color=color,label=label,lw=1)
 axes[1,0].plot(steps,[x['posterior'][i]['std']['p95'] for x in checks],style+'o',color=color,label=label)
for key,label,col in [('old_perfect','Old pair','#1769aa'),('new_perfect','New pair','#e07a17'),('simultaneous_perfect','All four','#222222')]:axes[1,1].plot([x['step'] for x in noise],[x[key] for x in noise],'-o',label=label,color=col)
axes[1,2].plot([x['update'] for x in updates],[x['clip_coefficient'] for x in updates],color='#555555')
for ax,title in zip(axes.flat,['Mu Edge F1 (%)','Mu actual Face F1 (%)','Per-mesh training reconstruction Soft4','Posterior sigma p95','Strict sampled reconstruction / 50','Global clip coefficient']):ax.set_title(title);ax.set_xlabel('Additional updates');ax.grid(alpha=.2)
axes[0,0].legend();axes[1,1].legend();axes[0,2].set_yscale('log');axes[1,0].set_yscale('log');axes[1,1].set_ylim(-1,51)
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160)
print(json.dumps(dict(initial=[dict(uid=x['uid'],edge=x['edge'],face=x['face']) for x in checks[0]['rec']],final=[dict(uid=x['uid'],edge=x['edge'],face=x['face']) for x in checks[-1]['rec']],noise=[{k:x[k] for k in ['step','simultaneous_perfect','old_perfect','new_perfect','per_mesh_perfect']} for x in noise]),indent=2))
# Separate suppression of false positives from recovery of true topology.
fig,axes=plt.subplots(2,2,figsize=(11,7))
for row,i in enumerate([2,3]):
 for col,kind in enumerate(['edge','face']):
  ax=axes[row,col]
  for metric,color in [('fp','#d16c20'),('fn','#8c2678')]:
   ax.plot(steps,[x['rec'][i][kind][metric] for x in checks],'-o',label=metric.upper(),color=color)
  if kind=='face':ax.plot(steps,[x['rec'][i]['gt_faces_missing_from_edge_candidates'] for x in checks],'--o',label='GT faces absent from predicted edge graph',color='#347c3a')
  ax.set_title(uids[i]+' '+kind);ax.set_yscale('symlog',linthresh=10);ax.set_xlabel('Additional updates');ax.set_ylabel('Count (symlog)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(ROOT/'new_mesh_errors.png',dpi=160)
summary['median_relative_updates']={g:float(np.median([x['actual_updates'][g]['relative_l2'] for x in updates])) for g in updates[0]['actual_updates']}
summary['face_FN_decomposition']=[dict(uid=uid,trajectory=[dict(step=x['step'],face_FN=x['rec'][i]['face']['fn'],missing_edge_triangle=x['rec'][i]['gt_faces_missing_from_edge_candidates'],present_but_rejected=x['rec'][i]['face']['fn']-x['rec'][i]['gt_faces_missing_from_edge_candidates']) for x in checks]) for i,uid in enumerate(uids)]
summary['first_last_100_training_reconstruction']=[float(np.mean([x['reconstruction_before_update'] for x in batch])) for batch in [updates[:100],updates[-100:]]]
summary['training_perturbation_nonzero_every_mesh_every_step']=all(p['actual_noise_nonzero_elements']>0 for x in updates for p in x['posterior_before_update'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
