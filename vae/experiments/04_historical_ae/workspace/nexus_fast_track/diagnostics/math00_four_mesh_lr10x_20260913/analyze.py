from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BRANCHES=['control','high10x']
LABELS={'control':'Original LR','high10x':'10x E/D/heads LR'}
COLORS=['#d46b19','#8c2a8c']
CHECKS=[0,1,10,20,50,100,200,300,400]

def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]

data={}
for branch in BRANCHES:
    root=ROOT/branch;updates=rows(root/'updates.jsonl')
    data[branch]=dict(
        manifest=read(root/'manifest.json'),complete=read(root/'complete.json'),updates=updates,
        mu=[read(root/f'mu_step{s:04d}.json') for s in CHECKS],
        fixed=[read(root/f'sample_step{s:04d}.json') for s in CHECKS],
        noise=[read(root/'independent_summary_step0000.json'),read(ROOT/'replay100'/branch/'independent_summary_step0100.json'),read(root/'independent_summary_step0200.json'),read(root/'independent_summary_step0400.json')],
        final_unseen=read(root/'final_unseen_summary_step0400.json'))

paired_epsilon=all(a['epsilon_sha256']==b['epsilon_sha256'] and a['rng_before_sha256']==b['rng_before_sha256'] and a['rng_after_sha256']==b['rng_after_sha256'] for a,b in zip(data['control']['updates'],data['high10x']['updates']))
assert paired_epsilon
assert data['control']['manifest']['parent_sha256']==data['high10x']['manifest']['parent_sha256']
assert data['control']['mu'][0]['rec']==data['high10x']['mu'][0]['rec']
uids=data['control']['manifest']['selected_uids']

summary=dict(parent_sha256=data['control']['manifest']['parent_sha256'],paired_training_epsilon_exact=paired_epsilon,checks=CHECKS,branches={})
for branch in BRANCHES:
    d=data[branch];updates=d['updates'];start=d['mu'][0];end=d['mu'][-1]
    summary['branches'][branch]=dict(
        lr=d['manifest']['lr'],
        endpoint=[dict(uid=x['uid'],edge=x['edge'],face=x['face'],gt_faces_missing_from_edge_candidates=x['gt_faces_missing_from_edge_candidates']) for x in end['rec']],
        changes=[dict(uid=uids[i],edge_tp=end['rec'][i]['edge']['tp']-start['rec'][i]['edge']['tp'],edge_fp=end['rec'][i]['edge']['fp']-start['rec'][i]['edge']['fp'],edge_fn=end['rec'][i]['edge']['fn']-start['rec'][i]['edge']['fn'],face_tp=end['rec'][i]['face']['tp']-start['rec'][i]['face']['tp'],face_fp=end['rec'][i]['face']['fp']-start['rec'][i]['face']['fp'],face_fn=end['rec'][i]['face']['fn']-start['rec'][i]['face']['fn']) for i in range(4)],
        monitor_noise=d['noise'],final_unseen=d['final_unseen'],
        reconstruction_mean_first50=float(np.mean([x['reconstruction_before_update'] for x in updates[:50]])),
        reconstruction_mean_last50=float(np.mean([x['reconstruction_before_update'] for x in updates[-50:]])),
        clipped_steps=sum(x['clip_coefficient']<1 for x in updates),median_clip=float(np.median([x['clip_coefficient'] for x in updates])),
        median_relative_update={g:float(np.median([x['actual_updates'][g]['relative_l2'] for x in updates])) for g in updates[0]['actual_updates']},
        cumulative_parameter_change=d['complete']['cumulative_parameter_change'],
        final_posterior=end['posterior'],seconds=d['complete']['seconds'])
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

fig,axes=plt.subplots(2,3,figsize=(15,8))
styles={'control':'-','high10x':'--'}
mesh_colors={2:'#d46b19',3:'#8c2a8c'}
for branch in BRANCHES:
    d=data[branch]
    for i in [2,3]:
        label=f"{LABELS[branch]} {uids[i][-6:]}"
        axes[0,0].plot(CHECKS,[100*x['rec'][i]['edge']['f1'] for x in d['mu']],styles[branch]+'o',color=mesh_colors[i],alpha=1 if branch=='high10x' else .55,label=label)
        axes[0,1].plot(CHECKS,[100*x['rec'][i]['face']['f1'] for x in d['mu']],styles[branch]+'o',color=mesh_colors[i],alpha=1 if branch=='high10x' else .55,label=label)
    axes[0,2].plot([x['update'] for x in d['updates']],[x['reconstruction_before_update'] for x in d['updates']],label=LABELS[branch],color=COLORS[BRANCHES.index(branch)],lw=1)
    axes[1,0].plot([0,100,200,400],[x['old_perfect'] for x in d['noise']],styles[branch]+'o',label=LABELS[branch],color=COLORS[BRANCHES.index(branch)])
    axes[1,1].plot([x['update'] for x in d['updates']],[x['clip_coefficient'] for x in d['updates']],label=LABELS[branch],color=COLORS[BRANCHES.index(branch)],lw=1)
    axes[1,2].plot(CHECKS,[np.mean([p['std']['p95'] for p in x['posterior']]) for x in d['mu']],styles[branch]+'o',label=LABELS[branch],color=COLORS[BRANCHES.index(branch)])
for ax,title in zip(axes.flat,['New-mesh Mu Edge F1','New-mesh Mu actual Face F1','Paired-noise training reconstruction','Old-pair strict sampled / 50','Global clip coefficient','Mean posterior sigma p95']):
    ax.set_title(title);ax.set_xlabel('Additional updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0,0].set_ylabel('%');axes[0,1].set_ylabel('%');axes[1,0].set_ylim(-1,51);axes[1,2].set_yscale('log')
fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160)

fig,axes=plt.subplots(2,2,figsize=(12,8))
for row,i in enumerate([2,3]):
    for col,kind in enumerate(['edge','face']):
        ax=axes[row,col]
        for branch,color in zip(BRANCHES,COLORS):
            d=data[branch]
            ax.plot(CHECKS,[x['rec'][i][kind]['fp'] for x in d['mu']],'-o',color=color,alpha=.8,label=LABELS[branch]+' FP')
            ax.plot(CHECKS,[x['rec'][i][kind]['fn'] for x in d['mu']],'--o',color=color,alpha=.8,label=LABELS[branch]+' FN')
        if kind=='face':
            for branch,color in zip(BRANCHES,COLORS):ax.plot(CHECKS,[x['rec'][i]['gt_faces_missing_from_edge_candidates'] for x in data[branch]['mu']],':',color=color,label=LABELS[branch]+' missing-edge face')
        ax.set_title(uids[i]+' '+kind);ax.set_yscale('symlog',linthresh=10);ax.set_xlabel('Additional updates');ax.set_ylabel('Count');ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(ROOT/'new_mesh_error_comparison.png',dpi=160)
print(json.dumps(summary,indent=2))
