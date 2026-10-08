"""Archive saved continuation evidence; never run the model."""
from pathlib import Path
import csv,hashlib,json,zipfile,statistics
ROOT=Path(__file__).resolve().parent
run=ROOT/'run'
rows=[]
for p in sorted(run.glob('eval-epoch*.jsonl')):
    for line in p.read_text().splitlines():
        r=json.loads(line)
        row=dict(epoch=r['epoch'],update=r['updates'],uid=r['uid'],participations=r['participations'],
                 edge_soft4=r['parts']['edge'],face_soft4=r['parts']['face'])
        for group in ['edge','face']:
            for metric in ['tp','fp','fn','f1']:row[group+'_'+metric]=r[group].get(metric)
        row.update(face_complete=r['face']['complete'],missing_gt_face_candidates=r['missing_gt_face_candidates'],
                   face_fp_outside_training_pool=r['face']['actual_fp_outside_training_pool'],joint_perfect=r['joint_perfect'],
                   old_pool_edge_soft4=r['old_pool_parts']['edge'],old_pool_face_soft4=r['old_pool_parts']['face'])
        rows.append(row)
if rows:
    with (ROOT/'per_mesh_evaluation.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    trend=[]
    for epoch in sorted({r['epoch'] for r in rows}):
        group=[r for r in rows if r['epoch']==epoch]
        trend.append(dict(epoch=epoch,update=group[0]['update'],meshes=len(group),
            edge_soft4_mean=statistics.mean(r['edge_soft4'] for r in group),
            face_soft4_mean=statistics.mean(r['face_soft4'] for r in group),
            old_pool_edge_soft4_mean=statistics.mean(r['old_pool_edge_soft4'] for r in group),
            old_pool_face_soft4_mean=statistics.mean(r['old_pool_face_soft4'] for r in group),
            joint_perfect=sum(r['joint_perfect'] for r in group),
            **{k:sum(r[k] for r in group) if all(r[k] is not None for r in group) else None
               for k in ['edge_fp','edge_fn','face_fp','face_fn']}))
    with (ROOT/'evaluation_trend.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(trend[0]));w.writeheader();w.writerows(trend)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    x=[r['update'] for r in trend]
    for k,label in [('edge_soft4_mean','Edge'),('face_soft4_mean','Face training pool')]:
        axes[0].plot(x,[r[k] for r in trend],'o-',label=label)
    for k in ['edge_fp','edge_fn','face_fp','face_fn']:
        axes[1].plot(x,[r[k] if r[k] is not None else float('nan') for r in trend],'o-',label=k)
    axes[1].set_yscale('symlog')
    axes[2].plot(x,[r['joint_perfect'] for r in trend],'o-',label='Joint strict pass /100')
    axes[2].set_ylim(-1,101)
    for a,title in zip(axes,['Mean evaluation Soft4','Actual reconstruction errors','Strict reconstruction']):
        a.set_title(title);a.set_xlabel('Cumulative optimizer updates');a.grid(alpha=.25);a.legend(fontsize=8)
    fig.savefig(ROOT/'evaluation_trend.png',dpi=150);plt.close(fig)
state=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'startup_failed'}
if (run/'failure.json').exists():state['failure']=json.loads((run/'failure.json').read_text())
exit_record=json.loads((ROOT/'runner_exit.json').read_text()) if (ROOT/'runner_exit.json').exists() else {'train_exit':None}
(ROOT/'package_status.json').write_text(json.dumps(dict(training_status=state,exit=exit_record,
    scope='Saved results/code/configuration only. Large checkpoint, raw mesh and pool arrays remain on server.'),indent=2))
files=[]
for directory in [ROOT,run,ROOT/'source_archive',ROOT/'review_runtime',ROOT/'C_graph_only']:
    candidates=directory.iterdir() if directory in [ROOT,run] else directory.rglob('*')
    for p in candidates:
        if p.is_file() and not p.name.startswith('._') and p.suffix in ['.py','.sh','.json','.jsonl','.csv','.md','.txt','.log','.png'] and p.name not in ['SHA256SUMS.txt','package-console.log']:
            files.append(p)
files.extend((ROOT/'mined').glob('*.npz'))
files.extend((ROOT/'control_evaluations').glob('*.jsonl'))
for folder in [ROOT/'hardneg_logits',ROOT/'control_hardneg']:
    files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.npz','.json','.jsonl'])
files=sorted(set(files))
(ROOT/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
dest=ROOT/'evaluation_package.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[ROOT/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(ROOT)))
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest())))
