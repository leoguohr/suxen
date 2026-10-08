"""Compare actual structure and observed hard-negative sign changes at matched checkpoints."""
from pathlib import Path
import csv,json,statistics
import numpy as np
ROOT=Path(__file__).resolve().parent
CONTROL=ROOT.parent/'math00_overfit100_hardneg_continue100ep_20260916'
old=[json.loads(l) for l in (CONTROL/'run/updates.jsonl').read_text().splitlines()]
old=[r for r in old if r['update']>16250]
new=[json.loads(l) for l in (ROOT/'run/updates.jsonl').read_text().splitlines()]
assert len(old)==len(new)==1250
for a,b in zip(old,new):
    assert a['update']==b['update']
    assert [m['uid'] for m in a['meshes']]==[m['uid'] for m in b['meshes']]
    for name,lr in b['lr'].items():assert abs(lr-.3*a['lr'][name])<1e-18
mining=json.loads((ROOT/'mining_complete.json').read_text());meta={r['uid']:r for r in mining['records']}
summary=[];per_mesh=[];events=[];strict={}
for branch,folder,logits_dir in [('Control',ROOT/'control_evaluations',ROOT/'control_hardneg'),('LowLR',ROOT/'run',ROOT/'hardneg_logits')]:
    strict[branch]=[];previous={}
    for epoch in [650,675,700]:
        rows=[json.loads(l) for l in (folder/f'eval-epoch{epoch}.jsonl').read_text().splitlines()]
        group=[]
        for r in rows:
            u=r['uid']
            with np.load(logits_dir/f'epoch{epoch}'/f'{u}.npz') as d:ids=d['ids'];logits=d['logits']
            if branch=='LowLR' and epoch==650:
                with np.load(ROOT/'control_hardneg/epoch650'/f'{u}.npz') as d:
                    assert np.array_equal(ids,d['ids']) and np.array_equal(logits,d['logits']),u
            row=dict(branch=branch,epoch=epoch,uid=u,vertices=meta[u]['vertices'],
                edge_fp=r['edge']['fp'],edge_fn=r['edge']['fn'],face_fp=r['face']['fp'],face_fn=r['face']['fn'],
                missing_gt_face_candidates=r['missing_gt_face_candidates'],
                face_fn_with_candidate=r['face']['fn']-r['missing_gt_face_candidates'],
                joint_perfect=r['joint_perfect'],face_complete=r['face']['complete'],
                edge_soft4=r['parts']['edge'],face_soft4=r['parts']['face'],
                mined_count=len(ids),mined_pool_fp=int((logits>0).sum()),negative_to_positive=0,positive_to_negative=0)
            if u in previous:
                prev_epoch,prev_ids,prev_logits=previous[u];assert np.array_equal(ids,prev_ids)
                rise=(prev_logits<=0)&(logits>0);fall=(prev_logits>0)&(logits<=0)
                row['negative_to_positive']=int(rise.sum());row['positive_to_negative']=int(fall.sum())
                for mask,kind in [(rise,'negative_to_positive'),(fall,'positive_to_negative')]:
                    for j in np.flatnonzero(mask):events.append(dict(branch=branch,uid=u,epoch_from=prev_epoch,epoch_to=epoch,
                        i=int(ids[j,0]),j=int(ids[j,1]),k=int(ids[j,2]),logit_from=float(prev_logits[j]),logit_to=float(logits[j]),event=kind))
            previous[u]=(epoch,ids,logits);group.append(row);per_mesh.append(row)
        perfect={r['uid'] for r in rows if r['joint_perfect']};strict[branch].append(perfect)
        summary.append(dict(branch=branch,epoch=epoch,strict_count=len(perfect),strict_uids=sorted(perfect),
            edge_soft4_mean=statistics.mean(r['edge_soft4'] for r in group),face_soft4_mean=statistics.mean(r['face_soft4'] for r in group),
            incomplete_faces=sum(not r['face_complete'] for r in group),
            **{k:sum(r[k] for r in group) if all(r[k] is not None for r in group) else None for k in
               ['edge_fp','edge_fn','face_fp','face_fn','missing_gt_face_candidates','face_fn_with_candidate','mined_pool_fp','negative_to_positive','positive_to_negative']}))
for name,rows in [('matched_comparison.csv',summary),('per_mesh_comparison.csv',per_mesh),('mined_sign_changes.csv',events)]:
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(completed=True,paired_orders_all1250_updates_equal=True,only_lr_diff=.3,
    initial_mined_logits_bitwise_equal=True,results=summary,
    strict_same_uids_all3={k:sorted(set.intersection(*v)) for k,v in strict.items()},
    hardneg_recurrence_scope='Sign changes observed at epoch650,675,700 only; mined-pool FP is not the actual predicted-edge Face FP count')
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,key in zip(axes.flat,['edge_fp','edge_fn','face_fp','face_fn','strict_count','mined_pool_fp']):
    for branch in ['Control','LowLR']:
        group=[r for r in summary if r['branch']==branch]
        ax.plot([r['epoch'] for r in group],[float('nan') if r[key] is None else r[key] for r in group],'o-',label=branch)
    ax.set_title(key);ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend()
fig.savefig(ROOT/'matched_comparison.png',dpi=150);plt.close(fig)
print(json.dumps(result))
