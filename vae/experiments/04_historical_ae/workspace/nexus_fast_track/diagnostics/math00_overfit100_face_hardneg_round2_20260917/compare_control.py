"""Compare completed branches on actual reconstruction and the same old-pool loss."""
from pathlib import Path
import csv,json,shutil,statistics
ROOT=Path(__file__).resolve().parent
CONTROL=ROOT.parent/'math00_overfit100_low_lr_epoch800_900_20260917'
old_updates=[json.loads(x) for x in (CONTROL/'run/updates.jsonl').read_text().splitlines()]
new_updates=[json.loads(x) for x in (ROOT/'run/updates.jsonl').read_text().splitlines()]
assert len(old_updates)==len(new_updates)==2500
for a,b in zip(old_updates,new_updates):
    assert a['update']==b['update'] and a['lr']==b['lr']
    assert [x['uid'] for x in a['meshes']]==[x['uid'] for x in b['meshes']]
summary=[];per_mesh=[]
folder=ROOT/'control_evaluations';folder.mkdir(exist_ok=True)
for epoch in [800,825,850,875,900]:
    for branch,path in [('Control_round1_pool',CONTROL),('Fixed_round2_hardneg',ROOT)]:
        p=path/'run'/f'eval-epoch{epoch}.jsonl'
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        if path==CONTROL:shutil.copy2(p,folder/p.name)
        for r in rows:
            part=r['parts'] if path==CONTROL else r['old_pool_parts']
            per_mesh.append(dict(branch=branch,epoch=epoch,uid=r['uid'],
                old_pool_edge_soft4=part['edge'],old_pool_face_soft4=part['face'],
                edge_tp=r['edge']['tp'],edge_fp=r['edge']['fp'],edge_fn=r['edge']['fn'],
                face_tp=r['face']['tp'],face_fp=r['face']['fp'],face_fn=r['face']['fn'],
                face_complete=r['face']['complete'],missing_gt_face_candidates=r['missing_gt_face_candidates'],
                face_fn_with_candidate=r['face']['fn']-r['missing_gt_face_candidates'],
                actual_face_fp_outside_old_pool=r['face']['actual_fp_outside_training_pool'],joint_perfect=r['joint_perfect']))
        group=per_mesh[-100:];assert len(rows)==100
        summary.append(dict(branch=branch,epoch=epoch,update=rows[0]['updates'],
            joint_perfect=sum(r['joint_perfect'] for r in rows),strict_uids=[r['uid'] for r in rows if r['joint_perfect']],
            old_pool_edge_soft4_mean=statistics.mean(r['old_pool_edge_soft4'] for r in group),
            old_pool_face_soft4_mean=statistics.mean(r['old_pool_face_soft4'] for r in group),
            incomplete_faces=sum(not r['face_complete'] for r in group),
            **{k:sum(r[k] for r in group) if all(r[k] is not None for r in group) else None for k in
               ['edge_fp','edge_fn','face_fp','face_fn','missing_gt_face_candidates','face_fn_with_candidate','actual_face_fp_outside_old_pool']}))
for name,rows in [('control_comparison.csv',summary),('paired_per_mesh.csv',per_mesh)]:
    with (ROOT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(paired_uid_orders_all2500_updates_equal=True,lrs_all2500_updates_equal=True,
    old_pool_diagnostic_comparable=True,common_pool_definition='epoch800 original plus round1 fixed negatives',training_pool_losses_not_used_to_rank_branches=True,
    additional_updates_each=2500,results=summary)
(ROOT/'control_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for ax,key in zip(axes.flat,['edge_fp','edge_fn','face_fp','face_fn','joint_perfect','old_pool_face_soft4_mean']):
    for branch in ['Control_round1_pool','Fixed_round2_hardneg']:
        group=[r for r in summary if r['branch']==branch]
        ax.plot([r['epoch'] for r in group],[float('nan') if r[key] is None else r[key] for r in group],'o-',label=branch)
    ax.set_title(key);ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(ROOT/'control_comparison.png',dpi=150);plt.close(fig)
print(json.dumps(result))
