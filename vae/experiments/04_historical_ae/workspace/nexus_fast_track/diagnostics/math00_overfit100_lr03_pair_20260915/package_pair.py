"""Package both finished branches and compare saved structural evaluations."""
import csv,hashlib,json,statistics,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def j(p):return json.loads(p.read_text())
def jl(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
plan=j(ROOT/'paired_epoch_orders.json');all_rows={};trends=[];flat=[];updates={}
for branch in ['A_hold','B_lr03']:
    run=ROOT/branch/'run';assert j(run/'complete.json')['updates']==12500
    assert set(j(run/'complete.json')['per_mesh_participations'].values())=={500}
    u=jl(run/'updates.jsonl');assert [r['update'] for r in u]==list(range(10001,12501));updates[branch]=u
    for epoch in range(401,501):assert j(run/f'epoch-order-{epoch:03d}.json')['uids']==plan['orders'][str(epoch)]
    for epoch in [400,425,450,475,500]:
        rows=jl(run/f'eval-epoch{epoch:03d}.jsonl');assert len(rows)==100
        all_rows[branch,epoch]={r['uid']:r for r in rows}
        s=j(run/f'eval-summary-epoch{epoch:03d}.json')
        t=dict(branch=branch,epoch=epoch,update=s['updates'],joint_perfect=s['joint_perfect'],
            strict_uids=';'.join(r['uid'] for r in rows if r['joint_perfect']),
            edge_soft4_mean=statistics.mean(r['parts']['edge'] for r in rows),face_pool_soft4_mean=statistics.mean(r['parts']['face'] for r in rows),
            edge_fp=s['total_edge_fp'],edge_fn=s['total_edge_fn'],face_fp=s['total_face_fp'],face_fn=s['total_face_fn'],
            missing_gt_face_candidates=sum(r['missing_gt_face_candidates'] for r in rows),face_incomplete=len(s['face_incomplete']))
        t['face_fn_with_candidate']=t['face_fn']-t['missing_gt_face_candidates'];trends.append(t)
        for r in rows:
            d=dict(branch=branch,epoch=epoch,uid=r['uid'],participations=r['participations'],edge_soft4=r['parts']['edge'],face_pool_soft4=r['parts']['face'])
            for group in ['edge','face']:
                for k in ['tp','fp','fn','f1']:d[group+'_'+k]=r[group][k]
            d.update(joint_perfect=r['joint_perfect'],missing_gt_face_candidates=r['missing_gt_face_candidates'],face_fn_with_candidate=r['face']['fn']-r['missing_gt_face_candidates'],face_complete=r['face']['complete'],face_fp_outside_training_pool=r['face']['actual_fp_outside_training_pool'])
            flat.append(d)
for a,b in zip(updates['A_hold'],updates['B_lr03']):assert [m['uid'] for m in a['meshes']]==[m['uid'] for m in b['meshes']]
def csvfile(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
csvfile('comparison_all_checkpoints.csv',trends);csvfile('per_mesh_all_evaluations.csv',flat)
compare=[]
for epoch in [400,425,450,475,500]:
    for uid,a in all_rows['A_hold',epoch].items():
        b=all_rows['B_lr03',epoch][uid];d=dict(epoch=epoch,uid=uid)
        for group in ['edge','face']:
            for metric in ['fp','fn']:
                av=a[group][metric];bv=b[group][metric]
                d[f'{group}_{metric}_A']=av;d[f'{group}_{metric}_B']=bv;d[f'{group}_{metric}_B_minus_A']=None if av is None or bv is None else bv-av
        d.update(A_perfect=a['joint_perfect'],B_perfect=b['joint_perfect']);compare.append(d)
csvfile('per_mesh_paired_comparison.csv',compare)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
for branch in ['A_hold','B_lr03']:
    rows=[r for r in trends if r['branch']==branch];x=[r['epoch'] for r in rows]
    for ax,key in zip(axes.flat,['edge_fp','edge_fn','face_fp','face_fn','joint_perfect','face_fn_with_candidate']):
        ax.plot(x,[r[key] if r[key] is not None else float('nan') for r in rows],'o-',label=branch)
        ax.set_title(key);ax.set_xlabel('Cumulative epoch');ax.grid(alpha=.25);ax.legend()
fig.suptitle('Same epoch400 checkpoint: fixed100 LR control vs 0.3x')
fig.savefig(ROOT/'paired_structural_results.png',dpi=150);plt.close(fig)
summary=dict(state='both_completed',updates_each=2500,end_update_each=12500,
             paired_orders_all2500_updates_verified=True,results=trends,
             interpretation='Compare multiple late checkpoints, FP/FN and strict UID identities. No automatic branch selection or further training.')
(ROOT/'comparison.json').write_text(json.dumps(summary,indent=2))
files={}
for branch in ['A_hold','B_lr03']:
    with zipfile.ZipFile(ROOT/branch/'evaluation_package.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            if not name.endswith('/'):files[branch+'/'+name]=z.read(name)
for p in ROOT.iterdir():
    if p.is_file() and p.suffix in ['.py','.sh','.json','.csv','.md','.png','.log'] and p.name!='package-console.log':files[p.name]=p.read_bytes()
files['SHA256SUMS.txt']=''.join(hashlib.sha256(v).hexdigest()+'  '+k+'\n' for k,v in sorted(files.items())).encode()
out=ROOT/'pair_evaluation_package.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for name,data in files.items():z.writestr(name,data)
with zipfile.ZipFile(out) as z:assert z.testzip() is None
print(json.dumps(dict(archive=str(out),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
