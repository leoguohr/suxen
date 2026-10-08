"""Read-only error identity comparison across six complete reconstruction checks."""
from pathlib import Path
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

r=Path(__file__).resolve().parent
steps=[0,100,200,300,400,500]
done=json.loads((r/'completion.json').read_text())
assert done['completed_updates']==5000 and done['additional_updates']==500
logs=[json.loads(x) for x in (r/'updates.jsonl').read_text().splitlines()]
assert [x['update'] for x in logs]==list(range(1,501))
checks=[json.loads((r/f'eval-update{s:04d}.json').read_text()) for s in steps]
errors=[json.loads((r/f'errors-update{s:04d}.json').read_text())['errors'] for s in steps]
def key(x): return (x['kind'],tuple(x['id']))
error_sets=[set(map(key,x)) for x in errors]
union=set.union(*error_sets)
trajectory=[]
for s,check,errs in zip(steps,checks,errors):
    d=np.load(r/f'candidate-snapshot-update{s:04d}.npz')
    tables={}
    for ids,y,l in zip(d['edge_pairs'],d['edge_labels'],d['edge_logits']):
        tables[('edge',tuple(ids))]=(int(y),float(l),True,True)
    for ids,y,l,pool in zip(d['face_actual_triples'],d['face_actual_labels'],d['face_actual_logits'],d['face_actual_in_training_pool']):
        tables[('face',tuple(ids))]=(int(y),float(l),True,bool(pool))
    for ids,l,actual in zip(d['face_gt_triples'],d['face_gt_logits'],d['face_gt_in_actual_candidates']):
        tables[('face',tuple(ids))]=(1,float(l),bool(actual),True)
    # Recompute threshold counts from the saved scoring arrays, not from the error labels.
    ey=d['edge_labels'];el=d['edge_logits'];fy=d['face_actual_labels'];fl=d['face_actual_logits']
    assert int(((el>0)&~ey).sum())==check['edge']['fp']
    assert int(((el<=0)&ey).sum())==check['edge']['fn']
    assert int(((fl>0)&~fy).sum())==check['face']['fp']
    assert 1604-int(((fl>0)&fy).sum())==check['face']['fn']
    train_keys=set(map(int,d['face_train_keys']))
    for e in errs:
        y,l,actual,pool=tables[key(e)]
        assert y==e['label'] and l==e['logit'] and actual==e['in_actual_candidates'] and pool==e['in_training_pool']
    for k in sorted(union):
        kind,ids=k
        if k in tables:
            y,l,actual,pool=tables[k]
        else:
            # A formerly false-positive non-GT face may leave the predicted edge graph.
            # It is no longer an actual output candidate; its score was not evaluated here.
            assert kind=='face'
            y,l,actual,pool=0,None,False,int(ids[0]*804**2+ids[1]*804+ids[2]) in train_keys
        predicted=actual and l is not None and l>0
        trajectory.append(dict(update=s,cumulative_update=4500+s,kind=kind,id='-'.join(map(str,ids)),
            label=y,logit=l,signed_margin=None if l is None else (2*y-1)*l,
            in_actual_candidates=actual,in_training_pool=pool,error=bool(predicted!=bool(y)),
            cause=('missing_edge_candidate' if y and not actual else 'classifier_negative' if y and not predicted
                   else 'false_positive' if predicted and not y else 'correct')))
with (r/'error_identity_trajectory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(trajectory[0]) if trajectory else ['update']);w.writeheader();w.writerows(trajectory)
summary={}
for kind,error in [('edge','FP'),('edge','FN'),('face','FP'),('face','FN')]:
    sets=[{tuple(x['id']) for x in es if x['kind']==kind and x['error']==error} for es in errors]
    final=sets[-1]
    summary[kind+'_'+error]=dict(counts=[len(x) for x in sets],unique_error_ids=len(set.union(*sets)),
        present_at_all_six=len(set.intersection(*sets)),final_also_wrong_at_start=len(final&sets[0]),
        final_wrong_at_last_three=len(final&sets[-2]&sets[-3]),
        final_not_wrong_at_start=len(final-sets[0]),
        transitions=[dict(from_step=steps[i-1],to_step=steps[i],resolved=len(sets[i-1]-sets[i]),
                          newly_wrong=len(sets[i]-sets[i-1]),persisting=len(sets[i-1]&sets[i])) for i in range(1,6)])
summary['final_face_fp_training_pool']=dict(
    inside=sum(e['kind']=='face' and e['error']=='FP' and e['in_training_pool'] for e in errors[-1]),
    outside=sum(e['kind']=='face' and e['error']=='FP' and not e['in_training_pool'] for e in errors[-1]))
summary['final_face_fn_causes']={cause:sum(e.get('cause')==cause for e in errors[-1]) for cause in ['missing_edge_candidate','classifier_negative']}
summary['interpretation']='Recurrence is measured at six checkpoints only; not evidence of persistence at every intermediate update. Absent non-GT Face candidate logit is null, not zero.'
summary['snapshot_counts_and_error_logits_verified']=True
(r/'error_identity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,k in zip(axes[0],['edge_soft4','face_soft4']):
    ax.plot([x['cumulative_update']-1 for x in logs],[x[k] for x in logs],lw=.8);ax.set_title(k+' (before update)')
for ax,k in zip(axes[1],['edge','face']):
    for m in ['fp','fn']:ax.plot([x['cumulative_update'] for x in checks],[x[k][m] for x in checks],marker='o',label=m.upper())
    ax.set_title('Actual '+k+' errors');ax.legend()
for ax in axes.flat:ax.set_xlabel('Cumulative updates');ax.grid(alpha=.2)
fig.suptitle('804 / latent512 / unchanged B LR / mu continuation 4500 to 5000')
fig.savefig(r/'training_curves.png',dpi=170);plt.close(fig)
lines=['# 512维B配置：累计4500至5000续训','',
       '新增500步，完整保留父权重、四组Adam、RNG和LR；μ、KL=0、logvar冻结、math00、fully-diff Soft4、候选与评分均不变。',
       '', '| 新增步数 | Edge TP/FP/FN | 实际Face TP/FP/FN | GT Face候选覆盖 | Edge/Face Soft4 |',
       '| --- | --- | --- | --- | --- |']
for x in checks:
    cnt=lambda k:'/'.join(str(x[k][m]) for m in ['tp','fp','fn'])
    lines.append(f'| {x["update"]} | {cnt("edge")} | {cnt("face")} | {x["gt_face_candidates"]}/1604 | {x["parts"][0]["edge"]:.6f}/{x["parts"][0]["face"]:.6f} |')
lines+=['',f'严格成功检查点：{done["perfect_checks"]}。已到预算停止，没有自动延长。',
        '', '## 错误身份追踪', '',
        '| 类型 | 六点出现过的不同ID | 六点均错 | 末尾错误 | 末尾也在起点错 | 末三点均错 |',
        '| --- | --- | --- | --- | --- | --- |']
for k in ['edge_FP','edge_FN','face_FP','face_FN']:
    x=summary[k];lines.append(f'| {k} | {x["unique_error_ids"]} | {x["present_at_all_six"]} | {x["counts"][-1]} | {x["final_also_wrong_at_start"]} | {x["final_wrong_at_last_three"]} |')
lines+=['',f'末尾Face FP训练pool归属：{summary["final_face_fp_training_pool"]}。',
        f'末尾Face FN来源：{summary["final_face_fn_causes"]}。',
        '六点都错只代表在这些检查点持续错误，不代表每一训练步都错。已离开实际候选的非GT Face，其logit记为空值，不伪造为0；这类错误消失可能来自边图变化。',
        '完整逐候选轨迹见error_identity_trajectory.csv；errors-update*.json保存当次错误的ID/label/logit/训练pool归属；candidate-snapshot-update*.npz保存原始评分快照。',
        '', '![训练曲线](training_curves.png)']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
