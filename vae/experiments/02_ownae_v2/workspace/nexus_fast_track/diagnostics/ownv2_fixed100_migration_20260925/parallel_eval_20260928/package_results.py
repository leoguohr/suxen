"""Read saved arrays on CPU, verify reported counts, and export the evaluation bundle."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL = ROOT/'run/evaluations/update-00024920'
OUT = ROOT/'delivery'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def main():
    result=json.loads((EVAL/'complete.json').read_text())
    runtime=json.loads((ROOT/'run/runtime.json').read_text())
    assert result['complete'] and result['optimizer_updates_during_evaluation']==0
    uids=runtime['data']['uids']; assert len(uids)==len(set(uids))==100
    assert [m['uid'] for m in result['meshes']]==uids
    OUT.mkdir(exist_ok=False)
    (OUT/'metadata').mkdir(); (OUT/'training_context').mkdir()
    rows=[]; totals={t:{k:0 for k in ('tp','fp','fn','tn')} for t in ('edge','face')}
    for m in result['meshes']:
        uid=m['uid']; folder=EVAL/uid
        assert m['identity']['checkpoint_sha256']==result['identity']['checkpoint_sha256']
        meta=json.loads((folder/'network-output.json').read_text())
        assert digest(folder/'network-output.npz')==meta['sha256']
        with np.load(folder/'network-output.npz',allow_pickle=False) as a:
            gt=a['edge_labels']; pred=a['edge_logits']>0
            edge=dict(tp=int((gt&pred).sum()),fp=int((~gt&pred).sum()),fn=int((gt&~pred).sum()),tn=int((~gt&~pred).sum()))
            assert edge==m['edge']
            assert np.array_equal(a['all_edge_pairs'][pred],a['predicted_edges'])
            n=len(a['vertices']); faces=a['gt_faces'].copy(); nfaces=len(faces)
            gtkeys=(faces[:,0]*n+faces[:,1])*n+faces[:,2]
            adj=np.zeros((n,n),bool);pe=a['predicted_edges'];adj[pe[:,0],pe[:,1]]=True
            covered=adj[faces[:,0],faces[:,1]]&adj[faces[:,0],faces[:,2]]&adj[faces[:,1],faces[:,2]]
            missing=int((~covered).sum())
        state=json.loads((folder/'face_shards/progress.json').read_text());assert state['complete']
        tp=fp=tn=candidates=0;last_key=-1
        for i in range(state['shards']):
            with np.load(folder/'face_shards'/f'part-{i:08d}.npz',allow_pickle=False) as a:
                ids=a['ids'].astype(np.int64);logits=a['logits'];labels=a['labels'];prediction=logits>0
                keys=(ids[:,0]*n+ids[:,1])*n+ids[:,2]
                assert np.isfinite(logits).all() and (np.diff(ids,axis=1)>0).all()
                assert (np.diff(keys)>0).all() and (not len(keys) or keys[0]>last_key)
                if len(keys):last_key=int(keys[-1])
                assert np.array_equal(labels,np.isin(keys,gtkeys))
                assert (adj[ids[:,0],ids[:,1]]&adj[ids[:,0],ids[:,2]]&adj[ids[:,1],ids[:,2]]).all()
                candidates+=len(ids);tp+=int((labels&prediction).sum());fp+=int((~labels&prediction).sum());tn+=int((~labels&~prediction).sum())
        # Independent triangle count, without the evaluator's chunk cursor.
        expected_candidates=sum(int(np.count_nonzero(adj[i]&adj[j])) for i,j in pe)
        assert candidates==expected_candidates==m['actual_face_candidates']
        face=dict(tp=tp,fp=fp,fn=nfaces-tp,tn=tn);assert face==m['face']
        assert missing==m['face_fn_missing'] and face['fn']-missing==m['face_fn_present']
        for t,counts in [('edge',edge),('face',face)]:
            for k in counts:totals[t][k]+=counts[k]
        ep=edge['fp']==edge['fn']==0;fp0=face['fp']==face['fn']==0
        rows.append(dict(uid=uid,vertices=n,edge_tp=edge['tp'],edge_fp=edge['fp'],edge_fn=edge['fn'],face_tp=face['tp'],face_fp=face['fp'],face_fn=face['fn'],face_fn_missing=missing,face_fn_present=face['fn']-missing,actual_face_candidates=candidates,edge_strict=ep,face_strict=fp0,joint_strict=ep and fp0))
    for task,counts in totals.items():
        assert counts=={k:result[task][k] for k in counts}
        f1=2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn'])
        assert f1==result[task]['micro_f1']
    assert sum(r['joint_strict'] for r in rows)==result['joint_perfect']
    with (OUT/'per_mesh.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(OUT/'verification.json',dict(checkpoint=result['checkpoint'],uids=100,
        original_npz_hashes_match=True,edge_counts_recomputed_from_all_pairs=True,
        face_counts_recomputed_from_all_shards=True,face_labels_and_edge_membership_checked=True,
        complete_triangle_counts_independently_verified=True,strict_sorted_unique_candidates=True,
        missing_GT_faces_counted_as_FN=True,aggregate_matches=True,
        additional_model_forwards=0,optimizer_updates=0))
    for filename in ('eval_config.json','LAUNCHED.json','PREPARED.json','baseline-step21280.json'):
        shutil.copy2(ROOT/filename,OUT/'metadata'/filename)
    for filename in ('status.json','runtime.json','complete.json'):
        shutil.copy2(ROOT/'run'/filename,OUT/'metadata'/filename)
    shutil.copy2(ROOT/'eval.stdout.log',OUT/'metadata'/'eval.stdout.log')
    shutil.copy2(ROOT/'run_eval.py',OUT/'run_eval.py');shutil.copy2(__file__,OUT/'package_results.py')
    shutil.copytree(ROOT/'code',OUT/'code',ignore=shutil.ignore_patterns('__pycache__'))
    entry=result['checkpoint'];archive=ROOT/'training_context_snapshot'
    for filename in ('config.json','runtime.json','budget.json','updates.jsonl'):
        shutil.copy2(archive/filename,OUT/'training_context'/filename)
    source=Path(runtime['data']['source'])
    for filename in ('selection.json','overfit100_manifest.csv'):
        shutil.copy2(source/filename,OUT/'metadata'/filename)
    write(OUT/'server_artifacts.json',dict(full_checkpoint=entry,
        evaluation_root=str(ROOT),training_root=json.loads((ROOT/'eval_config.json').read_text())['training_source_root'],
        excluded=['Full 2.96GB model/Adam/RNG checkpoint (server path, size and SHA256 above). Historical experiments and raw source mesh archives are not duplicated.'],
        included_inputs='Each network-output.npz includes the exact vertices, GT edges and GT faces used for this evaluation.',
        prediction_format='network-output.npz stores all Edge pair scores and predicted Edge ids. face_shards/part-*.npz stores every actual triangle id and Face logit; predicted Faces are rows with logits>0.'))
    e=result['edge'];f=result['face'];joint=result['joint_perfect']
    baseline=json.loads((ROOT/'baseline-step21280.json').read_text())
    assert baseline['complete'] and [m['uid'] for m in baseline['meshes']]==uids
    assert baseline['identity']['manifest_sha256']==result['identity']['manifest_sha256']
    prior={m['uid']:m for m in baseline['meshes']}
    before=set(baseline['perfect_uids']);after=set(result['perfect_uids'])
    comparison=dict(baseline_step=21280,evaluated_step=24920,
        added=sorted(after-before),lost=sorted(before-after),retained=sorted(before&after),
        joint_before=len(before),joint_after=len(after),
        edge_before=baseline['edge'],edge_after=result['edge'],
        face_before=baseline['face'],face_after=result['face'],
        edge_micro_f1_delta=result['edge']['micro_f1']-baseline['edge']['micro_f1'],
        face_micro_f1_delta=result['face']['micro_f1']-baseline['face']['micro_f1'])
    write(OUT/'comparison.json',comparison)
    changes=[]
    for row in rows:
        old=prior[row['uid']]
        change=dict(uid=row['uid'],vertices=row['vertices'])
        for task in ('edge','face'):
            for key in ('fp','fn'):
                field=task+'_'+key
                change[field+'_before']=old[task][key]
                change[field+'_after']=row[field]
                change[field+'_delta']=row[field]-old[task][key]
        change['joint_before']=row['uid'] in before
        change['joint_after']=row['uid'] in after
        changes.append(change)
    with (OUT/'per_mesh_comparison.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    threshold='已达到' if f['micro_f1']>=0.997 else '尚未达到'
    report=f"""# OwnAE-v2：step24920 完整100条实际重建评测

同一个checkpoint完整评测原固定100条。联合严格成功 **{joint}/100**，相比step21280的12/100，保留{len(before&after)}条、丢失{len(before-after)}条、新增{len(after-before)}条。实际Face micro-F1为 **{f['micro_f1']:.10f}**，{threshold}0.997阶段目标。阶段F1与全100条严格零错误是不同标准。

| 项目 | step21280 | step24920 |
|---|---:|---:|
| Edge FP / FN | {baseline['edge']['fp']} / {baseline['edge']['fn']} | {e['fp']} / {e['fn']} |
| 实际Face FP / FN | {baseline['face']['fp']} / {baseline['face']['fn']} | {f['fp']} / {f['fn']} |
| Edge micro-F1 | {baseline['edge']['micro_f1']:.10f} | {e['micro_f1']:.10f} |
| 实际Face micro-F1 | {baseline['face']['micro_f1']:.10f} | {f['micro_f1']:.10f} |
| Edge严格成功 | {len(baseline['edge_perfect_uids'])}/100 | {len(result['edge_perfect_uids'])}/100 |
| 实际Face严格成功 | {len(baseline['face_perfect_uids'])}/100 | {len(result['face_perfect_uids'])}/100 |
| 联合严格成功 | {baseline['joint_perfect']}/100 | {joint}/100 |

Face漏检分解：缺边未入候选{result['face_fn_missing']}；已入候选但判负{result['face_fn_present']}。Face TP={f['tp']}，实际候选TN={f['tn']}。逐UID增减及成功集合见comparison.json、per_mesh_comparison.csv，不拼接不同checkpoint的成功样本。

本次实际执行为原生B_v2_teacher_blocks、确定性mu、FP32 MATH路径的真实Encoder→mu→全部Decoder推理。训练配方为Hard4。没有optimizer更新，也没有修改训练进程。全部i<j pair参与Edge评价；Face从当前预测Edge图完整去重、流式枚举triangle，logit>0判正。未入候选的GT Face计FN，没有补GT边、候选截断或mesh修复。

评测位于36910独立GPU；原训练在32483 GPU0继续。成绩仅属于冻结的step24920，不能称为后续训练步数的成绩。本次不自动安排另一轮评测或改训练配方。

完整checkpoint路径、大小、SHA256见server_artifacts.json。SHA256：`{entry['sha256']}`。文件包含model/AdamW/RNG，通过硬链接固定于独立source目录，训练轮换恢复文件不会更换该文件内容。

verification.json记录CPU独立重算：所有Edge pair混淆计数、所有Face分片混淆计数、GT标签、预测Edge成员关系、候选排序和唯一性、独立triangle总量与候选外GT FN。全部100条均匹配；这一步没有追加网络前向。

包内evaluation/含逐mesh真实输入、全部Edge pair/logit、预测Edge、完整实际Face候选/logit；实际预测Face是logit>0的候选。per_mesh.csv为汇总。code/为checkpoint对应的实际源码，run_eval.py为本次入口。metadata/含配置、进程记录及step21280基线结果。

training_context/updates.jsonl为完整1—24920步日志；训练loss来自更新前的五mesh，不能用它代替同一checkpoint的全100条结构评价。context中的runtime/budget/status是冻结时的运行快照，其中动态状态可能略晚于step24920，模型身份以source checkpoint及其哈希为准。本模型24920有效更新对应1246轮固定100条，每轮20次五mesh更新。

包不重复收入2.96GB完整checkpoint或全部旧实验；相应模型已留在服务器。全部实际输入数组和此次重建预测均在包内。
"""
    (OUT/'REPORT.md').write_text(report)
    files=[(p,p.relative_to(OUT).as_posix()) for p in sorted(OUT.rglob('*')) if p.is_file()]
    files += [(p,'evaluation/'+p.relative_to(EVAL).as_posix()) for p in sorted(EVAL.rglob('*')) if p.is_file()]
    manifest=OUT/'SHA256SUMS.txt'
    manifest.write_text(''.join(f'{digest(p)}  {name}\n' for p,name in files))
    files.append((manifest,'SHA256SUMS.txt'))
    target=ROOT/'OwnAE_V2_step24920_eval100_20260928.zip'
    with zipfile.ZipFile(target,'w',allowZip64=True,strict_timestamps=False) as z:
        for p,name in files:z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    assert target.stat().st_size<900*1024**2
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    info=dict(path=str(target),bytes=target.stat().st_size,mib=target.stat().st_size/1024**2,sha256=digest(target),files=len(files),checkpoint_step=24920,all100_complete=True,independent_array_validation=True)
    write(ROOT/'PACKAGE.json',info);print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
