"""Read saved arrays on CPU, verify reported counts, and export the evaluation bundle."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL = ROOT/'run/evaluations/update-00021280'
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
    for filename in ('eval_config.json','LAUNCHED.json'):
        shutil.copy2(ROOT/filename,OUT/'metadata'/filename)
    for filename in ('status.json','runtime.json','complete.json'):
        shutil.copy2(ROOT/'run'/filename,OUT/'metadata'/filename)
    shutil.copy2(ROOT/'eval.stdout.log',OUT/'metadata'/'eval.stdout.log')
    shutil.copy2(ROOT/'run_eval.py',OUT/'run_eval.py');shutil.copy2(__file__,OUT/'package_results.py')
    shutil.copytree(ROOT/'code',OUT/'code',ignore=shutil.ignore_patterns('__pycache__'))
    entry=result['checkpoint'];archive=Path(entry['path']).parent
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
    report=f'''# OwnAE-v2：step21280 完整100条实际重建评测

同一checkpoint评测全部100条，联合严格成功 **{joint}/100**。实际Face micro-F1为 **{f['micro_f1']:.10f}**，尚未达到0.997阶段目标，也未完成100条严格overfit。

| 项目 | TP | FP | FN | micro-F1 | 严格成功 |
|---|---:|---:|---:|---:|---:|
| Edge | {e['tp']} | {e['fp']} | {e['fn']} | {e['micro_f1']:.10f} | {len(result['edge_perfect_uids'])}/100 |
| 实际Face | {f['tp']} | {f['fp']} | {f['fn']} | {f['micro_f1']:.10f} | {len(result['face_perfect_uids'])}/100 |

Face漏检：缺边导致未入候选 {result['face_fn_missing']}；进入候选后判负 {result['face_fn_present']}。当前实际候选上的Face TN={f['tn']}；应同时看FP，不能把GT召回当成完整成功。

这是原固定100条、原生B_v2_teacher_blocks、确定性mu、Hard4、FP32 MATH路径的真实Encoder→Decoder推理。不是CAD50，不是旧72/74模型，也不是训练Face pool评价。每条Edge覆盖全部i<j pair，Face从该checkpoint的预测Edge图完整流式枚举，logit>0判正，没有补边、截断或修复。

训练从step21280完整恢复后继续在32483的GPU0运行；本次评测在36910独立GPU上执行，optimizer更新数为0。报告只属于step21280，不能转用为后续checkpoint成绩。本包不自动安排后续评测或修改训练配方。

checkpoint SHA256：`{entry['sha256']}`。完整checkpoint（含Adam/RNG）留服务器，详见server_artifacts.json。

verification.json记录CPU独立重算：逐pair Edge计数、逐分片Face计数、GT标签、候选确实来自预测Edge、候选唯一性、完整triangle数量及候选外GT FN。全部100条计数与正式结果一致；没有额外模型前向或optimizer更新。

per_mesh.csv提供逐UID错误；evaluation/包含完整评价JSON、实际预测数组及全部Face候选/logit。training_context/updates.jsonl只到父step21280，loss是每次optimizer更新前的五mesh训练指标，不能与同一checkpoint的全量实际Face指标混同。

训练检查点一共21280次有效更新，对应1064轮固定100条。此处不与旧架构或不同checkpoint拼接成功样本。
'''
    (OUT/'REPORT.md').write_text(report)
    files=[(p,p.relative_to(OUT).as_posix()) for p in sorted(OUT.rglob('*')) if p.is_file()]
    files += [(p,'evaluation/'+p.relative_to(EVAL).as_posix()) for p in sorted(EVAL.rglob('*')) if p.is_file()]
    manifest=OUT/'SHA256SUMS.txt'
    manifest.write_text(''.join(f'{digest(p)}  {name}\n' for p,name in files))
    files.append((manifest,'SHA256SUMS.txt'))
    target=ROOT/'OwnAE_V2_step21280_eval100_20260927.zip'
    with zipfile.ZipFile(target,'w',allowZip64=True,strict_timestamps=False) as z:
        for p,name in files:z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED)
    assert target.stat().st_size<900*1024**2
    with zipfile.ZipFile(target) as z:assert z.testzip() is None
    info=dict(path=str(target),bytes=target.stat().st_size,mib=target.stat().st_size/1024**2,sha256=digest(target),files=len(files),checkpoint_step=21280,all100_complete=True,independent_array_validation=True)
    write(ROOT/'PACKAGE.json',info);print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
