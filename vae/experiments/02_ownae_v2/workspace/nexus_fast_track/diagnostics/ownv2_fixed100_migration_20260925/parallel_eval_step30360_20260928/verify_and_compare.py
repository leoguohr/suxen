"""Read saved arrays on CPU, verify reported counts, and export the evaluation bundle."""
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
EVAL = ROOT/'run/evaluations/update-00030360'
OUT = ROOT/'analysis'


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
    baseline=json.loads((ROOT/'baseline-step24920.json').read_text())
    assert baseline['complete'] and [m['uid'] for m in baseline['meshes']]==uids
    assert baseline['identity']['manifest_sha256']==result['identity']['manifest_sha256']
    old=set(baseline['perfect_uids']);new=set(result['perfect_uids'])
    comparison=dict(baseline_step=24920,evaluated_step=30360,
        edge_before=baseline['edge'],edge_after=result['edge'],
        face_before=baseline['face'],face_after=result['face'],
        edge_strict_before=len(baseline['edge_perfect_uids']),edge_strict_after=len(result['edge_perfect_uids']),
        face_strict_before=len(baseline['face_perfect_uids']),face_strict_after=len(result['face_perfect_uids']),
        joint_before=len(old),joint_after=len(new),retained=sorted(old&new),lost=sorted(old-new),added=sorted(new-old),
        face_fn_missing=result['face_fn_missing'],face_fn_present=result['face_fn_present'],
        stage_Face_micro_F1_at_least_0_997=result['face']['micro_f1']>=0.997,
        checkpoint=result['checkpoint'],all100_complete=True,raw_arrays_independently_validated=True)
    write(OUT/'comparison.json',comparison)
    old_by_uid={m['uid']:m for m in baseline['meshes']}
    changes=[]
    for m in result['meshes']:
        a=old_by_uid[m['uid']]
        changes.append(dict(uid=m['uid'],vertices=m['vertices'],edge_before=a['edge'],edge_after=m['edge'],face_before=a['face'],face_after=m['face'],joint_before=m['uid'] in old,joint_after=m['uid'] in new))
    write(OUT/'per_mesh_changes.json',changes)
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':main()
