"""Mine fixed, capped pool-external Face false positives at B epoch500. No updates."""
import json,time,fcntl
from pathlib import Path
from runtime import ROOT,BASE,T,np,c,b,setup,write,sha,tensor_hash
from evaluate import evaluate_mesh,FACE_CHUNK

PARENT=BASE/'diagnostics/math00_overfit100_lr03_pair_20260915/B_lr03'
SOURCE=PARENT/'run/checkpoint-update12500.pt'
SOURCE_SHA='dc28a6e72a8e30a84d4536cbd09168d48635d66c5a924a3f52ef907c36fc0038'

@T.no_grad()
def mine_faces(rows,pool,scales):
    n=len(pool['vertices']);gt=pool['positive'];limit=len(gt)
    oldkeys=c.m.probe.keys(np.concatenate([gt,pool['mixed']]),n)
    gtkeys=c.m.probe.keys(gt,n)
    adj=np.zeros((n,n),dtype=bool)
    for lo in range(0,n,64):
        ids=np.array([(i,j) for i in range(lo,min(n,lo+64)) for j in range(i+1,n)],dtype=np.int64).reshape(-1,2)
        logits=c.edge_logits(rows[2][0],T.as_tensor(ids,device='cuda'),scales).cpu().numpy()
        assert np.isfinite(logits).all();pred=logits>0
        adj[ids[pred,0],ids[pred,1]]=True
    best=np.empty((0,3),dtype=np.int64);scores=np.empty(0,dtype=np.float32)
    eligible=scored=0;pending=[]
    def score(tri):
        nonlocal best,scores,eligible,scored
        tri=np.asarray(tri,dtype=np.int64).reshape(-1,3);keys=c.m.probe.keys(tri,n)
        logits=c.face_logits(rows[3][0],T.as_tensor(tri,device='cuda'),scales).cpu().numpy()
        assert np.isfinite(logits).all()
        keep=(logits>0)&~np.isin(keys,gtkeys)&~np.isin(keys,oldkeys)
        eligible+=int(keep.sum());scored+=len(tri)
        best=np.concatenate([best,tri[keep]]);scores=np.concatenate([scores,logits[keep]])
        # Global top-K over all candidates, with stable candidate-ID tie breaking.
        order=np.lexsort((c.m.probe.keys(best,n),-scores))[:limit]
        best=best[order];scores=scores[order]
    for i in range(n):
        for j in np.flatnonzero(adj[i]):
            kk=np.flatnonzero(adj[i]&adj[j])
            if len(kk):pending.extend((i,int(j),int(k)) for k in kk)
            while len(pending)>=FACE_CHUNK:
                score(pending[:FACE_CHUNK]);pending=pending[FACE_CHUNK:]
    if pending:score(pending)
    keys=c.m.probe.keys(best,n)
    assert len(best)==min(limit,eligible) and len(np.unique(keys))==len(keys)
    assert not np.isin(keys,oldkeys).any() and not np.isin(keys,gtkeys).any()
    assert (scores>0).all()
    assert all(adj[i,j] and adj[i,k] and adj[j,k] for i,j,k in best)
    return best,scores,dict(eligible_outside_pool_fp=eligible,selected=len(best),cap=limit,
        actual_candidates_scored=scored,enumeration_complete=True,
        selected_logit_max=float(scores.max()) if len(scores) else None,
        selected_logit_min=float(scores.min()) if len(scores) else None)

def main():
    lock=(ROOT/'mining.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'mining_complete.json').exists(),'Refuse remine completed fixed pool'
    assert sha(SOURCE)==SOURCE_SHA
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    assert cp['epoch']==500 and cp['completed_updates']==12500
    model,opt,groups,uids,pools,forward,capture,args=setup()
    model.load_state_dict(cp['model'],strict=True)
    before=tensor_hash(model.named_parameters());del cp
    scales=model.scoring_contract()
    old={r['uid']:r for r in map(json.loads,(PARENT/'run/eval-epoch500.jsonl').read_text().splitlines())}
    (ROOT/'augmented_pools').mkdir(exist_ok=True);(ROOT/'mined').mkdir(exist_ok=True)
    records=[]
    with (ROOT/'mining.jsonl').open('x',buffering=1) as f:
        for i,u in enumerate(uids):
            write(ROOT/'mining_status.json',dict(state='mining',completed_meshes=i,uid=u))
            t=time.monotonic();rows=forward(u)
            detached=tuple(tuple(x.detach() for x in rr) for rr in rows);del rows
            _,parts,_=b.full_objective(detached,[pools[u]],scales)
            actual=evaluate_mesh(detached,pools[u],scales)
            assert actual['face']['complete'],u
            for key in ['edge','face','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect']:
                assert actual[key]==old[u][key],(u,key)
            assert parts[0]==old[u]['parts'],u
            added,logits,counts=mine_faces(detached,pools[u],scales);del detached
            assert counts['eligible_outside_pool_fp']==actual['face']['actual_fp_outside_training_pool'],u
            assert counts['actual_candidates_scored']==actual['face']['scored_candidates'],u
            merged=dict(pools[u],mixed=np.concatenate([pools[u]['mixed'],added]))
            p=ROOT/'augmented_pools'/f'{u}_pool.npz';np.savez_compressed(p,**merged)
            q=ROOT/'mined'/f'{u}.npz';np.savez_compressed(q,ids=added,logits=logits,labels=np.zeros(len(added),dtype=np.bool_))
            record=dict(uid=u,vertices=len(pools[u]['vertices']),gt_faces=len(pools[u]['positive']),
                old_negative_count=len(pools[u]['mixed']),new_negative_count=len(merged['mixed']),
                old_pool_sha256=sha(ROOT/'pools'/f'{u}_pool.npz'),new_pool_sha256=sha(p),
                selected_npz_sha256=sha(q),parent_actual_counts_exact=True,old_pool_parts=parts[0],
                **counts,seconds=time.monotonic()-t)
            records.append(record);f.write(json.dumps(record,allow_nan=False)+'\n')
            print('MINED',i+1,u,'added',len(added),'eligible',counts['eligible_outside_pool_fp'],flush=True)
    assert len(records)==100 and tensor_hash(model.named_parameters())==before
    write(ROOT/'mining_complete.json',dict(state='completed',parent_checkpoint=str(SOURCE),parent_sha256=SOURCE_SHA,
        optimizer_updates=0,model_unchanged=True,all100_parent_actual_baselines_exact=True,
        selection_rule='actual predicted-edge Face, logit>0, GT negative, outside old pool; descending logit then ascending canonical ID; cap=GT Face count',
        fixed_pool_no_refresh=True,records=records,total_added=sum(x['selected'] for x in records),
        total_eligible=sum(x['eligible_outside_pool_fp'] for x in records)))
    write(ROOT/'mining_status.json',dict(state='completed',completed_meshes=100,total_added=sum(x['selected'] for x in records)))

if __name__=='__main__':main()
