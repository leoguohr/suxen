"""Unchanged full enumeration and deterministic top-K mining from round1."""
from runtime import T,np,c
from evaluate import FACE_CHUNK

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
