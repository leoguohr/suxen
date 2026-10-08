"""Actual predicted-edge Face enumeration, with explicit incomplete accounting."""
import time
from runtime import T,np,c

FACE_SECONDS=30.0
FACE_CHUNK=32768
@T.no_grad()
def evaluate_mesh(rows,pool,scales):
    n=len(pool['vertices']);gt=np.sort(pool['positive'],axis=1);gtkeys=c.m.probe.keys(gt,n)
    edgekeys=np.unique(np.sort(pool['edges'].T,axis=1)@np.array([n,1]))
    adj=np.zeros((n,n),dtype=bool);tp=fp=fn=tn=0;minpos=minneg=float('inf')
    # All pairs, bounded scoring blocks. No pair sampling.
    for lo in range(0,n,64):
        ids=np.array([(i,j) for i in range(lo,min(n,lo+64)) for j in range(i+1,n)],dtype=np.int64).reshape(-1,2)
        logits=c.edge_logits(rows[2][0],T.as_tensor(ids,device='cuda'),scales).cpu().numpy()
        assert np.isfinite(logits).all()
        labels=np.isin(ids@np.array([n,1]),edgekeys);pred=logits>0
        tp+=int((pred&labels).sum());fp+=int((pred&~labels).sum());fn+=int((~pred&labels).sum());tn+=int((~pred&~labels).sum())
        if labels.any():minpos=min(minpos,float(logits[labels].min()))
        if (~labels).any():minneg=min(minneg,float((-logits[~labels]).min()))
        adj[ids[pred,0],ids[pred,1]]=True
    edge=dict(tp=tp,fp=fp,fn=fn,tn=tn,f1=2*tp/max(2*tp+fp+fn,1))
    covered=adj[gt[:,0],gt[:,1]]&adj[gt[:,0],gt[:,2]]&adj[gt[:,1],gt[:,2]]
    gl=c.face_logits(rows[3][0],T.as_tensor(gt,device='cuda'),scales).cpu().numpy();assert np.isfinite(gl).all()
    face_tp=int(((gl>0)&covered).sum());face_fn=len(gt)-face_tp
    trainkeys=c.m.probe.keys(np.concatenate([gt,pool['mixed']]),n)
    count=ftp=ffp=ftn=outside_fp=0;complete=True;pending=[];fminneg=None;t=time.monotonic()
    def score(tri):
        nonlocal count,ftp,ffp,ftn,outside_fp,fminneg
        tri=np.asarray(tri,dtype=np.int64).reshape(-1,3);keys=c.m.probe.keys(tri,n)
        logits=c.face_logits(rows[3][0],T.as_tensor(tri,device='cuda'),scales).cpu().numpy();assert np.isfinite(logits).all()
        y=np.isin(keys,gtkeys);pred=logits>0
        count+=len(tri);ftp+=int((pred&y).sum());ffp+=int((pred&~y).sum());ftn+=int((~pred&~y).sum())
        outside_fp+=int((pred&~y&~np.isin(keys,trainkeys)).sum())
        if (~y).any():fminneg=min(fminneg if fminneg is not None else float('inf'),float((-logits[~y]).min()))
    for i in range(n):
        for j in np.flatnonzero(adj[i]):
            kk=np.flatnonzero(adj[i]&adj[j])
            if len(kk):pending.extend((i,int(j),int(k)) for k in kk)
            while len(pending)>=FACE_CHUNK:
                score(pending[:FACE_CHUNK]);pending=pending[FACE_CHUNK:]
                if time.monotonic()-t>FACE_SECONDS:complete=False;break
            if not complete or time.monotonic()-t>FACE_SECONDS:complete=False;break
        if not complete:break
    if complete and pending:score(pending)
    if complete:assert ftp==face_tp
    face=dict(tp=face_tp,fn=face_fn,fp=ffp if complete else None,tn=ftn if complete else None,
              f1=2*face_tp/max(2*face_tp+ffp+face_fn,1) if complete else None,
              complete=complete,scored_candidates=count,fp_lower_bound=ffp,
              actual_fp_outside_training_pool=outside_fp if complete else None,
              actual_fp_outside_training_pool_lower_bound=outside_fp,
              incomplete_reason=None if complete else '30_second_per_mesh_face_enumeration_budget')
    ep=fp==fn==0;fperfect=(face_fn==0 and ffp==0) if complete else False if face_fn>0 or ffp>0 else None
    return dict(edge=edge,face=face,gt_face_candidates=int(covered.sum()),missing_gt_face_candidates=int((~covered).sum()),
        edge_perfect=ep,face_perfect=fperfect,joint_perfect=bool(ep and fperfect is True),
        margins=dict(edge_gt=minpos if np.isfinite(minpos) else None,edge_non_gt=minneg if np.isfinite(minneg) else None,
                     face_gt_all=float(gl.min()),face_non_gt_scored=fminneg),face_seconds=time.monotonic()-t)
