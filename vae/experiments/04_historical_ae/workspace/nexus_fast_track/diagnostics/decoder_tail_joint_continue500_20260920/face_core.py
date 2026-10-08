"""Exact existing Face scoring and fully-differentiable reduction, on frozen features."""
from runtime import T,np,b,c
FACE_CHUNK=32768

def center(head,h):
    raw=head(h);return raw,raw-raw.mean(0,keepdim=True)

def objective(e,tris,labels,scales):
    # Same order as c.objective: one complete [positive; mixed] Face pool, no chunk averaging.
    logits=c.face_logits(e,tris,scales)
    w=b.full_weights(logits,labels)
    if logits.requires_grad:assert w.requires_grad
    numerator,mass=c.soft(logits,labels,w)
    return (numerator/(mass+1e-8)).mean(),logits

def train_cycle(head,data,scales,backward):
    losses=[]
    for d in data:
        with T.set_grad_enabled(backward):
            raw,e=center(head,d['hidden']);loss,logits=objective(e,d['train_tris'],d['train_labels'],scales)
            losses.append(float(loss.detach()))
            if backward:(loss/100).backward()
        del raw,e,loss,logits
    return losses

def edge_check(d,scales):
    with T.no_grad():s=c.edge_logits(d['edge_embedding'],d['pairs'],scales);p=s>0;y=d['edge_labels']
    counts=dict(tp=int((p&y).sum()),fp=int((p&~y).sum()),fn=int((~p&y).sum()),tn=int((~p&~y).sum()))
    assert all(counts[k]==d['reference']['edge'][k] for k in counts)
    assert T.equal(p,d['edge_prediction'])
    return {**counts,'f1':2*counts['tp']/max(2*counts['tp']+counts['fp']+counts['fn'],1)}

@T.no_grad()
def evaluate_one(head,d,scales,embedding=None,save_logits=False):
    if embedding is None:raw,e=center(head,d['hidden'])
    else:e=embedding
    loss,train_logits=objective(e,d['train_tris'],d['train_labels'],scales)
    total=dict(tp=0,fp=0,tn=0);outside=0;parts=[];minneg=None
    for start in range(0,len(d['actual_tris']),FACE_CHUNK):
        sl=slice(start,start+FACE_CHUNK);l=c.face_logits(e,d['actual_tris'][sl],scales);y=d['actual_labels'][sl];p=l>0
        total['tp']+=int((p&y).sum());total['fp']+=int((p&~y).sum());total['tn']+=int((~p&~y).sum())
        outside+=int((p&~y&~d['actual_in_training_pool'][sl]).sum())
        if bool((~y).any()):minneg=min(minneg if minneg is not None else float('inf'),float((-l[~y]).min()))
        if save_logits:parts.append(l.cpu().numpy())
    gt_logits=c.face_logits(e,d['gt_faces'],scales)
    assert total['tp']==int(((gt_logits>0)&d['gt_covered']).sum())
    total['fn']=len(d['gt_faces'])-total['tp'];total['f1']=2*total['tp']/max(2*total['tp']+total['fp']+total['fn'],1)
    total.update(complete=True,scored_candidates=len(d['actual_tris']),actual_fp_outside_augmented_pool=outside,actual_fp_inside_augmented_pool=total['fp']-outside)
    edge=edge_check(d,scales);ep=edge['fp']==edge['fn']==0;fp=total['fp']==total['fn']==0
    row=dict(uid=d['uid'],vertices=len(d['hidden']),gt_edges=int(d['edge_labels'].sum()),gt_faces=len(d['gt_faces']),edge=edge,face=total,edge_perfect=ep,face_perfect=fp,joint_perfect=ep and fp,
        face_soft4=float(loss),training_pool=c.metrics(d['train_labels'].cpu().numpy().astype(bool),train_logits.cpu().numpy()),
        gt_face_candidates=int(d['gt_covered'].sum()),missing_gt_face_candidates=int((~d['gt_covered']).sum()),
        margins=dict(face_gt_all=float(gt_logits.min()),face_non_gt_actual=minneg))
    saved=None
    if save_logits:saved=dict(face_embedding=e.cpu().numpy(),train_logits=train_logits.cpu().numpy(),actual_logits=np.concatenate(parts) if parts else np.empty(0,dtype=np.float32),gt_logits=gt_logits.cpu().numpy())
    return row,saved

def summary(rows,step):
    return dict(step=step,meshes=rows,edge_perfect=sum(r['edge_perfect'] for r in rows),face_perfect=sum(r['face_perfect'] for r in rows),joint_perfect=sum(r['joint_perfect'] for r in rows),mean_face_soft4=sum(r['face_soft4'] for r in rows)/100,
        **{kind+'_'+key:sum(r[kind][key] for r in rows) for kind in ['edge','face'] for key in ['tp','fp','fn']})
