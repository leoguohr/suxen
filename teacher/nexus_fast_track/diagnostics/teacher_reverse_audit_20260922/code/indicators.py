"""Pure metrics and edge-first reconstruction. No ground truth needed to predict."""
import itertools
import numpy as np
import torch
from torch.nn import functional as F

def triangle_candidates(n, edges):
    adj=[set() for _ in range(n)]
    for a,b in edges:
        adj[a].add(b);adj[b].add(a)
    return [(a,b,c) for a in range(n) for b in sorted(adj[a]) if b>a for c in sorted(adj[a]&adj[b]) if c>b]

def edge_score(z,pairs,method,edge_threshold=1.,cosine_scale=10.):
    a,b=z[pairs[:,0]],z[pairs[:,1]]
    if method=='cosine':return cosine_scale*(F.cosine_similarity(a,b,dim=-1)-edge_threshold)
    d=a-b
    if method=='euclidean':return d.square().sum(-1)-edge_threshold
    return d[:,:16].square().sum(-1)-d[:,16:].square().sum(-1)

def face_score(z,faces,method,face_threshold=1.,cosine_scale=10.):
    a,b,c=z[faces[:,0]],z[faces[:,1]],z[faces[:,2]]
    if method=='cosine':
        m=(F.cosine_similarity(a,b,dim=-1)+F.cosine_similarity(b,c,dim=-1)+F.cosine_similarity(c,a,dim=-1))/3
        return cosine_scale*(m-face_threshold)
    a,b=a-c,b-c
    def g(x,y):return x.square().sum(-1)*y.square().sum(-1)-(x*y).sum(-1).square()
    if method=='euclidean':return g(a,b)-face_threshold
    return g(a[:,:16],b[:,:16])-g(a[:,16:],b[:,16:])

def predict_mesh(ze,zf,method='spacetime',et=1.,ft=1.,chunk=16384):
    pairs=torch.triu_indices(len(ze),len(ze),1,device=ze.device).T
    el=edge_score(ze,pairs,method,et)
    pe=pairs[el>0]
    cand=torch.tensor(triangle_candidates(len(ze),pe.tolist()),dtype=torch.long,device=ze.device).reshape(-1,3)
    fl=torch.cat([face_score(zf,cand[i:i+chunk],method,ft) for i in range(0,len(cand),chunk)]) if len(cand) else ze.new_empty((0,))
    return dict(edge_pairs=pairs,edge_logits=el,pred_edges=pe,face_candidates=cand,face_logits=fl,pred_faces=cand[fl>0])

def mesh_metrics(pred,gt_faces):
    fset=lambda a:set(tuple(sorted(f)) for f in (a.tolist() if isinstance(a,torch.Tensor) else a))
    pf=fset(pred['pred_faces']);gf=fset(gt_faces)
    pe=fset(pred['pred_edges']);ge=set(e for f in gf for e in itertools.combinations(f,2))
    c=fset(pred['face_candidates'])
    def m(p,g):
        tp=len(p&g);fp=len(p-g);fn=len(g-p)
        return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/max(2*tp+fp+fn,1))
    return dict(face=m(pf,gf),edge=m(pe,ge),candidate_count=len(c),gt_faces_missing_candidates=len(gf-c))

def aggregate(records):
    ans={}
    for task in ['edge','face']:
        d={k:sum(r[task][k] for r in records) for k in ['tp','fp','fn']}
        d['f1']=2*d['tp']/max(2*d['tp']+d['fp']+d['fn'],1);ans[task]=d
    ans['joint_strict']=sum(all(r[t]['fp']==r[t]['fn']==0 for t in ['face','edge']) for r in records)
    ans['face_exact_vs_reference']=sum(r.get('reference_face_symmetric_diff',-1)==0 for r in records)
    return ans
