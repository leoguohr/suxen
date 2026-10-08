@torch.no_grad()
def capture(rows,data,scales,out,baseline=None,save_npz=True):
 out.mkdir(exist_ok=True);results=[];tables=[]
 for i,(uid,d) in enumerate(zip(UIDS,data)):
  n=len(rows[2][i]);pairs=np.stack(np.triu_indices(n,1),axis=1);ids=pairs[:,0]*n+pairs[:,1]
  edges=d['edges'].T if d['edges'].shape[0]==2 else d['edges'];ge=np.sort(edges,axis=1);gt=np.unique(ge[:,0]*n+ge[:,1]);ey=np.isin(ids,gt)
  els=[]
  for k in range(0,len(pairs),65536):els.append(edge_logits(rows[2][i],torch.as_tensor(pairs[k:k+65536],device='cuda'),scales).cpu().numpy())
  el=np.concatenate(els);em=(2*ey.astype(np.int8)-1)*el
  adj=[set() for _ in range(n)]
  for a,b in pairs[el>0]:adj[a].add(int(b))
  actual=[]
  for a in range(n):
   for b in sorted(adj[a]):
    actual.extend((a,b,c) for c in sorted(adj[a].intersection(adj[b])))
  actual=np.asarray(actual,dtype=np.int64).reshape(-1,3)
  ak=m.probe.keys(actual,n);gtf=m.probe.keys(d['positive'],n)
  # Union includes GT faces missing from edge gating, and baseline candidates for matching crossings.
  fk=np.unique(np.r_[ak,gtf,[] if baseline is None else baseline[i]['face_ids']]).astype(np.int64)
  ft=m.probe.triples(fk,n);fy=np.isin(fk,gtf);fa=np.isin(fk,ak)
  fl=face_logits(rows[3][i],torch.as_tensor(ft,device='cuda'),scales).cpu().numpy();fm=(2*fy.astype(np.int8)-1)*fl
  table=dict(edge_ids=ids,edge_pairs=pairs,edge_labels=ey,edge_logits=el,edge_margin=em,
   face_ids=fk,face_triples=ft,face_labels=fy,face_actual=fa,face_logits=fl,face_margin=fm,
   mu=rows[0][i].detach().cpu().numpy(),logvar=rows[1][i].detach().cpu().numpy())
  if save_npz:np.savez_compressed(out/f'{uid}.npz',**table)
  tables.append(table)
  stats={name:qstats(vals) for name,vals in [('edge_gt',em[ey]),('edge_non_gt',em[~ey]),('face_gt',fm[fy]),('face_non_gt_actual',fm[~fy&fa])]}
  near={}
  for kind,y,mar,ks,trip in [('edge',ey,em,ids,pairs),('face',fy,fm,fk,ft)]:
   for label in [0,1]:
    ix=np.flatnonzero(y==label);ix=ix[np.argsort(mar[ix])[:32]]
    near[f'{kind}_{label}']=[dict(id=int(ks[j]),vertices=trip[j].tolist(),margin=float(mar[j])) for j in ix]
  cross={}
  if baseline is not None:
   for kind in ['edge','face']:
    b=baseline[i];idx=np.searchsorted(table[kind+'_ids'],b[kind+'_ids']);now=table[kind+'_margin'][idx];old=b[kind+'_margin'];ix=np.flatnonzero((old>0)&(now<=0))
    cross[kind]=[dict(id=int(b[kind+'_ids'][j]),vertices=b['edge_pairs' if kind=='edge' else 'face_triples'][j].tolist(),label=int(b[kind+'_labels'][j]),margin0=float(old[j]),margin=float(now[j]),hard_error_now=bool((table[kind+'_logits'][idx[j]]>0)!=table[kind+'_labels'][idx[j]])) for j in ix]
  results.append(dict(uid=uid,edge=metrics(ey,el),face=metrics(fy[fa],fl[fa],len(gtf)),gt_faces_missing_from_edge_candidates=int((fy&~fa).sum()),margins=stats,near=near,crossings=cross))
 return results,tables
