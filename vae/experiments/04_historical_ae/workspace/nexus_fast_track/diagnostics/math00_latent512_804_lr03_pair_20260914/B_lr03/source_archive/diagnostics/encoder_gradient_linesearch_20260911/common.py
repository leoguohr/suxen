import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
from pathlib import Path
import importlib.util,json,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('prev',ROOT.parent/'sampling_freeze_abc_20260911/run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
UIDS=m.UIDS
SEEDS=[970000,970001]
R_VALUES=[0.,1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7]

def write(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False))
def setup(gpu):
 torch.set_num_threads(1);torch.cuda.set_device(gpu);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 path=m.SOURCE/'B/checkpoint-16100.pt';assert m.probe.digest(path)==m.START_SHA
 m.probe.UIDS=UIDS;cp,model,batch=m.probe.setup_model(path);m.install_clamp(model,-20.)
 for p,sha in cp['diagnostic_run']['source_sha256'].items():assert m.probe.digest(p)==sha
 model.requires_grad_(False)
 prefixes=('vertex_input.','face_input.','encoder_blocks.','encoder_output_norm.','mu.')
 enc={n:p for n,p in model.autoencoder.named_parameters() if n.startswith(prefixes)}
 for p in enc.values():p.requires_grad_(True)
 data=[np.load(m.teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')) for uid in UIDS]
 for i,d in enumerate(data):
  assert np.array_equal(d['vertices'],batch.vertices[i,:len(d['vertices'])].cpu().numpy())
  assert np.array_equal(d['positive'],batch.face_set[i].cpu().numpy())
 global PAIR_CHUNK
 PAIR_CHUNK=cp['args']['pair_chunk_size']
 return cp,model,batch,enc,data

def edge_logits(e,pairs,scales):return m.probe.first_order_interval(e[pairs[:,0]],e[pairs[:,1]])*scales['edge_logit_scale']
def face_logits(e,tris,scales):return m.topology.face_interval_logits(*(e[tris[:,j]] for j in range(3)),logit_scale=scales['face_logit_scale'],area_factor=scales['face_interval_factor'])
def weights(logits,y):
 p=logits.detach().float().sigmoid();y=y.float()
 return torch.stack([y*p,(1-y)*(1-p),(1-y)*p,y*(1-p)])
def soft(logits,y,w=None):
 if w is None:w=weights(logits,y)
 b=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),y.float(),reduction='none')
 return (w*b).sum(1),(w.sum(1))
def objective(rows,data,scales,base=None):
 terms=[];detail=[];saved=[]
 for i,d in enumerate(data):
  e=rows[2][i];n=len(e);pairs=torch.triu_indices(n,n,1,device='cuda').T
  edges=np.asarray(d['edges']);edges=edges.T if edges.shape[0]==2 else edges
  keys=np.sort(edges,axis=1);keys=np.unique(keys[:,0]*n+keys[:,1]);ids=(pairs[:,0]*n+pairs[:,1]).cpu().numpy()
  y=torch.as_tensor(np.isin(ids,keys),device='cuda',dtype=torch.float32)
  sums=[];masses=[];edge_saved=[]
  for k in range(0,len(pairs),65536):
   l=edge_logits(e,pairs[k:k+65536],scales);edge_saved.append(l.detach().cpu().numpy())
   w=None if base is None else weights(torch.as_tensor(base[i]['edge_train_logits'][k:k+65536],device='cuda'),y[k:k+65536])
   ns,ms=soft(l,y[k:k+65536],w);sums.append(ns);masses.append(ms)
  le=(torch.stack(sums).sum(0)/(torch.stack(masses).sum(0)+1e-8)).mean()
  if base is None:
   canonical=torch.as_tensor(keys,device='cuda',dtype=torch.long)
   le,_=m.h.soft4_loss(e,canonical,PAIR_CHUNK,scales['edge_logit_scale'])
  tr=torch.as_tensor(np.concatenate([d['positive'],d['mixed']]),device='cuda',dtype=torch.long)
  fy=torch.cat([torch.ones(len(d['positive']),device='cuda'),torch.zeros(len(d['mixed']),device='cuda')])
  fl=face_logits(rows[3][i],tr,scales)
  fw=None if base is None else weights(torch.as_tensor(base[i]['face_train_logits'],device='cuda'),fy)
  ns,ms=soft(fl,fy,fw);lf=(ns/(ms+1e-8)).mean();terms.append(le+lf)
  detail.append(dict(edge=float(le.detach()),face=float(lf.detach())))
  saved.append(dict(edge_train_logits=np.concatenate(edge_saved),face_train_logits=fl.detach().cpu().numpy()))
 return torch.stack(terms).mean(),detail,saved

def metrics(y,l,gt_total=None):
 pred=l>0;y=y.astype(bool);tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum()) if gt_total is None else int(gt_total-tp)
 return dict(tp=tp,fp=fp,fn=fn,tn=int((~pred&~y).sum()),f1=2*tp/max(2*tp+fp+fn,1))
def qstats(x):
 if not len(x):return dict(count=0)
 return dict(count=len(x),minimum=float(x.min()),p001=float(np.quantile(x,.001)),p01=float(np.quantile(x,.01)),p05=float(np.quantile(x,.05)),median=float(np.median(x)),below_zero=int((x<0).sum()),at_zero=int((x==0).sum()),abs_lt_001=int((np.abs(x)<.01).sum()),abs_lt_01=int((np.abs(x)<.1).sum()))

@torch.no_grad()
def capture(rows,data,scales,out,baseline=None):
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
  np.savez_compressed(out/f'{uid}.npz',**table);tables.append(table)
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
