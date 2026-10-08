import sys,torch,json,itertools
from pathlib import Path
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).parent/'reconstruction'))
from teacher_ae import ReconstructedTeacherAE
sys.path.insert(0,'/mnt/data/Nexus_teacher_goal_pack/teacher_reconstruction')
from replay_teacher_ae import triangles
r=Path(__file__).parent/'source';data=torch.load(r/'data/point50/training.pt',weights_only=True,map_location='cpu')['samples'];torch.set_num_threads(1)
results={}
with torch.inference_mode():
 for m in ['cosine','euclidean']:
  d=torch.load(r/f'results/overfit/{m}/vae.pt',weights_only=True,map_location='cpu');model=ReconstructedTeacherAE(2);model.load_state_dict(d['state_dict']);model.eval();et=float(model.indicator.edge_threshold);ft=float(model.indicator.face_threshold)
  for j in [1,3,13,20]:
   s=data[j];ze,zf=model(s['vertices'],s['faces']);pi=torch.triu_indices(len(ze),len(ze),1).T
   gt=set(tuple(sorted(p)) for f in s['faces'].tolist() for p in itertools.combinations(f,2))
   ref=json.load(open(r/f'results/overfit/{m}/reconstructions/ae_{j:02d}_faces.json'))
   inv={int(v):i for i,v in enumerate(s['original_vertex_indices'])};ref=set(tuple(sorted(inv[i] for i in p)) for p in ref)
   vals={}
   for edge_kind in (['cos_minus','th_minus_cos','dot_minus'] if m=='cosine' else ['sq_minus','th_minus_sq','dist_minus','th_minus_dist']):
    if 'cos' in edge_kind:raw=F.cosine_similarity(ze[pi[:,0]],ze[pi[:,1]],dim=-1)
    elif 'dot' in edge_kind:raw=(ze[pi[:,0]]*ze[pi[:,1]]).sum(-1)
    else:
     raw=(ze[pi[:,0]]-ze[pi[:,1]]).square().sum(-1)
     if 'dist' in edge_kind:raw=raw.sqrt()
    pred_e=pi[(et-raw if edge_kind.startswith('th_') else raw-et)>0]
    ep=set(map(tuple,pred_e.tolist()));ef1=2*len(ep&gt)/(len(ep)+len(gt))
    cand=torch.tensor(triangles(len(ze),pred_e.tolist()),dtype=torch.long).reshape(-1,3)
    if len(cand)>100000:continue
    cs=F.normalize(zf,dim=-1)[cand] if m=='cosine' else zf[cand]
    a=cs[:,0]-cs[:,2];b=cs[:,1]-cs[:,2];area=a.square().sum(-1)*b.square().sum(-1)-(a*b).sum(-1).square()
    paircos=torch.stack([(cs[:,a]*cs[:,b]).sum(-1) for a,b in [(0,1),(1,2),(0,2)]],-1)
    fs={'area':area,'cosmean':paircos.mean(-1),'cosmin':paircos.min(-1).values,'cosprod':paircos.prod(-1),'cosdet':1+2*paircos.prod(-1)-paircos.square().sum(-1)}
    for fk in (['cosmean','cosmin','cosprod','cosdet','area'] if m=='cosine' else ['area']):
     for sign in [1,-1]:
      pset=set(map(tuple,cand[sign*(fs[fk]-ft)>0].tolist()));diff=len(pset^ref)
      vals[(edge_kind,fk,sign)]={'edge_f1':ef1,'faces_diff':diff,'nfaces':len(pset),'ref_n':len(ref)}
   best=sorted(vals.items(),key=lambda x:x[1]['faces_diff'])[:8]
   print(m,j,'thr',et,ft,'best',best,flush=True);results[f'{m}_{j}']=[(str(k),v) for k,v in best]
(Path(__file__).parent/'evidence/indicator_hypotheses.json').write_text(json.dumps(results,indent=2))
