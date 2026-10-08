import torch,json,pathlib
r=pathlib.Path('/mnt/data/teacher_full_reverse/source')
paths=['data/point50/training.pt','results/point_diffusion/text_conditions.pt','results/point_diffusion/latest.pt','results/point_prior_candidate/candidate.pt','results/minkowski_target_099/best_flow.pt','results/overfit/cosine/vae.pt','results/overfit/cosine/flow.pt']
def brief(d, level=0):
 if isinstance(d,torch.Tensor):return {'shape':list(d.shape),'dtype':str(d.dtype),'min':float(d.min()) if d.numel() else None,'max':float(d.max()) if d.numel() else None}
 if isinstance(d,dict):return {str(k):brief(v,level+1) if level<2 else ('dict '+str(list(v)[:10]) if isinstance(v,dict) else brief(v,level+1)) for k,v in d.items() if k!='optimizer'}
 if isinstance(d,(list,tuple)):
  return {'type':str(type(d)),'len':len(d),'first':brief(d[0],level+1) if d else None}
 return str(d)[:500]
for p in paths:
 d=torch.load(r/p,map_location='cpu',weights_only=True)
 print('\n\n---',p,'---')
 print(json.dumps(brief(d),ensure_ascii=False,indent=1))
