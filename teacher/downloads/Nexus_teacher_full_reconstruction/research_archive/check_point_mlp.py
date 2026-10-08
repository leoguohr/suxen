import torch,itertools,json
from torch.nn import functional as F
r='/mnt/data/teacher_full_reverse/source/'
torch.set_num_threads(1)
p=torch.load(r+'results/point_diffusion/latest.pt',weights_only=True,map_location='cpu');sd=p['state_dict'];data=torch.load(r+'data/point50/training.pt',weights_only=True,map_location='cpu');ft=data['features']
lin=lambda x,pre:F.linear(x,sd[pre+'.weight'],sd[pre+'.bias'])
ln=lambda x,pre:F.layer_norm(x,(x.shape[-1],),sd[pre+'.weight'],sd[pre+'.bias'])
act={'gelu':F.gelu,'silu':F.silu,'relu':F.relu,'tanh':torch.tanh}
rs=[]
for a in act:
 h=lin(ln(ft,'text.0'),'text.1');h=lin(act[a](h),'text.3')
 for b in act:
  counts=lin(act[b](h),'count.1').argmax(-1)
  acc=sum(int(c)==len(s['vertices']) for c,s in zip(counts,data['samples']))
  rs.append({'text_act':a,'count_act':b,'count_correct':acc,'pred_counts':counts.tolist()})
print('COUNT',sorted(rs,key=lambda x:-x['count_correct'])[:6])
for a in act:
 x=lin(act[a](lin(ln(ft,'coordinate_prior.0'),'coordinate_prior.1')),'coordinate_prior.3').reshape(50,274,3)
 diff=torch.cat([(x[i,:len(s['vertices'])]-s['vertices']).flatten() for i,s in enumerate(data['samples'])]);print('prior',a,'rmse',diff.square().mean().sqrt().item(),'max',diff.abs().max().item(),'range',x.min().item(),x.max().item())
