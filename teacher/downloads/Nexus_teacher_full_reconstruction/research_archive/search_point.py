import sys,itertools,json,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'reconstruction'))
from flow_candidates import PointFlow
r=Path(__file__).parent/'source';out=Path(__file__).parent/'evidence'
torch.set_num_threads(1);torch.manual_seed(1200)
d=torch.load(r/'data/point50/training.pt',map_location='cpu',weights_only=True)
sd=torch.load(r/'results/point_diffusion/latest.pt',map_location='cpu',weights_only=True)
f=PointFlow();f.load_state_dict(sd['state_dict'],strict=True);f.eval()
s=d['samples'][0];xyz=s['vertices'][None].repeat(4,1,1);text=d['features'][0:1].expand(4,-1)
with torch.inference_mode():
 e=torch.randn_like(xyz);t=torch.tensor([.1,.5,.9,.99]);xt=(1-t[:,None,None])*e+t[:,None,None]*xyz
 results=[];start=time.time()
 for head,rm,rs,act,add,ep in itertools.product([4,6],['pair','half','none'],[1.,torch.pi],['silu','gelu'],[False,True],[1e-5]):
  if rm=='none' and rs!=1.:continue
  f.op.update(heads=head,rope=rm,rope_scale=rs,text_act=act,add_text=add,eps=ep)
  v=f.denoise(xt,t,text);loss=(v-xyz).square().mean().item();m=(v[-1]-xyz[-1]).square().mean().sqrt().item()
  results.append(dict(f.op,mse=loss,terminal_rmse=m))
 results.sort(key=lambda a:a['mse']);(out/'point_forward_candidates.json').write_text(json.dumps(results,indent=2));print('elapsed',time.time()-start,'BEST',results[:12])
