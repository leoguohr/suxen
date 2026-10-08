import sys,itertools,json,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent/'reconstruction'))
from teacher_ae import load_teacher_ae
from flow_candidates import TopologyFlow
r=Path(__file__).parent/'source';out=Path(__file__).parent/'evidence'
torch.set_num_threads(1);torch.manual_seed(1200)
d=torch.load(r/'data/point50/training.pt',map_location='cpu',weights_only=True)
ae=load_teacher_ae(r/'results/minkowski_target_099/vae.pt')
sd=torch.load(r/'results/minkowski_target_099/best_flow.pt',map_location='cpu',weights_only=True)
f=TopologyFlow(sd['layers']);f.load_state_dict(sd['state_dict'],strict=True);f.eval()
norm=torch.load(r/'results/minkowski_target_099/latent_normalization.pt',map_location='cpu',weights_only=True)
s=d['samples'][0];xyz=s['vertices'][None].repeat(3,1,1)
with torch.inference_mode():
 mu,_=ae.encode(s['vertices'],s['faces']);z1=(mu-norm['mean'])/norm['std'];z1=z1[None].expand(3,-1,-1)
 e=torch.randn_like(z1);t=torch.tensor([.1,.5,.9]);zt=(1-t[:,None,None])*e+t[:,None,None]*z1;target=z1-e
 results=[];start=time.time()
 for head,fk,rm,rs,ep,ff in itertools.product([4,6],['inverse','pow2','linear'],['pair','half'],[1.,torch.pi],[1e-5,1e-6],['gelu','silu']):
  ts,to,td=1000.,'cs',32
  f.op.update(heads=head,time_scale=ts,time_order=to,time_den=td,rope=rm,rope_scale=rs,freq_kind=fk,eps=ep,ff=ff)
  v=f(zt,t,xyz);loss=(v-target).square().mean().item();results.append(dict(f.op,mse=loss))
  if len(results)%20==0:print(len(results),time.time()-start,sorted(results,key=lambda r:r['mse'])[0],flush=True)
 results.sort(key=lambda a:a['mse']);(out/'flow_forward_refined.json').write_text(json.dumps(results,indent=2));print('BEST',results[:10])
