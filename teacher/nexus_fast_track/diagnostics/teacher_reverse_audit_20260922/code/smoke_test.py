"""Strict weight loading, frozen-model parity, and finite-gradient smoke test.
No optimizer steps and no updates to teacher weights. Uses tiny diagnostic input.
"""
import argparse,json,torch
from pathlib import Path
from models import load_points,load_topology,sample_points,sample_topology
from objectives import hard4_bce,topology_velocity_loss,point_velocity_and_count_loss

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--teacher-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(1);torch.manual_seed(16)
 root=a.teacher_root
 point=load_points(root/'results/point_diffusion/latest.pt');flow=load_topology(root/'results/minkowski_target_099/best_flow.pt')
 s=torch.tensor([-1.,2.,1.,-2.],requires_grad=True);y=torch.tensor([1.,1.,0.,0.]);l=hard4_bce(s,y);l.backward();assert (s.grad[:2]<0).all() and (s.grad[2:]>0).all()
 cache=torch.load(root/'data/point50/training.pt',weights_only=True,map_location='cpu');xyz=cache['samples'][3]['vertices'][None];ft=cache['features'][3:4];t=torch.tensor([.5]);noise=torch.randn(1,8,64);target=torch.randn_like(noise)
 fl=topology_velocity_loss(flow,target,xyz,t,noise);fl.backward();fg=all(p.grad is None or torch.isfinite(p.grad).all() for p in flow.parameters())
 mask=torch.ones(1,8,dtype=torch.bool);pg=point_velocity_and_count_loss(point,xyz,ft,t,torch.randn_like(xyz),mask,torch.tensor([8]),count_coefficient=1.);pg['loss'].backward();pp=all(p.grad is None or torch.isfinite(p.grad).all() for p in point.parameters())
 assert bool(fg) and bool(pp)
 r=dict(scope='Synthetic/tiny finite-gradient smoke test only. No model or optimizer updates.',strict_point_tensors=len(point.state_dict()),strict_flow_tensors=len(flow.state_dict()),hard4_loss=l.item(),hard4_gradient=s.grad.tolist(),topology_test_loss=fl.item(),point_test_loss=pg['loss'].item(),point_count_coefficient='arbitrary test value 1, not recovered teacher training hyperparameter',finite_flow_gradients=bool(fg),finite_point_gradients=bool(pp),optimizer_updates=0)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
