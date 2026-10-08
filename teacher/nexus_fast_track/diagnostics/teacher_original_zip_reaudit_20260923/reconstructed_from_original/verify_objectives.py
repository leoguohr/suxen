"""Fixed CPU component/gradient comparison; no model training or optimizer."""
import argparse
import json
from pathlib import Path
import sys
import torch
from recovered_objectives import (grouped_bce, standard_normal_kl_elements,
    topology_velocity_mse, point_velocity_count_loss, cached_prior_mse)


class FixedField:
    def __call__(self, x, time, condition, mask=None):
        return x * .2 + .3

    def count_logits(self, features):
        return features

    def prior(self, features):
        return torch.zeros(features.shape[0], 4, 3)

    residual_scale = torch.tensor(.125)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.baseline_root))
    import objectives as yesterday
    torch.manual_seed(314)
    scores=torch.tensor([2.,-3.,-1.,4.],requires_grad=True)
    labels=torch.tensor([1,0,1,0])
    mean=torch.randn(2,4,6,requires_grad=True)
    logvar=torch.randn(2,4,6,requires_grad=True)
    valid=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    target=torch.randn(2,4,3)
    noise=torch.randn_like(target)
    time=torch.tensor([0.,.99])
    features=torch.randn(2,5)
    counts=torch.tensor([2,4])
    model=FixedField()
    new_point=point_velocity_count_loss(model,target,features,noise,time,valid,counts,count_weight=.5)
    old_point=yesterday.point_velocity_and_count_loss(model,target,features,time,noise,valid,counts,count_coefficient=.5)
    pairs={
        'grouped_bce':(grouped_bce(scores,labels),yesterday.hard4_bce(scores,labels)),
        'kl_elements_mean':(standard_normal_kl_elements(mean,logvar).mean(),yesterday.kl_standard_normal(mean,logvar)),
        'topology_velocity_mse':(topology_velocity_mse(model,target,target,noise,time,valid),yesterday.topology_velocity_loss(model,target,target,time,noise,valid)),
        'point_velocity_mse':(new_point['velocity_mse'],old_point['velocity_mse']),
        'point_count_ce':(new_point['count_ce'],old_point['count_ce']),
        'point_total':(new_point['total'],old_point['loss']),
        'cached_prior_mse':(cached_prior_mse(model,features,noise,target,valid),yesterday.cached_prior_loss(model,features,noise,target,valid))}
    result={'device':'cpu','optimizer_updates':0,'author_model':'gpt-6-astra','reasoning_effort':'xhigh',
            'qualification':'Formula equivalence at fixed synthetic inputs; does not prove the missing original trainer.'}
    for name,(a,b) in pairs.items():
        assert torch.allclose(a,b,rtol=1e-6,atol=1e-7),name
        result[name]={'new':float(a.detach()),'yesterday':float(b.detach()),'abs_difference':float((a-b).abs().detach())}
    (pairs['grouped_bce'][0]+pairs['kl_elements_mean'][0]).backward()
    result['component_gradients_finite']=all(bool(torch.isfinite(x.grad).all()) for x in (scores,mean,logvar))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
