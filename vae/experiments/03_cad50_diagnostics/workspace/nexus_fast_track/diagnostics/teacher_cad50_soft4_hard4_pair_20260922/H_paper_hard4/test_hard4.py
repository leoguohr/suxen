"""Bounded CPU test of the actual production Edge/Face loss functions."""
import json
from pathlib import Path
import torch as T
from torch.nn import functional as F
import effective_loss_and_scoring as s
T.set_num_threads(1)
T.manual_seed(924)
s.set_mode('paper_hard4')
cases=[('four_nonempty',[2.,-2.,1.,-1.],[1,0,0,1]),
       ('empty_fp_fn',[2.,-2.,4.,-4.],[1,0,1,0]),
       ('only_fn',[-2.,-1.,0.],[1,1,1]),
       ('zero_is_negative',[0.,0.,1.,-1.],[1,0,1,0])]
results=[]
for name,values,targets in cases:
    x=T.tensor(values,requires_grad=True);y=T.tensor(targets).float()
    n,c=s.hard4_sums(x,y);loss=s.group_means(n,c).sum()/4
    pred=x.detach()>0;gt=y==1
    masks=[gt&pred,(~gt)&(~pred),(~gt)&pred,gt&(~pred)]
    bce=F.binary_cross_entropy_with_logits(x,y,reduction='none')
    reference=sum(bce[g].mean() if g.any() else bce.sum()*0 for g in masks)/4
    T.testing.assert_close(loss,reference,rtol=0,atol=0)
    gradient=T.autograd.grad(loss,x,retain_graph=True)[0]
    rg=T.autograd.grad(reference,x)[0]
    T.testing.assert_close(gradient,rg,rtol=0,atol=0)
    expected=T.zeros_like(x)
    for mask in masks:
        if mask.any():expected[mask]=(x.detach().sigmoid()[mask]-y[mask])/(4*int(mask.sum()))
    T.testing.assert_close(gradient,expected,rtol=1e-6,atol=1e-8)
    assert (gradient[y==1]<0).all() and (gradient[y==0]>0).all()
    assert not c.requires_grad and n.requires_grad and loss.requires_grad
    if name=='zero_is_negative':assert c.tolist()==[1,2,0,1]
    results.append(dict(case=name,counts=c.tolist(),loss=float(loss.detach()),gradient_direction_correct=True))
x=T.linspace(-5.3,6.2,257).requires_grad_();y=(T.arange(257)%3==0).float()
whole=s.face_reconstruction_loss(x,y,len(x));gw=T.autograd.grad(whole,x)[0]
chunks=[]
for chunk in [1,7,31,100]:
    loss=s.face_reconstruction_loss(x,y,chunk);g=T.autograd.grad(loss,x)[0]
    T.testing.assert_close(loss,whole,rtol=2e-6,atol=1e-7)
    T.testing.assert_close(g,gw,rtol=2e-6,atol=1e-8)
    chunks.append(dict(task='face',chunk=chunk,loss_error=float(abs(loss-whole).detach()),gradient_error=float((g-gw).abs().max())))
# Actual edge scorer, all pairs, different chunk boundaries, and explicit reference.
z=(T.randn(11,32)*.15).requires_grad_();keys=T.tensor([1,2,3,14,27,40,53,66,79,92,105])
pairs=T.triu_indices(11,11,1).T;labels=T.isin(pairs[:,0]*11+pairs[:,1],keys).float()
logits=s.first_order_interval(z[pairs[:,0]],z[pairs[:,1]])*.93
ref=s.face_reconstruction_loss(logits,labels,len(logits));gr=T.autograd.grad(ref,z)[0]
for chunk in [1,7,55]:
    loss,_=s.edge_reconstruction_loss(z,keys,chunk,.93);g=T.autograd.grad(loss,z)[0]
    T.testing.assert_close(loss,ref,rtol=2e-6,atol=1e-7)
    T.testing.assert_close(g,gr,rtol=2e-5,atol=1e-7)
    chunks.append(dict(task='edge',chunk=chunk,loss_error=float(abs(loss-ref).detach()),gradient_error=float((g-gr).abs().max())))
# Finite difference away from zero, while the hard groups stay unchanged.
x=T.tensor([-3.,-1.2,.8,2.4],requires_grad=True);y=T.tensor([1.,0.,1.,0.]);d=T.tensor([.2,-.3,.4,-.1])
loss=s.face_reconstruction_loss(x,y,2);g=T.autograd.grad(loss,x)[0];h=.002
assert T.equal((x.detach()+h*d)>0,(x.detach()-h*d)>0)
fd=(s.face_reconstruction_loss(x.detach()+h*d,y,2)-s.face_reconstruction_loss(x.detach()-h*d,y,2))/(2*h)
T.testing.assert_close(fd,(g*d).sum(),rtol=.003,atol=2e-5)
# Preserve the original fully differentiable Soft4 definition exactly.
s.set_mode('soft4');x=T.linspace(-4,4,37).requires_grad_();y=(T.arange(37)%3==0).float()
p=x.sigmoid();w=T.stack([y*p,(1-y)*(1-p),(1-y)*p,y*(1-p)])
reference=((w*F.binary_cross_entropy_with_logits(x,y,reduction='none')).sum(1)/(w.sum(1)+1e-8)).mean()
actual=s.face_reconstruction_loss(x,y,7)
T.testing.assert_close(actual,reference,rtol=0,atol=0)
T.testing.assert_close(T.autograd.grad(actual,x,retain_graph=True)[0],T.autograd.grad(reference,x)[0],rtol=0,atol=0)
assert not T.cuda.is_initialized()
result=dict(passed=True,cases=results,chunk_tests=chunks,fixed_group_finite_difference_passed=True,
    original_soft4_value_and_gradient_bitwise_equal=True,optimizer_updates=0,gpu_used=False,
    empty_groups='Explicit experiment convention: zero contribution, outer denominator always4; not an official-code claim')
Path(__file__).with_name('hard4_test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
