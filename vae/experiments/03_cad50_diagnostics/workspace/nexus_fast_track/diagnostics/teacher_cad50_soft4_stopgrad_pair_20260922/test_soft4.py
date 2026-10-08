"""One bounded check of the exact production loss and its intentional update rule."""
import json
from pathlib import Path
import torch as T
from torch.nn import functional as F
import effective_loss_and_scoring as s

T.set_num_threads(1)
T.manual_seed(197)
x=T.linspace(-4,4,37).requires_grad_()
y=(T.arange(37)%3==0).float()
def loss(z,mode,chunk=37):
    s.set_mode(mode)
    terms=[s.soft4_sums(a,b) for a,b in zip(z.split(chunk),y.split(chunk))]
    n=sum(v[0] for v in terms);m=sum(v[1] for v in terms)
    return (n/(m+1e-8)).mean()
full=loss(x,False);gf=T.autograd.grad(full,x)[0]
stop=loss(x,True);gs=T.autograd.grad(stop,x)[0]
assert T.equal(full,stop)
p=x.detach().sigmoid();w=T.stack([y*p,(1-y)*(1-p),(1-y)*p,y*(1-p)])
bce=F.binary_cross_entropy_with_logits(x,y,reduction='none')
m=w.sum(1);n=(w*bce.detach()).sum(1)
fixed=((w*bce).sum(1)/(m+1e-8)).mean()
gfixed=T.autograd.grad(fixed,x)[0]
T.testing.assert_close(gs,gfixed,rtol=0,atol=0)
assert gs.abs().max()>0 and not w.requires_grad
dp=p*(1-p);dw=T.stack([y*dp,-(1-y)*dp,(1-y)*dp,-y*dp])
chain=(dw*(bce.detach()[None,:]*(m[:,None]+1e-8)-n[:,None])/(m[:,None]+1e-8).square()).mean(0)
T.testing.assert_close(gf-gs,chain,rtol=2e-5,atol=2e-8)
checks=[]
for mode in [False,True]:
    whole=loss(x,mode);gw=T.autograd.grad(whole,x)[0]
    chunked=loss(x,mode,7);gc=T.autograd.grad(chunked,x)[0]
    T.testing.assert_close(whole,chunked,rtol=1e-6,atol=1e-7)
    T.testing.assert_close(gw,gc,rtol=1e-5,atol=1e-7)
    z=T.randn(11,32,requires_grad=True)*.1;keys=T.tensor([1,2,3,14,27,40,53,66,79,92,105])
    a,_=s.soft4_loss(z,keys,55,1.);ga=T.autograd.grad(a,z,retain_graph=True)[0]
    b,_=s.soft4_loss(z,keys,7,1.);gb=T.autograd.grad(b,z)[0]
    T.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    T.testing.assert_close(ga,gb,rtol=2e-5,atol=1e-7)
    checks.append(dict(stop_weight_grad=mode,chunk_loss_error=float(abs(a-b)),chunk_gradient_max_error=float((ga-gb).abs().max())))
direction=T.linspace(-1,1,37);h=.01
def frozen(z):return ((w*F.binary_cross_entropy_with_logits(z,y,reduction='none')).sum(1)/(m+1e-8)).mean()
fd=(frozen(x.detach()+h*direction)-frozen(x.detach()-h*direction))/(2*h)
pred=(gs*direction).sum()
T.testing.assert_close(fd,pred,rtol=.002,atol=1e-5)
result=dict(passed=True,optimizer_updates=0,forward_bitwise_equal=True,bce_gradient_nonzero=True,
    stopgrad_equals_fixed_weight_gradient=True,chain_max_error=float((gf-gs-chain).abs().max()),
    fixed_weight_fd=float(fd),fixed_weight_autograd=float(pred),chunk_checks=checks,
    note='Stopgrad intentionally does not differentiate freshly recomputed weights; FD uses fixed baseline weights.')
Path(__file__).with_name('soft4_test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
