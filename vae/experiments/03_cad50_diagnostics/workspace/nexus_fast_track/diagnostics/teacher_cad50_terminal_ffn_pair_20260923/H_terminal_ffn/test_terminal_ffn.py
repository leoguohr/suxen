"""CPU checks for zero-init identity, gradients, Adam extension, hook reload.

These are small synthetic tests, not training updates to the real experiment.
The full-network/all50 startup gate is executed by train.py before release.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import hashlib
import io
import json
from pathlib import Path
import torch as T
from terminal_ffn import CONTRACT,install_terminal_ffn,load_extended_state

ROOT=Path(__file__).resolve().parent


class ToyAE(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.before=T.nn.Linear(16,1024)
        self.decoder_output_norm=T.nn.LayerNorm(1024)
        self.after=T.nn.Linear(1024,7)

    def forward(self,x):
        return self.after(self.decoder_output_norm(self.before(x)))


class ToyModel(T.nn.Module):
    def __init__(self):
        super().__init__()
        self.autoencoder=ToyAE()

    def forward(self,x):
        return self.autoencoder(x)


def equal(a,b):
    if T.is_tensor(a):return T.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    T.set_num_threads(1);T.manual_seed(2718)
    model=ToyModel().eval();x=T.randn(8,16);target=T.randn(8,7)
    old_parameters=list(model.parameters())
    opt=T.optim.Adam([dict(params=old_parameters,name='old',lr=3e-5)],betas=(.9,.999),eps=1e-8,weight_decay=0)
    # Seed a real Adam history on the toy only, then prove it is unchanged.
    (model(x)-target).square().mean().backward();opt.step();opt.zero_grad(set_to_none=True)
    original_optimizer=opt.state_dict()
    original_output=model(x)
    old_grad=T.autograd.grad((original_output-target).square().mean(),old_parameters)
    rng=T.get_rng_state().clone()
    ffn=install_terminal_ffn(model.autoencoder)
    assert not T.equal(rng,T.get_rng_state())
    opt.add_param_group(dict(params=list(ffn.parameters()),name='terminal_ffn',lr=3e-5,
                            betas=(.9,.999),eps=1e-8,weight_decay=0))
    T.set_rng_state(rng)
    assert T.equal(T.get_rng_state(),rng)
    current=opt.state_dict()
    assert equal(current['state'],original_optimizer['state'])
    assert equal(current['param_groups'][:1],original_optimizer['param_groups'])
    assert all(p not in opt.state or not opt.state[p] for p in ffn.parameters())
    new_output=model(x)
    assert T.equal(original_output,new_output)
    assert original_output.detach().numpy().tobytes()==new_output.detach().numpy().tobytes()
    active=old_parameters+list(ffn.parameters())
    gradients=T.autograd.grad((new_output-target).square().mean(),active)
    assert all(T.equal(a,b) for a,b in zip(gradients,old_grad))
    new_grad=dict(zip(dict(ffn.named_parameters()),gradients[len(old_parameters):]))
    assert new_grad['out_proj.weight'].count_nonzero()>0
    assert all(g.count_nonzero()==0 for n,g in new_grad.items() if not n.startswith('out_proj.'))
    for p,g in zip(active,gradients):p.grad=g
    T.nn.utils.clip_grad_norm_(active,1.);opt.step();opt.zero_grad(set_to_none=True)
    (model(x)-target).square().mean().backward()
    assert ffn.in_proj.weight.grad.count_nonzero()>0 and ffn.norm.weight.grad.count_nonzero()>0
    assert {int(opt.state[p]['step']) for p in old_parameters}=={2}
    assert {int(opt.state[p]['step']) for p in ffn.parameters()}=={1}
    # Checkpoint round trip must reinstall placement, not merely store weights.
    data=io.BytesIO()
    T.save(dict(model=model.state_dict(),config=dict(terminal_ffn=CONTRACT)),data);data.seek(0)
    restored=ToyModel().eval()
    load_extended_state(restored,T.load(data,map_location='cpu',weights_only=True))
    assert T.equal(restored(x),model(x))
    try:
        install_terminal_ffn(restored.autoencoder)
        raise RuntimeError('Duplicate hook accepted')
    except AssertionError:
        pass
    assert not T.cuda.is_initialized()
    result=dict(passed=True,device='cpu',torch=T.__version__,author_model='gpt-6-astra',reasoning_effort='xhigh',
        real_experiment_optimizer_updates=0,toy_optimizer_updates=2,
        exact_zero_init_output_bytes=True,exact_old_gradients_before_clip=True,
        old_adam_state_preserved=True,fresh_ffn_state_empty=True,parent_cpu_rng_restored=True,
        first_ffn_hidden_and_norm_gradients_zero=True,first_ffn_out_proj_weight_gradient_nonzero=True,
        second_backward_in_proj_and_norm_gradients_nonzero=True,old_new_adam_counters_separate=True,
        checkpoint_reload_reinstalls_hook=True,duplicate_hook_rejected=True,
        parameter_count=sum(p.numel() for p in ffn.parameters()),contract=CONTRACT,
        code_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['terminal_ffn.py','test_terminal_ffn.py']})
    (ROOT/'terminal_ffn_test.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
