"""Measure all-100 Edge scoring/backward without any parameter update."""
from head_core import ROOT,T,np,load,forward,metrics,summary,cycle,write,norm,sha
import time,json
cfg,meta,head,data,scoring=load();scale=meta['scales']['edge_logit_scale']
before=[p.detach().clone() for p in head.parameters()]
T.cuda.synchronize();resident=T.cuda.memory_allocated();records=[]
# Verify standalone cached execution against exact network-exported values and gradients.
for d in data:
    values=forward(head,d,scoring,scale);row=metrics(d,values);ref=d['reference']
    with np.load(ROOT/'cache'/f"{d['uid']}.npz") as a:
        assert np.array_equal(values[0].detach().cpu().numpy(),a['original_edge_raw'])
        assert np.array_equal(values[1].detach().cpu().numpy(),a['original_edge_centered'])
        grads=T.autograd.grad(values[3]/100,tuple(head.parameters()))
        assert np.array_equal(grads[0].cpu().numpy(),a['original_weight_gradient'])
        assert np.array_equal(grads[1].cpu().numpy(),a['original_bias_gradient'])
    assert all(row[k]==ref['counts'][k] for k in ['tp','fp','fn','tn'])
    assert row['edge_soft4']==ref['edge_loss']
    records.append(row);del values,grads
write(ROOT/'baseline_verification.json',dict(all100_raw_center_loss_gradient_bitwise=True,baseline=summary(records,0)))
T.cuda.synchronize();T.cuda.reset_peak_memory_stats();t=time.monotonic()
scored=cycle(head,data,scoring,scale,False);T.cuda.synchronize();forward_seconds=time.monotonic()-t
assert scored==records
times=[];grads0=None
for repeat in range(3):
    head.zero_grad(set_to_none=True);T.cuda.synchronize();t=time.monotonic()
    rows=cycle(head,data,scoring,scale,True);T.cuda.synchronize();times.append(time.monotonic()-t)
    assert rows==records
    gradients=[p.grad.detach().clone() for p in head.parameters()]
    if grads0 is None:grads0=gradients
    else:assert all(T.equal(a,b) for a,b in zip(grads0,gradients))
assert all(T.equal(a,b) for a,b in zip(before,head.parameters()))
result=dict(meshes=100,vertices=meta['total_vertices'],all_pairs=meta['total_pairs'],hidden_bytes=meta['hidden_total_bytes'],
    resident_gpu_bytes=resident,peak_allocated_gpu_bytes=T.cuda.max_memory_allocated(),peak_reserved_gpu_bytes=T.cuda.max_memory_reserved(),
    forward_all100_seconds=forward_seconds,forward_backward_all100_seconds=times,mean_forward_backward_seconds=sum(times)/len(times),
    total_gradient_norm=(sum(norm(g)**2 for g in grads0))**.5,repeat_forward_backward_bitwise=True,parameters_unchanged=True,
    optimizer_updates=0,torch=T.__version__,cuda=T.version.cuda,gpu=T.cuda.get_device_name(0),
    no_upstream_network_in_benchmark=True,objective='mean100(Edge Soft4), no extra outer 1/4')
write(ROOT/'benchmark.json',result);write(ROOT/'status.json',dict(state='benchmark_complete_budget_not_started',optimizer_updates=0))
print('BENCHMARK',json.dumps(result),flush=True)
