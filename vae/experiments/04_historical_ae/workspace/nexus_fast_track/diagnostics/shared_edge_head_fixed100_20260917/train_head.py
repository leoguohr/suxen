"""Bounded all-100 shared-head fit, starting from the original source head."""
import fcntl,json,time
from head_core import ROOT,T,np,load,forward,metrics,summary,cycle,write,norm,sha

lock=(ROOT/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (ROOT/'benchmark.json').exists()
plan=json.loads((ROOT/'training_plan.json').read_text());cfg,meta,head,data,scoring=load()
assert plan['all100_per_update'] and plan['objective']==cfg['objective']
out=ROOT/'run';out.mkdir(exist_ok=False)
assert sha(ROOT/'cache/head_original.npz')==meta['head_sha256']
original={k:v.detach().clone() for k,v in head.state_dict().items()}
params=list(head.parameters());scale=meta['scales']['edge_logit_scale'];budget=plan['updates']
opt=T.optim.Adam(params,lr=plan['lr'],betas=(.9,.999),eps=1e-8,weight_decay=0)
assert not opt.state
manifest=dict(config=cfg,plan=plan,head_instances=1,trainable_parameters=sum(p.numel() for p in params),
    uids=[d['uid'] for d in data],hidden_requires_grad=False,all_pairs_every_mesh_every_update=True,
    initialization='original epoch900 head',upstream_network_loaded=False,face_or_kl_or_distillation=False,
    optimizer='fresh Adam',betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,
    cache_manifest_sha256=sha(ROOT/'cache/manifest.json'),source_sha256=cfg['source_sha256'],
    code_sha256={n:sha(ROOT/n) for n in ['head_core.py','train_head.py','config.json','training_plan.json']},
    torch=T.__version__,cuda=T.version.cuda,gpu=T.cuda.get_device_name(0))
assert manifest['trainable_parameters']==32800
write(out/'manifest.json',manifest)
initial_perfect=None;ever_perfect=set();first_all=None;count_all=streak=longest=0;history=[];start=time.monotonic()
def save(step,record,name):
    T.save(dict(step=step,head={k:v.detach().cpu() for k,v in head.state_dict().items()},optimizer=opt.state_dict(),
        metrics=record,manifest=manifest,torch_rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all()),out/(name+'.pt'))
    write(out/(name+'.json'),record)

with (out/'evaluations.jsonl').open('x',buffering=1) as ev,(out/'updates.jsonl').open('x',buffering=1) as up:
    for step in range(budget+1):
        opt.zero_grad(set_to_none=True)
        rows=cycle(head,data,scoring,scale,step<budget)
        record=summary(rows,step);perfect={r['uid'] for r in rows if r['perfect']};ever_perfect|=perfect
        if initial_perfect is None:
            initial_perfect=perfect
            expected=json.loads((ROOT/'baseline_verification.json').read_text())['baseline']
            assert record==expected
        record.update(initial_success_retained=len(perfect&initial_perfect),initial_success_lost=len(initial_perfect-perfect),
            new_success_count=len(perfect-initial_perfect),elapsed_seconds=time.monotonic()-start)
        history.append(record);ev.write(json.dumps(record,allow_nan=False)+'\n')
        if step in plan['checks']:save(step,record,f'checkpoint-step{step:04d}')
        if record['all100_perfect']:
            if first_all is None:first_all=step;save(step,record,'first-all100-perfect')
            if step>0:count_all+=1;streak+=1;longest=max(longest,streak)
        else:streak=0
        if step==budget:break
        before=[p.detach().clone() for p in params]
        gradnorms={k:norm(p.grad) for k,p in head.named_parameters()}
        gn=float(T.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        opt.step()
        updates={k:dict(l2=norm(p.detach()-old),relative_l2=norm(p.detach()-old)/max(norm(old),1e-30),
            changed_elements=int((p.detach()!=old).sum()),norm=norm(p)) for (k,p),old in zip(head.named_parameters(),before)}
        u=dict(update=step+1,meshes=100,mesh_coefficient=0.01,objective_before=record['objective'],gradient_norms=gradnorms,
            global_gradient_norm=gn,clip_coefficient=min(1.,1./(gn+1e-6)),parameter_updates=updates,
            weight_norm_ratio=norm(head.weight)/norm(original['weight']),elapsed_seconds=time.monotonic()-start)
        assert all(T.isfinite(p).all() for p in params)
        up.write(json.dumps(u,allow_nan=False)+'\n')
        if step%25==0:
            write(ROOT/'status.json',dict(state='training',completed_updates=step+1,last_evaluated_step=step,
                edge_perfect=record['edge_perfect'],fp=record['total_fp'],fn=record['total_fn'],budget=budget))
            print('UPDATE',step+1,'evaluated_at',step,'perfect',record['edge_perfect'],'fp',record['total_fp'],'fn',record['total_fn'],
                'loss',record['objective'],'seconds',time.monotonic()-start,flush=True)
    final=history[-1];final_set={r['uid'] for r in final['meshes'] if r['perfect']}
    complete=dict(updates=budget,baseline=history[0],final=final,first_all100_perfect_step=first_all,
        all100_perfect_updates=count_all,longest_all100_perfect=longest,
        last500_all100_perfect=sum(r['all100_perfect'] for r in history[-min(500,budget):]),
        ever_edge_perfect_uids=sorted(ever_perfect),initial_success_retained=sorted(final_set&initial_perfect),
        initial_success_lost=sorted(initial_perfect-final_set),new_final_success=sorted(final_set-initial_perfect),
        all100_equal_weight_every_update=True,participations_per_mesh=budget,
        original_network_updates=0,head_instances=1,wall_seconds=time.monotonic()-start,
        final_weight_norm=norm(head.weight),initial_weight_norm=norm(original['weight']),stopped_at_budget=True)
    write(out/'complete.json',complete)
write(ROOT/'status.json',dict(state='training_complete_pending_verification',completed_updates=budget))
print('TRAINING_COMPLETE',budget,final['edge_perfect'],final['total_fp'],final['total_fn'],flush=True)
