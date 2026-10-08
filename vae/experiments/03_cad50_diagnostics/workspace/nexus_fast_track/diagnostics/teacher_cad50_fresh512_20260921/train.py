"""CAD50 fresh512: 2000 complete-50-mesh gradient accumulations, then stop."""
import argparse,fcntl,json,random,shutil,sys,time,traceback
from pathlib import Path
from runtime import ROOT,BASE,T,np,b,c,setup,write,sha,norm,tensor_hash
from evaluate import evaluate_mesh

CHECKS=list(range(0,2001,100))
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=T.get_rng_state(),cuda=T.cuda.get_rng_state_all())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);T.set_rng_state(x['torch']);T.cuda.set_rng_state_all(x['cuda'])

def main(preflight):
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,opt,groups,uids,pools,forward,objective,capture,args,batches=setup()
    active=[p for g in groups.values() for p in g.values()];scales=model.scoring_contract()
    frozen_hash=tensor_hash(model.autoencoder.log_variance.named_parameters())
    init_hash=tensor_hash(model.named_parameters());participation={u:0 for u in uids}
    config=dict(experiment='Teacher original CAD50 targets x current512 AE strict overfit',seed=0,data_seed=0,pool_seed=0,
        initialization='normal constructor seed0; no learned weights loaded',full_network_training=True,
        trainable_groups={g:list(ps) for g,ps in groups.items()},trainable_parameter_counts={g:sum(p.numel() for p in ps.values()) for g,ps in groups.items()},
        args=args,scales=scales,uids=uids,updates=2000,microbatch=1,meshes_per_optimizer_update=50,
        objective='sum50(EdgeSoft4+FaceSoft4)/50',soft4=dict(membership_detach=False,numerator_detach=False,denominator_detach=False,tau=1,epsilon=1e-8,internal_divisor=4,extra_outer_quarter=False,reduction='FP32'),
        warmup='lr_g(t)=target_g*min(t/100,1), t=1..2000; no optimizer update in preflight',
        lr_target={g:1e-5 if g=='encoder_mu' else 1e-4 for g in groups},betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,
        sampling=False,KL=0,logvar_frozen=True,model_mode='eval with gradients enabled; latent input asserted exactly equal to mu',
        backend='unchanged math00 deterministic Graph and FP32 SDPA MATH E+D; TF32/autocast off; matching nonreentrant recompute contexts',
        traversal='fixed teacher_cad50_00..49, each once before one clip and Adam',checkpoints=CHECKS,
        actual_face='complete enumeration of predicted Edge graph triangles, no repair/sampling/truncation',
        data_manifest_sha256=sha(ROOT/'data/manifest.json'),pool_manifest_sha256=sha(ROOT/'pool_manifest.json'),
        source_archive=json.loads((ROOT/'source_archive.json').read_text()),initial_weights_hash=init_hash,
        parameter_loading='only own freshly initialized step0 checkpoint after preflight; no teacher or previous-run weights',
        entry_sha256={n:sha(ROOT/n) for n in ['train.py','runtime.py','loader.py','evaluate.py','effective_loss_and_scoring.py','construction_args.json']})
    sources={}
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if raw and raw.endswith('.py') and str(BASE) in raw and Path(raw).is_file():
            p=Path(raw)
            if ROOT in p.parents:continue
            dest=ROOT/'source_archive'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            sources[str(p)]=sha(p)
    config['effective_source_sha256']=sources
    def grads():return {g:norm(p.grad for p in ps.values() if p.grad is not None) for g,ps in groups.items()}
    def frozen_check():
        assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
        assert tensor_hash(model.autoencoder.log_variance.named_parameters())==frozen_hash
    if preflight:
        assert not (ROOT/'initial.pt').exists()
        out=ROOT/'preflight';out.mkdir(exist_ok=True);write(ROOT/'config.json',config)
        initial_rng=rng_state();opt.zero_grad(set_to_none=True);records=[];started=time.monotonic();T.cuda.reset_peak_memory_stats()
        for i,u in enumerate(uids):
            start=time.monotonic();rows,loss,parts=objective(u)
            first=tuple(tuple(x.detach().clone() for x in group) for group in rows)
            scalar=float(loss.detach());del rows,loss
            rows,loss,parts2=objective(u)
            assert scalar==float(loss.detach()) and parts==parts2
            assert all(T.equal(x,y) for group,ref in zip(rows,first) for x,y in zip(group,ref))
            if i in [0,13,20]:
                old,oldparts,_=b.full_objective(rows,[pools[u]],scales)
                assert T.equal(loss,old) and oldparts[0]==parts
                newg=T.autograd.grad(loss,(rows[2][0],rows[3][0]),retain_graph=True)
                oldg=T.autograd.grad(old,(rows[2][0],rows[3][0]),retain_graph=True)
                assert all(T.equal(x,y) for x,y in zip(newg,oldg));del old,newg,oldg
            (loss/50).backward()
            channel=capture['mu'].grad.detach().abs().sum(0)
            assert channel.shape==(512,) and T.isfinite(channel).all()
            rec=dict(uid=u,parts=parts,repeat_forward_bitwise=True,mu_equals_z=True,mu_gradient_channels=int((channel>0).sum()),seconds=time.monotonic()-start)
            records.append(rec);print('PREFLIGHT',i+1,u,rec['seconds'],flush=True)
            del rows,loss,first,channel;capture.clear()
        missing=[n for g in groups.values() for n,p in g.items() if p.grad is None];assert not missing,missing
        assert all(T.isfinite(p.grad).all() for p in active)
        gn=grads();assert all(v>0 for v in gn.values())
        frozen_check();assert init_hash==tensor_hash(model.named_parameters()) and not opt.state
        opt.zero_grad(set_to_none=True);restore_rng(initial_rng)
        T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),config=config,completed_updates=0,participation=participation),ROOT/'initial.pt')
        result=dict(passed=True,optimizer_updates=0,weights_unchanged=True,optimizer_states=0,
            all50_forward_backward=True,all50_repeated_forward_bitwise=True,mu_path=True,logvar_unchanged=True,
            all_trainable_parameters_connected=True,gradient_norms=gn,reference_loss_and_embedding_gradients_bitwise=[uids[i] for i in [0,13,20]],
            seconds=time.monotonic()-started,per_mesh=records,peak_gpu_gib=T.cuda.max_memory_allocated()/2**30,
            gpu=T.cuda.get_device_name(0),initial_checkpoint_sha256=sha(ROOT/'initial.pt'),entry_sha256=config['entry_sha256'])
        write(out/'result.json',result);write(ROOT/'status.json',dict(state='preflight_passed',optimizer_updates=0));print('PREFLIGHT COMPLETE',flush=True);return
    pre=json.loads((ROOT/'preflight/result.json').read_text());oldconfig=json.loads((ROOT/'config.json').read_text())
    assert pre['passed'] and pre['entry_sha256']==config['entry_sha256']
    assert config==oldconfig and sha(ROOT/'initial.pt')==pre['initial_checkpoint_sha256']
    initial=T.load(ROOT/'initial.pt',map_location='cpu',mmap=True,weights_only=False)
    assert all(T.equal(p.detach().cpu(),initial['model'][n]) for n,p in model.named_parameters())
    assert not initial['optimizer']['state'] and not opt.state;restore_rng(initial['rng']);del initial
    out=ROOT/'run';out.mkdir(exist_ok=False);step=0;best=-1;first_perfect=None;start=time.monotonic()
    def save(tag=None):
        frozen_check();path=out/(tag or f'checkpoint-step{step:04d}.pt');tmp=path.with_suffix('.tmp')
        T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),config=config,
            completed_updates=step,participation=participation),tmp);tmp.replace(path)
        return path
    def evaluate(path):
        nonlocal best,first_perfect
        before=tensor_hash(model.named_parameters());before_rng=rng_state();result=[];t=time.monotonic()
        predictions=out/f'predictions-step{step:04d}';predictions.mkdir(exist_ok=False)
        for i,u in enumerate(uids):
            rows,loss,parts=objective(u);detached=tuple(tuple(x.detach() for x in group) for group in rows)
            del rows,loss;capture.clear()
            p=predictions/f'{u}.npz';metrics=evaluate_mesh(detached,pools[u],scales,p)
            assert metrics['face']['complete']
            metrics['face']['actual_fp_inside_training_pool']=metrics['face']['fp']-metrics['face']['actual_fp_outside_training_pool']
            metrics['face']['fn_missing_candidate']=metrics['missing_gt_face_candidates']
            metrics['face']['fn_present_but_negative']=metrics['face']['fn']-metrics['missing_gt_face_candidates']
            result.append(dict(uid=u,vertices=len(pools[u]['vertices']),step=step,participations=participation[u],parts=parts,
                prediction_path=str(p.relative_to(ROOT)),prediction_sha256=sha(p),**metrics))
            write(ROOT/'status.json',dict(state='evaluating',updates=step,evaluated=i+1,total=50))
            del detached
        counts={kind:{k:sum(r[kind][k] for r in result) for k in ['tp','fp','fn','tn']} for kind in ['edge','face']}
        for kind,d in counts.items():d['micro_f1']=2*d['tp']/max(2*d['tp']+d['fp']+d['fn'],1)
        summary=dict(step=step,checkpoint=str(path),checkpoint_sha256=sha(path),seconds=time.monotonic()-t,
            losses={k:sum(r['parts'][k] for r in result)/50 for k in ['edge','face']},counts=counts,
            edge_perfect=sum(r['edge_perfect'] for r in result),face_perfect=sum(r['face_perfect'] for r in result),
            joint_perfect=sum(r['joint_perfect'] for r in result),perfect_uids=[r['uid'] for r in result if r['joint_perfect']],
            missing_gt_face_candidates=sum(r['missing_gt_face_candidates'] for r in result),
            face_fn_present_but_negative=sum(r['face']['fn_present_but_negative'] for r in result),
            face_fp_inside_pool=sum(r['face']['actual_fp_inside_training_pool'] for r in result),
            face_fp_outside_pool=sum(r['face']['actual_fp_outside_training_pool'] for r in result),
            special_cases={u:next(r for r in result if r['uid']==u) for u in [uids[0],uids[13],uids[20]]},meshes=result)
        assert tensor_hash(model.named_parameters())==before;frozen_check()
        assert T.equal(before_rng['torch'],T.get_rng_state()) and all(T.equal(a,z) for a,z in zip(before_rng['cuda'],T.cuda.get_rng_state_all()))
        write(out/f'eval-step{step:04d}.json',summary)
        if summary['joint_perfect']>best:
            best=summary['joint_perfect'];write(out/'best.json',dict(step=step,joint_perfect=best,checkpoint=str(path),sha256=summary['checkpoint_sha256']))
        if best==50 and first_perfect is None:
            first_perfect=step;write(out/'first50perfect.json',dict(step=step,checkpoint=str(path),sha256=summary['checkpoint_sha256'],verified_checkpoint_only=True))
        print('EVAL_COMPLETE',step,json.dumps({k:v for k,v in summary.items() if k not in ['meshes','special_cases']}),flush=True)
        return summary
    try:
        path=save();evaluate(path)
        with (out/'updates.jsonl').open('x',buffering=1) as log:
            for step in range(1,2001):
                t=time.monotonic();opt.zero_grad(set_to_none=True)
                for pg in opt.param_groups:pg['lr']=config['lr_target'][pg['name']]*min(step/100,1)
                details=[]
                for u in uids:
                    rows,loss,parts=objective(u);(loss/50).backward();participation[u]+=1
                    details.append(dict(uid=u,participations=participation[u],**parts))
                    del rows,loss;capture.clear()
                assert set(participation.values())=={step}
                group_norms=grads();gn=float(T.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                before={g:[p.detach().clone() for p in ps.values()] for g,ps in groups.items()}
                opt.step();updates={}
                for g,ps in groups.items():
                    delta=norm(p.detach()-v for p,v in zip(ps.values(),before[g]));base=norm(before[g])
                    assert np.isfinite(delta)
                    updates[g]=dict(delta_l2=delta,relative_l2=delta/max(base,1e-30))
                del before
                assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
                record=dict(update=step,meshes=details,mesh_count=50,objective_before=sum(r['edge']+r['face'] for r in details)/50,
                    losses_before={k:sum(r[k] for r in details)/50 for k in ['edge','face']},gradient_norms=group_norms,
                    total_grad_norm=gn,clip_coefficient=min(1.,1/(gn+1e-6)),actual_updates=updates,
                    lrs={g['name']:g['lr'] for g in opt.param_groups},seconds=time.monotonic()-t)
                log.write(json.dumps(record,allow_nan=False)+'\n')
                write(ROOT/'status.json',dict(state='training',updates=step,budget=2000,per_mesh_participations=step,last_update_seconds=record['seconds'],last_loss=record['objective_before']))
                if step<=3 or step%10==0:print('UPDATE',step,record['objective_before'],record['seconds'],flush=True)
                if step in CHECKS:path=save();summary=evaluate(path)
        assert step==2000 and set(participation.values())=={2000}
        final=T.load(path,map_location='cpu',mmap=True,weights_only=False);model.load_state_dict(final['model'],strict=True)
        assert all(int(opt.state[p]['step'])==2000 for p in active)
        model_path=out/'model-step2000-inference.pt';T.save(dict(model=model.state_dict(),args=args,config=config,completed_updates=2000),model_path)
        completion=dict(state='complete',updates=2000,mesh_participations=100000,stopped_at_budget=True,
            per_mesh_participations=participation,final_summary={k:v for k,v in summary.items() if k not in ['meshes','special_cases']},
            first50perfect_step=first_perfect,best=json.loads((out/'best.json').read_text()),
            full_model=str(model_path),full_model_sha256=sha(model_path),seconds=time.monotonic()-start)
        write(ROOT/'complete.json',completion);write(ROOT/'status.json',completion);print('COMPLETE_2000',flush=True)
    except BaseException as exc:
        write(ROOT/'failure.json',dict(step=step,error=str(exc),traceback=traceback.format_exc(),participation=participation))
        save('failure-state.pt');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');args=p.parse_args();main(args.preflight)
