"""Recover CAD50 at durable step200; verify replay201..287, continue same trajectory to2000."""
import argparse,fcntl,json,random,shutil,sys,time,traceback
from pathlib import Path
from runtime import ROOT,BASE,T,np,b,c,setup,write,sha,norm,tensor_hash
from evaluate import evaluate_mesh

CHECKS=list(range(0,2001,100))
def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=T.get_rng_state(),cuda=T.cuda.get_rng_state_all())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);T.set_rng_state(x['torch']);T.cuda.set_rng_state_all(x['cuda'])

def main(preflight, verify_resume_only=False):
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
    out=ROOT/'run';step=200;start=time.monotonic()
    assert not (ROOT/'complete.json').exists() and not (ROOT/'failure.json').exists()
    source=out/'checkpoint-step0200.pt';source_sha=sha(source)
    anchor=json.loads((out/'eval-step0200.json').read_text())
    assert source_sha==anchor['checkpoint_sha256']=='ea991ab586b69c87cfaa67efe7b8726fe0f06056260e31fcfb4321d5a52131e8'
    cp=T.load(source,map_location='cpu',mmap=True,weights_only=False)
    assert cp['config']==config and cp['completed_updates']==200
    assert set(cp['participation'].values())=={200} and set(cp['participation'])==set(uids)
    model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer'])
    assert all(T.equal(v.detach().cpu(),cp['model'][n]) if T.is_tensor(v) else v==cp['model'][n] for n,v in model.state_dict().items())
    for pg,saved in zip(opt.param_groups,cp['optimizer']['param_groups']):
        assert pg['name']==saved['name'] and pg['lr']==config['lr_target'][pg['name']]
        assert tuple(pg['betas'])==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
        for param,pid in zip(pg['params'],saved['params']):
            for k,v in cp['optimizer']['state'][pid].items():
                actual=opt.state[param][k]
                assert T.equal(actual.detach().cpu(),v) if T.is_tensor(v) else actual==v
    assert len(opt.state)==len(active) and all(int(opt.state[p]['step'])==200 for p in active)
    participation=dict(cp['participation']);restore_rng(cp['rng']);frozen_check()
    parameter_hash=tensor_hash(model.named_parameters());baseline=[]
    for u,expected in zip(uids,anchor['meshes']):
        assert u==expected['uid']
        rows,loss,parts=objective(u);assert parts==expected['parts'],(u,parts,expected['parts'])
        baseline.append(dict(uid=u,parts=parts,exact_saved_step200_loss=True))
        del rows,loss;capture.clear()
    assert tensor_hash(model.named_parameters())==parameter_hash
    restore_rng(cp['rng'])
    verification=dict(source_checkpoint=str(source),sha256=source_sha,completed_updates=200,
        weights_exact=True,adam_exact=True,adam_steps=200,rng_restored=True,
        logvar_unchanged=True,all50_step200_loss_exact=True,checks=baseline,
        optimizer_updates_in_precheck=0,remaining_trajectory_updates=1800,
        replay_expected_updates=list(range(201,288)),resume_entry_sha256=sha(ROOT/'resume_train.py'))
    write(ROOT/'resume_precheck.json',verification)
    print('RESUME_PRECHECK_PASSED',source_sha,flush=True)
    if verify_resume_only:return
    prior_lines=(out/'updates.jsonl').read_text().splitlines(keepends=True)
    prior=[json.loads(x) for x in prior_lines]
    assert [x['update'] for x in prior]==list(range(1,288))
    assert not list(out.glob('checkpoint-step0[3-9]00.pt'))
    archived=ROOT/'interruption_20260921';archived.mkdir(exist_ok=False)
    for p in [out/'updates.jsonl',ROOT/'status.json',ROOT/'console.log',ROOT/'config.json']:
        shutil.copy2(p,archived/p.name)
    write(archived/'manifest.json',{p.name:sha(p) for p in archived.iterdir() if p.is_file()})
    tmp=out/'updates.jsonl.restore-tmp';tmp.write_text(''.join(prior_lines[:200]));tmp.replace(out/'updates.jsonl')
    replay={row['update']:row for row in prior[200:]}
    best=json.loads((out/'best.json').read_text())['joint_perfect'];assert best==13
    first_perfect=None;assert not (out/'first50perfect.json').exists()
    write(ROOT/'recovery.json',dict(**{k:v for k,v in verification.items() if k!='checks'},
        state='replaying',interrupted_after_logged_update=287,durable_restore_update=200,
        historical_updates_without_durable_checkpoint=87,replay_updates_verified=0,
        logical_final_budget=2000,recorded_optimizer_calls_if_complete=2087,
        note='201..287 are deterministic recovery replay, not extra trajectory budget; old logs preserved separately'))
    del cp,prior

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
        path=source
        with (out/'updates.jsonl').open('a',buffering=1) as log:
            for step in range(201,2001):
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
                if step in replay:
                    expected={k:v for k,v in replay[step].items() if k!='seconds'}
                    actual={k:v for k,v in record.items() if k!='seconds'}
                    if actual!=expected:
                        write(ROOT/'replay_mismatch.json',dict(update=step,expected=expected,actual=actual))
                        raise RuntimeError(f'Replay differs at update {step}; saved evidence, no automatic reset')
                log.write(json.dumps(record,allow_nan=False)+'\n')
                if step==287:
                    replay_checkpoint=save('checkpoint-replayed0287.pt')
                    recovery=json.loads((ROOT/'recovery.json').read_text())
                    recovery.update(state='replay_verified_continuing',replay_updates_verified=87,
                        replay_matches='all record fields exact except wall-clock seconds',
                        replay_checkpoint=str(replay_checkpoint),replay_checkpoint_sha256=sha(replay_checkpoint))
                    write(ROOT/'recovery.json',recovery)
                    print('RECOVERY_REPLAY_87_UPDATES_EXACT',flush=True)
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
        completion['recovery']=json.loads((ROOT/'recovery.json').read_text())
        completion['recovery']['state']='complete'
        completion['recorded_optimizer_calls_including_interrupted_replay']=2087
        write(ROOT/'recovery.json',completion['recovery'])
        write(ROOT/'complete.json',completion);write(ROOT/'status.json',completion);print('COMPLETE_2000',flush=True)
    except BaseException as exc:
        write(ROOT/'failure.json',dict(step=step,error=str(exc),traceback=traceback.format_exc(),participation=participation))
        save('failure-state.pt');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');args=p.parse_args();main(False,args.verify_only)
