"""Bounded fresh100 mu training: 4 whole meshes/update, 200 epochs, no KL."""
import argparse,time,json,random,fcntl,shutil,sys,traceback
from pathlib import Path
from runtime import ROOT,BASE,SEED,T,np,c,b,setup,write,sha,norm,tensor_hash
from evaluate import evaluate_mesh,FACE_SECONDS

CHECK_EPOCHS=[0,10,25,50,100,150,200]
def main(preflight):
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,opt,groups,uids,pools,forward,capture,args=setup()
    active=[p for g in groups.values() for p in g.values()];scales=model.scoring_contract()
    freeze_hash=tensor_hash(model.autoencoder.log_variance.named_parameters())
    initial_hash=tensor_hash(model.named_parameters())
    source_files={}
    for mod in list(sys.modules.values()):
        p=getattr(mod,'__file__',None)
        if p and p.endswith('.py') and str(BASE) in p and Path(p).is_file():
            p=Path(p)
            if ROOT in p.parents:continue
            dest=ROOT/'source_archive'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
            source_files[str(p)]=sha(p)
    manifest=dict(seed=SEED,random_initialization=True,old_model_checkpoint_loaded=False,initial_model_sha256=initial_hash,
        args=args,uids=uids,logical_batch=4,microbatch=1,mesh_reduction='mean of four per-mesh EdgeSoft4+FaceSoft4',
        backend='math00 FP32 MATH E+D, deterministic Graph forward/backward, TF32/autocast off',
        sampling=False,kl=0,logvar='frozen, excluded from optimizer',optimizer='fresh Adam',
        betas=[.9,.999],eps=1e-8,wd=0,global_clip=1,warmup='update1: E1e-6/Dheads1e-5; update100: E1e-5/Dheads1e-4; linear (update-1)/99; constant afterwards',
        epochs=200,updates=5000,participations=20000,checks_epochs=CHECK_EPOCHS,
        evaluation_layout='fixed checkpoint, no updates, 100 sequential complete-mesh forwards; not packed100',
        face_enumeration_seconds_per_mesh=FACE_SECONDS,incomplete_face_policy='FP/F1 null, lower bounds explicit; never strict pass',
        data_ready=json.loads((ROOT/'READY.json').read_text()),source_sha256=source_files,
        entry_sha256={n:sha(ROOT/n) for n in ['train.py','runtime.py','evaluate.py','construction_args.json']},
        groups={n:list(g) for n,g in groups.items()},scales=scales,
        size_restriction='selection locked to4..2600 vertices before model evaluation')
    def objective(uid):
        rows=forward(uid);loss,parts,saved=b.full_objective(rows,[pools[uid]],scales)
        assert T.isfinite(loss) and all(np.isfinite(x).all() for x in saved[0].values())
        return rows,loss,parts[0],saved[0]
    def gradnorms():return {n:norm(p.grad for p in g.values() if p.grad is not None) for n,g in groups.items()}
    if preflight:
        assert not (ROOT/'initial.pt').exists(),'Preflight already saved an initialization'
        out=ROOT/'preflight';out.mkdir(exist_ok=True);write(out/'manifest.json',manifest)
        # Check a simultaneous sum vs separate microbatch backward, with identical scaling.
        opt.zero_grad(set_to_none=True);terms=[]
        for u in uids[:2]:
            rows,loss,_,_=objective(u);terms.append(loss/4)
        T.stack(terms).sum().backward();ref=[p.grad.clone() if p.grad is not None else None for p in active]
        del rows,loss,terms;opt.zero_grad(set_to_none=True)
        for u in uids[:2]:
            rows,loss,_,_=objective(u);(loss/4).backward();del rows,loss
        diff=norm(p.grad-g for p,g in zip(active,ref) if g is not None);den=norm(g for g in ref if g is not None)
        assert diff/max(den,1e-30)<1e-4
        accumulation_relative_error=diff/max(den,1e-30);del ref;opt.zero_grad(set_to_none=True)
        T.cuda.reset_peak_memory_stats();times=[];norms=[];t0=time.monotonic()
        with (out/'traversal.jsonl').open('w',buffering=1) as f:
            for i,u in enumerate(uids):
                if i%4==0:opt.zero_grad(set_to_none=True)
                t=time.monotonic();rows,loss,parts,_=objective(u);(loss/4).backward()
                assert all(T.isfinite(p.grad).all() for p in active if p.grad is not None)
                T.cuda.synchronize();seconds=time.monotonic()-t;times.append(seconds)
                row=dict(uid=u,vertices=len(pools[u]['vertices']),faces=len(pools[u]['positive']),parts=parts,seconds=seconds,
                         microbatch_weight=.25,peak_gib=T.cuda.max_memory_allocated()/2**30)
                if i%4==3:
                    row['group_grad_norms']=gradnorms();row['global_norm']=float(T.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True));norms.append(row['global_norm'])
                f.write(json.dumps(row)+'\n');print('PREFLIGHT_MESH',i+1,u,round(seconds,3),flush=True)
                del rows,loss
        traversal=time.monotonic()-t0;opt.zero_grad(set_to_none=True)
        assert not opt.state and tensor_hash(model.named_parameters())==initial_hash
        # No optimizer.step is used in preflight; estimate excludes update/checkpoint/evaluation overhead.
        initial=dict(model=model.state_dict(),optimizer=opt.state_dict(),args=args,completed_updates=0,epoch=0,
                     python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),
                     initial_model_sha256=initial_hash,manifest=manifest)
        T.save(initial,ROOT/'initial.pt')
        result=dict(passed=True,optimizer_updates=0,all100_full_forward_backward=True,traversal_seconds=traversal,
            estimated_200_epoch_forward_backward_hours=traversal*200/3600,estimate_excludes='Adam/update statistics, checkpoints, evaluation and shared-system variation',
            peak_gib=T.cuda.max_memory_allocated()/2**30,gpu=T.cuda.get_device_name(0),memory_total_gib=T.cuda.get_device_properties(0).total_memory/2**30,
            accumulation_relative_error=accumulation_relative_error,all_weights_unchanged=True,adam_states=0,
            entry_sha256=manifest['entry_sha256'],initial_checkpoint_sha256=sha(ROOT/'initial.pt'),ready_sha256=sha(ROOT/'READY.json'))
        write(out/'result.json',result);print('PREFLIGHT_COMPLETE',json.dumps(result),flush=True);return
    pre=json.loads((ROOT/'preflight/result.json').read_text())
    assert pre['passed'] and pre['entry_sha256']==manifest['entry_sha256'] and pre['ready_sha256']==sha(ROOT/'READY.json')
    assert sha(ROOT/'initial.pt')==pre['initial_checkpoint_sha256']
    initial=T.load(ROOT/'initial.pt',map_location='cpu',mmap=True)
    assert initial['initial_model_sha256']==initial_hash and not opt.state
    # Explicitly restore only this experiment's own fresh step0 state, never historical learned weights.
    assert all(T.equal(p.detach().cpu(),initial['model'][n]) for n,p in model.named_parameters())
    random.setstate(initial['python_rng']);np.random.set_state(initial['numpy_rng']);T.set_rng_state(initial['torch_rng']);T.cuda.set_rng_state_all(initial['cuda_rng'])
    out=ROOT/'run';out.mkdir(exist_ok=True);assert not (out/'updates.jsonl').exists(),'Refuse overwrite of an existing training run'
    write(out/'manifest.json',manifest)
    shuffle=np.random.default_rng(SEED+1);participation={u:0 for u in uids};step=0;epoch=0;first_perfect=False
    def save():
        path=out/f'checkpoint-update{step:05d}.pt';tmp=path.with_suffix('.tmp')
        assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
        assert tensor_hash(model.autoencoder.log_variance.named_parameters())==freeze_hash
        T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),args=args,completed_updates=step,epoch=epoch,
                    participation=participation,shuffle_rng=shuffle.bit_generator.state,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    torch_rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),manifest=manifest),tmp);tmp.replace(path)
        write(out/'latest.json',dict(checkpoint=str(path),updates=step,epoch=epoch));return path
    def evaluate(checkpoint):
        before=tensor_hash(model.named_parameters());result=[];t=time.monotonic()
        write(out/'status.json',dict(state='evaluating',epoch=epoch,updates=step,evaluated_meshes=0))
        with (out/f'eval-epoch{epoch:03d}.jsonl').open('w',buffering=1) as f:
            for i,u in enumerate(uids):
                rows,loss,parts,saved=objective(u);detached=tuple(tuple(x.detach() for x in rr) for rr in rows)
                del rows,loss
                metrics=evaluate_mesh(detached,pools[u],scales)
                fy=np.r_[np.ones(len(pools[u]['positive']),dtype=bool),np.zeros(len(pools[u]['mixed']),dtype=bool)]
                row=dict(uid=u,epoch=epoch,updates=step,participations=participation[u],parts=parts,
                         face_training_pool=c.metrics(fy,saved['face_train_logits']),**metrics)
                result.append(row);f.write(json.dumps(row,allow_nan=False)+'\n');del detached,saved
                write(out/'status.json',dict(state='evaluating',epoch=epoch,updates=step,evaluated_meshes=i+1))
                print('EVAL',epoch,i+1,u,'joint',row['joint_perfect'],'face_complete',row['face']['complete'],flush=True)
        assert tensor_hash(model.named_parameters())==before
        summary=dict(epoch=epoch,updates=step,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-t,
            edge_perfect=sum(x['edge_perfect'] for x in result),face_perfect=sum(x['face_perfect'] is True for x in result),
            joint_perfect=sum(x['joint_perfect'] for x in result),face_incomplete=[x['uid'] for x in result if not x['face']['complete']],
            failed_uids=[x['uid'] for x in result if not x['joint_perfect']],total_edge_fp=sum(x['edge']['fp'] for x in result),
            total_edge_fn=sum(x['edge']['fn'] for x in result),total_face_fn=sum(x['face']['fn'] for x in result),
            total_face_fp=sum(x['face']['fp'] for x in result) if all(x['face']['complete'] for x in result) else None,
            total_face_fp_lower_bound=sum(x['face']['fp_lower_bound'] for x in result),
            all100_complete_and_perfect=all(x['joint_perfect'] and x['face']['complete'] for x in result))
        write(out/f'eval-summary-epoch{epoch:03d}.json',summary);return summary
    try:
        path=save();evaluate(path)
        with (out/'updates.jsonl').open('x',buffering=1) as f:
            for epoch in range(1,201):
                order=shuffle.permutation(uids).tolist();assert len(set(order))==100
                write(out/f'epoch-order-{epoch:03d}.json',dict(epoch=epoch,uids=order))
                for offset in range(0,100,4):
                    batch=order[offset:offset+4];step+=1;t=time.monotonic();opt.zero_grad(set_to_none=True)
                    factor=1+9*min((step-1)/99,1)
                    for pg in opt.param_groups:pg['lr']=(1e-6 if pg['name']=='encoder_mu' else 1e-5)*factor
                    details=[]
                    for u in batch:
                        rows,loss,parts,saved=objective(u);(loss/4).backward()
                        channel=capture['mu'].grad.detach().double().square().sum(0).sqrt()
                        details.append(dict(uid=u,parts=parts,weighted_loss=float(loss.detach())/4,participation=participation[u]+1,
                            mu_gradient_nonzero_channels=int((channel>0).sum()),mu_rms=float(rows[0][0].detach().square().mean().sqrt()),
                            hidden_rms=float(capture['hidden'].square().mean().sqrt()),edge_rms=float(rows[2][0].detach().square().mean().sqrt()),
                            face_rms=float(rows[3][0].detach().square().mean().sqrt())))
                        participation[u]+=1;del rows,loss,saved,channel
                    grads=gradnorms();gn=float(T.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                    old={g:[p.detach().clone() for p in ps.values()] for g,ps in groups.items()}
                    opt.step();movement={}
                    for g,ps in groups.items():
                        pn=norm(old[g]);dn=norm(p.detach()-v for p,v in zip(ps.values(),old[g]));assert np.isfinite(dn)
                        movement[g]=dict(delta_l2=dn,relative_l2=dn/(pn+1e-30))
                    del old
                    row=dict(update=step,epoch=epoch,batch_in_epoch=offset//4+1,mesh_participations=4*step,meshes=details,
                             objective=sum(x['weighted_loss'] for x in details),lr={pg['name']:pg['lr'] for pg in opt.param_groups},
                             gradient_norms=grads,global_norm=gn,clip_coefficient=min(1.,1/(gn+1e-6)),actual_updates=movement,
                             head_weight_norms={k:norm([getattr(model.autoencoder,k+'_embedding').weight]) for k in ['edge','face']},seconds=time.monotonic()-t)
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                    write(out/'status.json',dict(state='training',updates=step,epoch=epoch,batch_in_epoch=offset//4+1,mesh_participations=4*step,last_update_seconds=row['seconds']))
                    if step<=3 or step%25==0:print('UPDATE',step,'EPOCH',epoch,'LOSS',row['objective'],'SECONDS',row['seconds'],flush=True)
                assert set(participation.values())=={epoch}
                if epoch%10==0 or epoch in CHECK_EPOCHS:path=save()
                if epoch in CHECK_EPOCHS:
                    result=evaluate(path)
                    if result['all100_complete_and_perfect'] and not first_perfect:
                        loaded=T.load(path,map_location='cpu',mmap=True);model.load_state_dict(loaded['model'],strict=True)
                        prior=out/f'eval-epoch{epoch:03d}.jsonl';prior.rename(out/f'eval-before-reload-epoch{epoch:03d}.jsonl')
                        again=evaluate(path);assert again['all100_complete_and_perfect'];first_perfect=True
                        write(out/'first100perfect.json',dict(epoch=epoch,update=step,reloaded_checkpoint_verified=True))
        assert step==5000 and set(participation.values())=={200}
        result=dict(state='completed',epochs=200,updates=5000,mesh_participations=20000,per_mesh_participations=participation,
                    final_evaluation=json.loads((out/'eval-summary-epoch200.json').read_text()),stopped_at_budget=True)
        write(out/'complete.json',result);write(out/'status.json',result);print('COMPLETE_5000',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(updates=step,epoch=epoch,error=str(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args();main(args.preflight)
