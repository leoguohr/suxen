"""Bounded continuation: resume update5000, stop at10000, identical reconstruction training."""
import time,json,random,fcntl,traceback
from pathlib import Path
from runtime import ROOT,BASE,T,np,c,b,setup,write,sha,norm,tensor_hash
from evaluate import evaluate_mesh,FACE_SECONDS

PARENT=BASE/'diagnostics/math00_overfit100_fresh_20260914'
SOURCE=PARENT/'run/checkpoint-update05000.pt'
SOURCE_SHA='6a652045b865b776ca9bfbddd68bf1fc454c1fe29ec5aa83dec6cc3629186457'
CHECK_EPOCHS=[200,250,300,350,400]

def same(a,b):
    if T.is_tensor(a):return T.is_tensor(b) and a.dtype==b.dtype and T.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def rng_snapshot(shuffle):
    import copy
    return dict(python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=T.get_rng_state(),
                cuda_rng=T.cuda.get_rng_state_all(),shuffle_rng=copy.deepcopy(shuffle.bit_generator.state))

def restore_rng(state,shuffle):
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);T.set_rng_state(state['torch_rng'])
    T.cuda.set_rng_state_all(state['cuda_rng']);shuffle.bit_generator.state=state['shuffle_rng']

def main():
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'run';out.mkdir(exist_ok=True)
    assert not (out/'updates.jsonl').exists(),'Refuse overwrite or duplicate continuation'
    assert sha(SOURCE)==SOURCE_SHA,'Parent checkpoint differs from recorded epoch200 checkpoint'
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==5000 and cp['epoch']==200
    old_manifest=cp['manifest']
    for p,digest in old_manifest['source_sha256'].items():assert sha(p)==digest,p
    model,opt,groups,uids,pools,forward,capture,args=setup()
    assert uids==old_manifest['uids'] and cp['participation'].keys()==set(uids)
    assert set(cp['participation'].values())=={200}
    assert {n:list(g) for n,g in groups.items()}==old_manifest['groups']
    assert [g['name'] for g in opt.param_groups]==[g['name'] for g in cp['optimizer']['param_groups']]
    model.load_state_dict(cp['model'],strict=True)
    opt.load_state_dict(cp['optimizer'])
    assert same(model.state_dict(),cp['model']) and same(opt.state_dict(),cp['optimizer'])
    for pg in opt.param_groups:
        assert abs(pg['lr']-(1e-5 if pg['name']=='encoder_mu' else 1e-4))<1e-18
        assert pg['betas']==(.9,.999) and pg['eps']==1e-8 and pg['weight_decay']==0
    assert all(p not in opt.state for p in model.autoencoder.log_variance.parameters())
    active=[p for g in groups.values() for p in g.values()];scales=model.scoring_contract()
    assert scales==old_manifest['scales']
    freeze_hash=tensor_hash(model.autoencoder.log_variance.named_parameters())
    shuffle=np.random.default_rng();restore_rng(cp,shuffle)
    assert same(rng_snapshot(shuffle),{k:cp[k] for k in rng_snapshot(shuffle)})
    participation=dict(cp['participation']);step=5000;epoch=200;first_perfect=False
    import copy
    manifest=copy.deepcopy(old_manifest)
    manifest.update(random_initialization=False,initialization='resume own fresh100 epoch200 checkpoint',
        optimizer='restored four-group Adam, all moments and counters preserved',
        parent_checkpoint=str(SOURCE),parent_sha256=SOURCE_SHA,start_epoch=200,start_update=5000,
        epochs=400,updates=10000,additional_epochs=200,additional_updates=5000,participations=40000,
        checks_epochs=CHECK_EPOCHS,warmup='none in continuation; retain terminal parent LR',
        entry_sha256={n:sha(ROOT/n) for n in ['train_continue.py','runtime.py','evaluate.py','construction_args.json']})
    manifest['args']=dict(cp['args'],output=str(ROOT),steps=10000)
    args=manifest['args']
    write(out/'manifest.json',manifest)
    write(out/'resume_verification.json',dict(parent_checkpoint=str(SOURCE),sha256=SOURCE_SHA,
        model_bitwise_equal=True,adam_all_states_bitwise_equal=True,rng_all_states_equal=True,
        uid_order_equal=True,participations_equal=True,parameter_group_names_equal=True,
        lrs={pg['name']:pg['lr'] for pg in opt.param_groups},
        adam_state_count=len(opt.state),logvar_frozen=True,optimizer_updates_at_verification=0,
        parent_source_hashes_verified=True,logvar_parameter_hash=freeze_hash))
    del cp
    def objective(uid):
        rows=forward(uid);loss,parts,saved=b.full_objective(rows,[pools[uid]],scales)
        assert T.isfinite(loss) and all(np.isfinite(x).all() for x in saved[0].values())
        return rows,loss,parts[0],saved[0]
    def gradnorms():return {n:norm(p.grad for p in g.values() if p.grad is not None) for n,g in groups.items()}
    def save():
        path=out/f'checkpoint-update{step:05d}.pt';tmp=path.with_suffix('.tmp')
        assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
        assert tensor_hash(model.autoencoder.log_variance.named_parameters())==freeze_hash
        T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),args=args,completed_updates=step,epoch=epoch,
                    participation=participation,shuffle_rng=shuffle.bit_generator.state,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                    torch_rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),manifest=manifest),tmp);tmp.replace(path)
        write(out/'latest.json',dict(checkpoint=str(path),updates=step,epoch=epoch));return path
    def evaluate(checkpoint):
        rng=rng_snapshot(shuffle)
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
        restore_rng(rng,shuffle);assert same(rng_snapshot(shuffle),rng)
        write(out/f'eval-summary-epoch{epoch:03d}.json',summary);return summary
    try:
        path=SOURCE;evaluate(path)
        old_rows=[json.loads(s) for s in (PARENT/'run/eval-epoch200.jsonl').read_text().splitlines()]
        new_rows=[json.loads(s) for s in (out/'eval-epoch200.jsonl').read_text().splitlines()]
        mismatches=[]
        for old,new in zip(old_rows,new_rows):
            for k in ['uid','parts','edge','face_training_pool','gt_face_candidates','missing_gt_face_candidates','margins','joint_perfect']:
                if old[k]!=new[k]:mismatches.append(dict(uid=old['uid'],field=k,old=old[k],new=new[k]))
            if old['face']!=new['face']:mismatches.append(dict(uid=old['uid'],field='face',old=old['face'],new=new['face']))
        write(out/'baseline_comparison.json',dict(meshes=len(new_rows),exact_match=not mismatches,mismatches=mismatches))
        assert len(new_rows)==len(old_rows)==100 and not mismatches,'Baseline differs; do not begin updates'
        write(out/'training_ready.json',dict(baseline_verified=True,start_update=5000,end_update=10000))
        print('RESUME_BASELINE_VERIFIED',flush=True)
        with (out/'updates.jsonl').open('x',buffering=1) as f:
            for epoch in range(201,401):
                order=shuffle.permutation(uids).tolist();assert len(set(order))==100
                write(out/f'epoch-order-{epoch:03d}.json',dict(epoch=epoch,uids=order))
                for offset in range(0,100,4):
                    batch=order[offset:offset+4];step+=1;t=time.monotonic();opt.zero_grad(set_to_none=True)
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
                    row=dict(update=step,additional_update=step-5000,epoch=epoch,batch_in_epoch=offset//4+1,mesh_participations=4*step,meshes=details,
                             objective=sum(x['weighted_loss'] for x in details),lr={pg['name']:pg['lr'] for pg in opt.param_groups},
                             gradient_norms=grads,global_norm=gn,clip_coefficient=min(1.,1/(gn+1e-6)),actual_updates=movement,
                             head_weight_norms={k:norm([getattr(model.autoencoder,k+'_embedding').weight]) for k in ['edge','face']},seconds=time.monotonic()-t)
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                    write(out/'status.json',dict(state='training',updates=step,epoch=epoch,batch_in_epoch=offset//4+1,mesh_participations=4*step,last_update_seconds=row['seconds']))
                    if step<=5003 or step%25==0:print('UPDATE',step,'EPOCH',epoch,'LOSS',row['objective'],'SECONDS',row['seconds'],flush=True)
                assert set(participation.values())=={epoch}
                if epoch%10==0 or epoch in CHECK_EPOCHS:path=save()
                if epoch in CHECK_EPOCHS:
                    result=evaluate(path)
                    if result['all100_complete_and_perfect'] and not first_perfect:
                        loaded=T.load(path,map_location='cpu',mmap=True);model.load_state_dict(loaded['model'],strict=True)
                        prior=out/f'eval-epoch{epoch:03d}.jsonl';prior.rename(out/f'eval-before-reload-epoch{epoch:03d}.jsonl')
                        again=evaluate(path);assert again['all100_complete_and_perfect'];first_perfect=True
                        write(out/'first100perfect.json',dict(epoch=epoch,update=step,reloaded_checkpoint_verified=True))
        assert step==10000 and set(participation.values())=={400}
        result=dict(state='completed',epochs=400,updates=10000,additional_updates=5000,mesh_participations=40000,per_mesh_participations=participation,
                    final_evaluation=json.loads((out/'eval-summary-epoch400.json').read_text()),stopped_at_budget=True)
        write(out/'complete.json',result);write(out/'status.json',result);print('COMPLETE_10000_ADDITIONAL5000',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(updates=step,epoch=epoch,error=str(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    main()
