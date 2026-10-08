"""One independently seeded architecture branch; exactly five meshes per update."""
import argparse
import fcntl
import time
import traceback
from run_support import *
from native_models import NativeTopologyAE, Config, Graph, VARIANTS
from data_objective import load_dataset, epoch_batches, negative_faces, objective


def main(variant):
    root=ROOT/variant;root.mkdir(exist_ok=True)
    lock=(root/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'config.json').exists(),'Refuse duplicate or implicit resume'
    account=ResourceAccount('train-'+variant)
    completed=0;model=opt=None;manifest=None;started=time.monotonic()
    try:
        configure();torch.cuda.set_device(0)
        initialization=read(OUT/'INITIALIZATION.json')
        entry=initialization['files'][variant]
        assert sha(entry['path'])==entry['sha256']
        initial=torch.load(entry['path'],map_location='cpu',mmap=True,weights_only=False)
        model=NativeTopologyAE(Config(**initial['model_config'])).cuda().float()
        model.load_state_dict(initial['model'],strict=True);model.train()
        active=[p for p in model.parameters() if p.requires_grad]
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        opt=torch.optim.AdamW(active,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=True)
        assert not opt.state
        cpu_items,manifest=load_dataset(ROOT/'data',ROOT/'pools');uids=manifest['uids']
        items={u:move_item(x,'cuda') for u,x in cpu_items.items()}
        graphs={u:Graph.from_faces(x['faces'],len(x['vertices'])) for u,x in items.items()}
        frozen=tensor_hash(model.log_variance.state_dict())
        configure(0)
        config=dict(model_variant=variant,model_config=model.cfg.to_dict(),seed=0,initialization=entry,
            teacher_weights_used=False,old_checkpoint_used=False,optimizer='fresh AdamW',lr=1e-4,
            betas=[.9,.999],eps=1e-8,weight_decay=.01,clip=1,warmup_updates=100,
            warmup_formula='lr=1e-4*min(update/100,1), update starts at1',microbatch=1,meshes_per_update=5,
            max_updates=20000,max_mesh_participations=100000,checkpoints=CHECKS,
            data=manifest,negative_sampler='sha256 domain/seed/epoch/UID -> NumPy PCG64 uniform unique nonGT triples',
            negative_ratio=1.5,objective='(Edge Hard4+Face Hard4)/5 per mesh; global full-mesh group sums; empty0; /4',
            sampling=False,KL=0,logvar_frozen=True,dropout=0,threshold='logit>0',
            scales=dict(edge=0.9306077080970389,face=0.39804385828730726,face_area_factor=.25),
            numerical='FP32 SDPA MATH; deterministic Graph forward/backward; TF32/autocast/fastpath off',
            trainable_parameters=sum(p.numel() for p in active),optimizer_parameter_names=names,
            code_sha256=code_hashes(),torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'])
        write(root/'config.json',config)
        # Full-size real-mesh backward and train/eval path gate. Zero steps.
        uid=max(uids,key=lambda u:len(items[u]['vertices']))
        item=items[uid];neg,negative_sha=negative_faces(cpu_items[uid],0)
        torch.cuda.reset_peak_memory_stats();pre=time.monotonic()
        rows=model(item['vertices'],item['faces'],sample_latent=False,graph=graphs[uid])
        assert rows['latent'] is rows['mu']
        loss,parts=objective(rows,item,neg)
        loss.backward();assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
        block_grad={}
        for name,p in model.named_parameters():
            if p.requires_grad and any(k in name for k in ['.ffn.','.graph.','vertex_input','face_input']):
                block_grad[name]=float(p.grad.norm())
        detached={k:v.detach().clone() for k,v in rows.items()};del rows,loss
        opt.zero_grad(set_to_none=True);model.eval()
        with torch.no_grad():again=model(item['vertices'],item['faces'],sample_latent=False,graph=graphs[uid])
        exact=all(torch.equal(detached[k],again[k]) for k in detached)
        assert exact and not opt.state
        gate=dict(passed=True,optimizer_updates=0,uid=uid,vertices=len(item['vertices']),
            train_eval_mu_outputs_bitwise_equal=exact,all_trainable_gradients_finite=True,
            module_gradient_norms=block_grad,loss_parts=parts,negative_sha256=negative_sha,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-pre,
            trainable_parameters=config['trainable_parameters'],initial_model_sha256=entry['sha256'])
        del detached,again;model.train();configure(0)
        write(root/'preflight.json',gate)
        participation={u:0 for u in uids}
        def save(reason):
            assert tensor_hash(model.log_variance.state_dict())==frozen
            p=root/f'checkpoint-{completed:05d}.pt'
            value=dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),config=config,
                model_variant=variant,model_config=model.cfg.to_dict(),completed_updates=completed,
                participation=participation,completed_epochs=completed//10,
                data_generator=dict(seed=0,next_epoch=completed//10,next_batch=completed%10,algorithm='sha256 order/PCG64'),
                negative_generator=dict(seed=0,algorithm='sha256 negative/epoch/UID/PCG64',state='stateless; next epoch/batch above'),reason=reason)
            entry=save_torch(p,value)
            write(root/f'checkpoint-{completed:05d}.json',entry)
            request=dict(variant=variant,step=completed,checkpoint=entry,kind='evaluation')
            write(ROOT/'queue'/f'{completed:05d}-{variant}.json',request)
            return entry
        initial_saved=save('initial_random')
        write(root/'ready.json',dict(passed=True,optimizer_updates=0,initial_checkpoint=initial_saved))
        while not (ROOT/'release.json').exists() or not (root/'evaluations/eval-00000.json').exists():
            if (ROOT/'ABORT_STARTUP.json').exists():raise RuntimeError('Startup aborted; no optimizer updates')
            if (root/'evaluation_failure.json').exists():raise RuntimeError('Initial evaluation failed')
            assert account.update()<TRAIN_GPU_SECONDS
            time.sleep(2)
        assert read(ROOT/'release.json')['passed'] and read(root/'evaluations/eval-00000.json')['complete']
        stop_reason='max_updates';last_checkpoint=initial_saved
        with (root/'updates.jsonl').open('x',buffering=1) as log:
            for step in range(1,20001):
                if (root/'TARGET_REACHED.json').exists():stop_reason='target_checkpoint_reached';break
                if (root/'evaluation_failure.json').exists():raise RuntimeError('Full evaluation failed')
                if account.update()>=TRAIN_GPU_SECONDS:stop_reason='resource_budget';break
                epoch,batch_index=divmod(step-1,10);batch=epoch_batches(uids,epoch)[batch_index]
                lr=1e-4*min(step/100,1)
                for group in opt.param_groups:group['lr']=lr
                opt.zero_grad(set_to_none=True);details=[];tic=time.monotonic()
                for uid in batch:
                    item=items[uid];neg,neg_sha=negative_faces(cpu_items[uid],epoch)
                    rows=model(item['vertices'],item['faces'],sample_latent=False,graph=graphs[uid])
                    assert rows['latent'] is rows['mu']
                    loss,parts=objective(rows,item,neg)
                    assert torch.isfinite(loss),'Nonfinite reconstruction loss'
                    (loss/5).backward()
                    details.append(dict(uid=uid,negative_sha256=neg_sha,**parts))
                    del rows,loss
                gn=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True,foreach=True))
                opt.step();completed=step
                for uid in batch:participation[uid]+=1
                assert all(p.grad is None for p in model.log_variance.parameters())
                assert int(opt.state[active[0]]['step'])==step
                record=dict(update=step,epoch=epoch,batch_index=batch_index,uids=batch,meshes=details,
                    loss_before=sum(x['edge_loss']+x['face_loss'] for x in details)/5,
                    gradient_norm_before_clip=gn,clip_coefficient=min(1.,1/(gn+1e-6)),lr=lr,
                    adam_step=step,participations=sum(participation.values()),seconds=time.monotonic()-tic,
                    timing='loss/Edge counts before update; Adam step/participation after update')
                log.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                write(root/'status.json',dict(state='training',completed_updates=step,budget=20000,
                    loss_before=record['loss_before'],gradient_norm=gn,seconds_per_update=record['seconds']))
                if step<=3 or step%100==0:print('UPDATE',variant,step,record['loss_before'],record['seconds'],flush=True)
                if step in CHECKS:
                    assert set(participation.values())=={step//10}
                    last_checkpoint=save('scheduled')
        if completed not in CHECKS:last_checkpoint=save(stop_reason)
        write(root/'training_complete.json',dict(state='training_complete',completed_updates=completed,
            mesh_participations=sum(participation.values()),stop_reason=stop_reason,final_checkpoint=last_checkpoint,
            maximum_updates_respected=completed<=20000,seconds=time.monotonic()-started))
        print('TRAINING_STOPPED',variant,completed,stop_reason,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(completed_updates=completed,error=str(error),traceback=traceback.format_exc()))
        if model is not None and opt is not None:
            save_torch(root/'failure-state.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),completed_updates=completed))
        raise
    finally:account.update('finish')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True)
    main(p.parse_args().variant)
