"""Explicit full-state continuation, using the unchanged original update body."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import math
import traceback
from run_support import *
from native_models import NativeTopologyAE, Config, Graph
from data_objective import load_dataset, epoch_batches, negative_faces, objective


def recorded_first_step(plan,variant,step):
    old=Path(plan['original_root'])/variant/'updates.jsonl'
    if step>plan['branches'][variant]['original_recorded_updates']:return None
    with old.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['update']==step:return row
    raise AssertionError('Missing previously recorded replay step')


def check_replay(old,new):
    for key in ['update','epoch','batch_index','uids','lr','adam_step','participations']:
        assert old[key]==new[key],key
    for key in ['loss_before','gradient_norm_before_clip','clip_coefficient']:
        assert math.isclose(old[key],new[key],rel_tol=1e-5,abs_tol=1e-6),(key,old[key],new[key])
    for a,b in zip(old['meshes'],new['meshes']):
        for key in ['uid','negative_sha256','face_positives','face_negatives']:assert a[key]==b[key]
        for task in ['edge','face']:
            assert a[task]['counts']==b[task]['counts']
            for x,y in zip(a[task]['group_bce_sums'],b[task]['group_bce_sums']):
                assert math.isclose(x,y,rel_tol=1e-5,abs_tol=1e-6),(task,x,y)


def main(variant,end):
    branch=ROOT/variant;plan=read(OUT/'RECOVERY_PLAN.json');current=read(branch/'recovery_current.json')
    start=current['step'];completed=start;model=opt=None
    assert start<end<=plan['common_logical_stop']
    record_dir=branch/'segments'/f'{start+1:05d}-{end:05d}';record_dir.mkdir(parents=True,exist_ok=False)
    locks=[]
    for path in [branch/'execution.lock',Path(plan['original_root'])/variant/'execution.lock']:
        lock=path.open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    account=ResourceAccount(f'resume-{variant}-{start+1:05d}-{end:05d}')
    try:
        configure();torch.cuda.set_device(0)
        receipt=current['checkpoint'];assert sha(receipt['path'])==receipt['sha256']
        cp=torch.load(receipt['path'],map_location='cpu',mmap=True,weights_only=False)
        assert cp['completed_updates']==start and cp['model_variant']==variant
        config=cp['config'];assert config['code_sha256']==code_hashes()
        model=NativeTopologyAE(Config(**cp['model_config'])).cuda().float()
        model.load_state_dict(cp['model'],strict=True);model.train()
        active=[p for p in model.parameters() if p.requires_grad]
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        assert names==config['optimizer_parameter_names']
        opt=torch.optim.AdamW(active,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=True)
        opt.load_state_dict(cp['optimizer'])
        assert tensor_hash(model.state_dict())==tensor_hash(cp['model'])
        loaded=opt.state_dict()
        assert loaded['param_groups']==cp['optimizer']['param_groups']
        for key,slot in loaded['state'].items():
            for name,value in slot.items():
                assert torch.equal(value.cpu(),cp['optimizer']['state'][key][name].cpu())
            assert int(slot['step'])==start
        cpu_items,manifest=load_dataset(ROOT/'data',ROOT/'pools');assert manifest==config['data']
        uids=manifest['uids'];items={u:move_item(x,'cuda') for u,x in cpu_items.items()}
        graphs={u:Graph.from_faces(x['faces'],len(x['vertices'])) for u,x in items.items()}
        frozen=tensor_hash(model.log_variance.state_dict());participation=cp['participation'].copy()
        assert cp['data_generator']['next_epoch']==start//10 and cp['data_generator']['next_batch']==start%10
        expected=recorded_first_step(plan,variant,start+1)
        restore_rng(cp['rng']);restored=rng_state()
        assert restored['python']==cp['rng']['python']
        assert restored['numpy'][0]==cp['rng']['numpy'][0] and np.array_equal(restored['numpy'][1],cp['rng']['numpy'][1])
        assert restored['numpy'][2:]==cp['rng']['numpy'][2:]
        assert torch.equal(restored['torch'],cp['rng']['torch'])
        assert all(torch.equal(x,y) for x,y in zip(restored['cuda'],cp['rng']['cuda']))
        write(record_dir/'RESTORE_AUDIT.json',dict(passed=True,parent=receipt,step=start,
            model_bitwise_equal=True,all_adam_slots_bitwise_equal=True,rng_bitwise_equal=True,
            optimizer_parameter_mapping_equal=True,original_runtime_code_unchanged=True,optimizer_updates=0))
        del cp,loaded
        stop_reason='segment_complete';started=time.monotonic()
        with (branch/'updates.jsonl').open('a',buffering=1) as log:
            for step in range(start+1,end+1):
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
                proposed=dict(update=step,epoch=epoch,batch_index=batch_index,uids=batch,meshes=details,
                    loss_before=sum(x['edge_loss']+x['face_loss'] for x in details)/5,
                    gradient_norm_before_clip=gn,clip_coefficient=min(1.,1/(gn+1e-6)),lr=lr,
                    adam_step=step,participations=sum(participation.values())+5)
                if step==start+1 and expected is not None:
                    check_replay(expected,proposed)
                    write(record_dir/'FIRST_REPLAY_AUDIT.json',dict(passed=True,step=step,
                        exact_uid_negatives_edge_face_counts=True,loss_and_gradient_rtol=1e-5,atol=1e-6,
                        before_optimizer_step=True,original=expected,recomputed=proposed))
                opt.step();completed=step
                for uid in batch:participation[uid]+=1
                assert all(p.grad is None for p in model.log_variance.parameters())
                assert int(opt.state[active[0]]['step'])==step
                record=dict(**proposed,seconds=time.monotonic()-tic,
                    timing='loss/Edge counts before update; Adam step/participation after update')
                log.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                physical=plan['branches'][variant]['conservative_original_updates']+completed-plan['branches'][variant]['parent_step']
                assert physical<=20000
                write(branch/'status.json',dict(state='training',completed_updates=step,
                    budget=plan['common_logical_stop'],conservative_physical_updates=physical,
                    loss_before=record['loss_before'],gradient_norm=gn,seconds_per_update=record['seconds']))
                if step==start+1 or step%100==0:print('UPDATE',variant,step,record['loss_before'],record['seconds'],flush=True)
        assert all(int(opt.state[p]['step'])==completed for p in active)
        assert tensor_hash(model.log_variance.state_dict())==frozen
        if completed>start:
            path=branch/f'checkpoint-{completed:05d}.pt';assert not path.exists()
            value=dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),config=config,
                model_variant=variant,model_config=model.cfg.to_dict(),completed_updates=completed,
                participation=participation,completed_epochs=completed//10,
                data_generator=dict(seed=0,next_epoch=completed//10,next_batch=completed%10,algorithm='sha256 order/PCG64'),
                negative_generator=dict(seed=0,algorithm='sha256 negative/epoch/UID/PCG64',state='stateless; next epoch/batch above'),
                reason=stop_reason,recovery=dict(parent=receipt,plan=str(OUT/'RECOVERY_PLAN.json')))
            receipt=save_torch(path,value);write(branch/f'checkpoint-{completed:05d}.json',receipt)
            write(branch/'recovery_current.json',dict(step=completed,checkpoint=receipt))
            write(ROOT/'queue'/f'{completed:05d}-{variant}.json',dict(variant=variant,step=completed,checkpoint=receipt,kind='evaluation'))
        write(record_dir/'complete.json',dict(start=start,end=completed,optimizer_updates=completed-start,
            requested_end=end,stop_reason=stop_reason,checkpoint=receipt,seconds=time.monotonic()-started))
        print('SEGMENT_COMPLETE',variant,completed,stop_reason,flush=True)
    except BaseException as error:
        write(branch/'failure.json',dict(completed_updates=completed,error=str(error),traceback=traceback.format_exc(),segment=str(record_dir)))
        if model is not None and opt is not None:
            save_torch(record_dir/'failure-state.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(),completed_updates=completed))
        raise
    finally:account.update('finish')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--variant',required=True);parser.add_argument('--end',type=int,required=True)
    args=parser.parse_args();main(args.variant,args.end)
