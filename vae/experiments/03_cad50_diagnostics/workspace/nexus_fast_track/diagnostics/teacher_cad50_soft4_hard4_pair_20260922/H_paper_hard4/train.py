"""Matched B2500 continuation; sole branch variable is reconstruction loss."""
import argparse,fcntl,json,os,shutil,sys,time,traceback,subprocess
from pathlib import Path
from runtime import ROOT,BASE,T,np,c,setup,write,sha,norm,tensor_hash,scoring
from helpers import SOURCE,SOURCE_SHA,PARENT,rng,restore_rng,equal,group_steps,edge_observation,evaluate,save_checkpoint

CHECKS=[0,25,50,75,100]
LRS=dict(encoder_mu=3e-6,decoder=3e-5,edge_head=3e-5,face_head=3e-5)

def main(mode):
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'config.json').exists(),'Refuse to overwrite an existing run'
    assert sha(SOURCE)==SOURCE_SHA
    old=json.loads((PARENT/'B_lr03/config.json').read_text())
    assert sha(ROOT/'data/manifest.json')==old['parent']['data_manifest_sha256']
    assert sha(ROOT/'pool_manifest.json')==old['parent']['pool_manifest_sha256']
    for name in ['loader.py','evaluate.py','construction_args.json']:
        assert sha(ROOT/name)==sha(PARENT/name),name
    model,opt,groups,uids,pools,forward,objective,capture,args,batches=setup(mode)
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    assert cp['completed_updates']==2500 and cp['config']==old
    assert list(cp['participation'])==uids and set(cp['participation'].values())=={2500}
    assert {k:list(v) for k,v in groups.items()}==old['trainable_groups']
    model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    assert equal(model.state_dict(),cp['model']) and equal(opt.state_dict(),cp['optimizer']) and equal(rng(),cp['rng'])
    assert group_steps(opt)=={k:2500 for k in groups}
    for g in opt.param_groups:
        assert g['lr']==LRS[g['name']] and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
    assert not model.training and not model.autoencoder.training
    assert not T.backends.cuda.matmul.allow_tf32 and not T.backends.cudnn.allow_tf32
    totals=dict(meshes=len(uids),vertices=sum(len(p['vertices']) for p in pools.values()),
        edges=sum(p['edges'].shape[1] for p in pools.values()),faces=sum(len(p['positive']) for p in pools.values()),
        pairs=sum(len(p['vertices'])*(len(p['vertices'])-1)//2 for p in pools.values()),
        face_pool=sum(len(p['positive'])+len(p['mixed']) for p in pools.values()))
    assert totals==dict(meshes=50,vertices=2872,edges=8364,faces=5576,pairs=235741,face_pool=13946)
    parent_eval=json.loads((PARENT/'B_lr03/eval-new0500.json').read_text())
    assert parent_eval['counts']['edge']['fp']==5159 and parent_eval['counts']['edge']['fn']==1656
    assert parent_eval['counts']['face']['fp']==5359 and parent_eval['counts']['face']['fn']==3255
    assert parent_eval['joint_perfect']==32
    write(ROOT/'partition.json',dict(parent_success32=parent_eval['perfect_uids'],parent_failed18=[u for u in uids if u not in parent_eval['perfect_uids']]))
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if raw and raw.endswith('.py') and str(BASE) in raw and Path(raw).is_file():
            p=Path(raw)
            if ROOT in p.parents:continue
            dest=ROOT/'source_archive'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    env=dict(python=sys.version,torch=T.__version__,cuda=T.version.cuda,cudnn=T.backends.cudnn.version(),
        gpu=T.cuda.get_device_name(0),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],logical_device=0,
        nvidia_smi=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True))
    config=dict(branch=ROOT.name,mode=mode,soft_membership_used=mode=='soft4',source=str(SOURCE),source_sha256=SOURCE_SHA,
        lrs=LRS,lr_schedule='constant; inherited LR; no warmup',optimizer='inherited Adam',betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,
        trainable_groups={k:list(v) for k,v in groups.items()},uids=uids,totals=totals,
        data_manifest_sha256=sha(ROOT/'data/manifest.json'),pool_manifest_sha256=sha(ROOT/'pool_manifest.json'),
        objective='mean50(EdgeTask + FaceTask); '+mode+'; each task group mean /4, no extra outer0.25',tau=1,soft4_epsilon=1e-8 if mode=='soft4' else None,
        task_reduction='FP32; sum all chunk numerators and counts/masses before group ratios',
        weight_rule='hard logit.detach()>0 and GT masks; BCE logits differentiable' if mode=='paper_hard4' else 'fully differentiable sigmoid membership, numerator and denominator',
        empty_hard_group_rule='zero contribution; outer divisor always4; explicit experiment convention, not verified official-author code',
        effective_backend='math00 FP32 SDPA MATH; deterministic Graph; TF32/autocast off; same recompute context',
        model_mode='eval with gradients enabled; unchanged dropout',sampling=False,KL=0,logvar_frozen=True,
        microbatch=1,meshes_per_update=50,new_updates=100,initial_completed_updates=2500,final_completed_updates=2600,
        checkpoints=CHECKS,environment=env,scales=model.scoring_contract(),
        step_log_semantics='loss/Edge metrics before update; displacement and Adam steps after update',
        code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')})
    write(ROOT/'config.json',config)
    write(ROOT/'restore_verification.json',dict(model_exact=True,optimizer_exact=True,rng_exact=True,adam_steps=group_steps(opt),
        checkpoint_sha256=SOURCE_SHA,actual_lrs=LRS,parameter_groups_exact=True,data_pool_verified=True))
    frozen=tensor_hash(model.autoencoder.log_variance.named_parameters());scales=model.scoring_contract()
    active=[p for g in groups.values() for p in g.values()];completed=0;best_f1=-1.;best_joint=-1;first=None;started=time.monotonic()
    def checkpoint():return save_checkpoint(ROOT/f'checkpoint-new{completed:04d}-step{2500+completed:04d}.pt',model,opt,config,cp,completed)
    def assess(path):
        nonlocal best_f1,best_joint,first
        e=evaluate(model,uids,pools,objective,capture,batches,ROOT,completed,path,parent_eval)
        for metric,value,previous in [('face_f1',e['counts']['face']['micro_f1'],best_f1),('strict',e['joint_perfect'],best_joint)]:
            if value>previous:
                write(ROOT/f'best_{metric}.json',dict(new_step=completed,completed_updates=2500+completed,value=value,
                    checkpoint=str(path),sha256=e['checkpoint_sha256'],full_model_in_checkpoint=True))
        best_f1=max(best_f1,e['counts']['face']['micro_f1']);best_joint=max(best_joint,e['joint_perfect'])
        if e['joint_perfect']==50 and first is None:
            first=completed;write(ROOT/'first50perfect.json',dict(new_step=completed,checkpoint=str(path),sha256=e['checkpoint_sha256']))
        return e
    try:
        summary=assess(checkpoint())
        # Match the recorded control's full-Soft4 gradient on the assigned GPU.
        scoring.set_mode('soft4')
        rows,loss,parts=objective('teacher_cad50_20')
        grad=T.autograd.grad(loss,active)
        device_check=dict(uid='teacher_cad50_20',full_mode_loss=float(loss.detach()),
            full_gradient_sha256=tensor_hash((str(i),v) for i,v in enumerate(grad)),gradient_norm=norm(grad))
        del grad,rows,loss;capture.clear();scoring.set_mode(mode)
        assert equal(rng(),cp['rng']) and equal(opt.state_dict(),cp['optimizer']) and equal(model.state_dict(),cp['model'])
        write(ROOT/'ready.json',dict(passed=True,source_sha256=SOURCE_SHA,baseline=summary['counts'],device_check=device_check,
            model_optimizer_rng_exact=True,optimizer_updates=0))
        print('READY',ROOT.name,device_check,flush=True)
        reuse=json.loads((ROOT.parent/'repro_outputs/CONTROL_REUSE_AUDIT.json').read_text())
        assert reuse['passed'] and reuse['reuse_control']
        reference=json.loads((Path(reuse['control'])/'ready.json').read_text())['device_check']
        assert device_check==reference, 'Same-weight Soft4 full-network gradient differs on the newly assigned GPU'
        scoring.CALL_COUNTS.update(edge=0,face=0)
        rows,loss,parts=objective('teacher_cad50_20')
        grad=T.autograd.grad(loss,active)
        assert all(T.isfinite(g).all() for g in grad)
        assert scoring.CALL_COUNTS['edge']>0 and scoring.CALL_COUNTS['face']>0
        effective_path=dict(mode=scoring.LOSS_MODE,calls=dict(scoring.CALL_COUNTS),
            scalar=float(loss.detach()),gradient_norm=norm(grad),
            loss_file=str(Path(scoring.__file__).resolve()),no_loss_custom_backward=True,
            note='Actual Edge and Face objective; autograd reaches all active AE/head parameters. Original graph/attention recompute remains unchanged.')
        del grad,rows,loss;capture.clear()
        restore_rng(cp['rng']);model.eval()
        assert equal(rng(),cp['rng']) and equal(opt.state_dict(),cp['optimizer']) and equal(model.state_dict(),cp['model'])
        write(ROOT/'startup_gate.json',dict(passed=True,control_reused=True,
            same_weight_full_soft4_gradient_matches_control=True,effective_path=effective_path,
            model_optimizer_rng_exact_after_checks=True,optimizer_updates=0))
        with (ROOT/'updates.jsonl').open('x',buffering=1) as log:
            for target in range(1,101):
                started_step=time.monotonic();opt.zero_grad(set_to_none=True);details=[]
                assert scoring.LOSS_MODE==mode
                assert {g['name']:g['lr'] for g in opt.param_groups}==LRS
                for uid in uids:
                    rows,loss,parts=objective(uid);observation=edge_observation(rows,pools[uid],scales)
                    (loss/50).backward();details.append(dict(uid=uid,**parts,**observation))
                    del rows,loss;capture.clear()
                assert all(p.grad is not None and T.isfinite(p.grad).all() for p in active)
                group_norms={g:norm(p.grad for p in ps.values()) for g,ps in groups.items()}
                gn=float(T.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                before={g:[p.detach().clone() for p in ps.values()] for g,ps in groups.items()}
                opt.step();completed=target;updates={}
                assert all(T.isfinite(p).all() for p in active)
                for g,ps in groups.items():
                    delta=norm(p.detach()-v for p,v in zip(ps.values(),before[g]));base=norm(before[g])
                    updates[g]=dict(delta_l2=delta,relative_l2=delta/max(base,1e-30))
                del before
                assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
                steps=group_steps(opt);assert set(steps.values())=={2500+completed}
                losses={k:sum(x[k] for x in details)/50 for k in ['edge','face']}
                row=dict(new_update=completed,completed_updates=2500+completed,state_before=2499+completed,
                    timing='loss and per-UID Edge counts BEFORE update; delta and Adam counters AFTER',meshes=details,
                    losses_before=losses,total_loss_before=losses['edge']+losses['face'],
                    edge_perfect_uids_before=[x['uid'] for x in details if x['edge_perfect']],
                    gradient_norms=group_norms,total_grad_norm=gn,clip_coefficient=min(1.,1/(gn+1e-6)),
                    lrs={g['name']:g['lr'] for g in opt.param_groups},actual_updates=updates,adam_steps=steps,
                    participation_per_mesh=2500+completed,seconds=time.monotonic()-started_step)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                write(ROOT/'status.json',dict(state='training',new_updates=completed,budget=100,total_loss_before=row['total_loss_before'],last_update_seconds=row['seconds']))
                if completed<=3 or completed%10==0:print('UPDATE',ROOT.name,completed,row['total_loss_before'],row['seconds'],flush=True)
                if completed in CHECKS:
                    assert tensor_hash(model.autoencoder.log_variance.named_parameters())==frozen
                    summary=assess(checkpoint())
        full=ROOT/'model-step2600-inference.pt'
        T.save(dict(model=model.state_dict(),args=args,config=config,completed_updates=2600),full)
        done=dict(state='complete',branch=ROOT.name,new_updates=completed,completed_updates=2600,stopped_at_budget=completed==100,
            summary={k:v for k,v in summary.items() if k!='meshes'},first50perfect_new_step=first,
            full_model=str(full),full_model_sha256=sha(full),seconds=time.monotonic()-started,
            frozen_logvar_unchanged=tensor_hash(model.autoencoder.log_variance.named_parameters())==frozen)
        write(ROOT/'complete.json',done);write(ROOT/'status.json',done);print('BRANCH_COMPLETE',ROOT.name,flush=True)
    except BaseException as exc:
        write(ROOT/'failure.json',dict(completed_new_updates=completed,error=str(exc),traceback=traceback.format_exc()))
        save_checkpoint(ROOT/'failure-state.pt',model,opt,config,cp,completed)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['soft4','paper_hard4'],required=True)
    main(p.parse_args().mode)
