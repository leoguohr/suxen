"""B2500 continuation; only new structure is the zero-output terminal FFN."""
import argparse,fcntl,json,os,shutil,sys,time,traceback,subprocess
from pathlib import Path
from runtime import ROOT,BASE,T,np,c,setup,write,sha,norm,tensor_hash,scoring
from helpers import SOURCE,SOURCE_SHA,PARENT,rng,restore_rng,equal,group_steps,edge_observation,evaluate,save_checkpoint
from terminal_ffn import CONTRACT,install_terminal_ffn,gradient_details

CHECKS=[0,25,50,75,100]
LRS=dict(encoder_mu=3e-6,decoder=3e-5,edge_head=3e-5,face_head=3e-5,terminal_ffn=3e-5)

def verify_original_state(model,opt,groups,cp):
    """Match every old Adam slot to its original parameter NAME and group."""
    current=model.state_dict()
    old_state={n:v for n,v in current.items() if not n.startswith('autoencoder.terminal_ffn.')}
    assert equal(old_state,cp['model'])
    saved_groups=cp['optimizer']['param_groups'];names=cp['config']['trainable_groups']
    assert [g['name'] for g in opt.param_groups[:4]]==[g['name'] for g in saved_groups]
    current_serialized=opt.state_dict()
    assert equal(current_serialized['param_groups'][:4],saved_groups)
    records=[]
    for original,live in zip(saved_groups,opt.param_groups[:4]):
        group_name=original['name'];expected_names=names[group_name]
        assert list(groups[group_name])==expected_names
        assert len(expected_names)==len(original['params'])==len(live['params'])
        for name,old_id,parameter in zip(expected_names,original['params'],live['params']):
            assert groups[group_name][name] is parameter
            assert equal(opt.state[parameter],cp['optimizer']['state'][old_id]),name
            records.append(dict(name='autoencoder.'+name,group=group_name,parent_optimizer_id=old_id,
                shape=list(parameter.shape),step=int(opt.state[parameter]['step'])))
    assert len(records)==len(cp['optimizer']['state'])
    if len(opt.param_groups)>4:
        assert opt.param_groups[4]['name']=='terminal_ffn'
        assert all(p not in opt.state or not opt.state[p] for p in opt.param_groups[4]['params'])
    return records

def main(mode):
    assert mode=='terminal_ffn','This branch preserves fully differentiable Soft4'
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'config.json').exists(),'Refuse to overwrite an existing run'
    assert sha(SOURCE)==SOURCE_SHA
    old=json.loads((PARENT/'B_lr03/config.json').read_text())
    assert sha(ROOT/'data/manifest.json')==old['parent']['data_manifest_sha256']
    assert sha(ROOT/'pool_manifest.json')==old['parent']['pool_manifest_sha256']
    for name in ['loader.py','evaluate.py','construction_args.json']:
        assert sha(ROOT/name)==sha(PARENT/name),name
    model,opt,groups,uids,pools,forward,objective,capture,args,batches=setup(mode=='stopgrad')
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
    inheritance=verify_original_state(model,opt,groups,cp)
    old_active=[p for g in groups.values() for p in g.values()]
    # Actual unextended network, before installing any terminal hook.
    rows,loss,parts=objective('teacher_cad50_20')
    baseline_grad=T.autograd.grad(loss,old_active)
    baseline_device_check=dict(uid='teacher_cad50_20',full_mode_loss=float(loss.detach()),
        full_gradient_sha256=tensor_hash((str(i),v) for i,v in enumerate(baseline_grad)),gradient_norm=norm(baseline_grad))
    baseline_grad=[g.detach().cpu() for g in baseline_grad]
    del rows,loss;capture.clear()
    assert equal(rng(),cp['rng'])
    ffn=install_terminal_ffn(model.autoencoder)
    groups['terminal_ffn']={'terminal_ffn.'+n:p for n,p in ffn.named_parameters()}
    template=opt.param_groups[1]
    new_group={k:v for k,v in template.items() if k not in ('params','name','lr')}
    new_group.update(params=list(groups['terminal_ffn'].values()),name='terminal_ffn',lr=LRS['terminal_ffn'])
    opt.add_param_group(new_group)
    # Module construction consumes RNG; do not advance the parent's train stream.
    restore_rng(cp['rng']);model.eval()
    assert equal(rng(),cp['rng'])
    assert all(T.count_nonzero(p)==0 for p in [ffn.out_proj.weight,ffn.out_proj.bias])
    assert {id(p) for ps in groups.values() for p in ps.values()}=={id(p) for p in model.parameters() if p.requires_grad}
    assert verify_original_state(model,opt,groups,cp)==inheritance
    assert group_steps(opt)==dict(encoder_mu=2500,decoder=2500,edge_head=2500,face_head=2500,terminal_ffn=0)
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
    config=dict(branch=ROOT.name,mode=mode,stop_weight_grad=False,source=str(SOURCE),source_sha256=SOURCE_SHA,
        lrs=LRS,lr_schedule='constant; inherited LR; no warmup',optimizer='inherited Adam',betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,
        trainable_groups={k:list(v) for k,v in groups.items()},uids=uids,totals=totals,
        data_manifest_sha256=sha(ROOT/'data/manifest.json'),pool_manifest_sha256=sha(ROOT/'pool_manifest.json'),
        objective='mean50(EdgeSoft4 + FaceSoft4); group mean /4 inside, no extra outer0.25',tau=1,soft4_epsilon=1e-8,
        soft4_reduction='FP32; sum all chunk numerators and masses before group ratios',
        weight_rule='current sigmoid recomputed on every forward; detached only in stopgrad mode; BCE logits differentiable',
        effective_backend='math00 FP32 SDPA MATH; deterministic Graph; TF32/autocast off; same recompute context',
        model_mode='eval with gradients enabled; unchanged dropout',sampling=False,KL=0,logvar_frozen=True,
        microbatch=1,meshes_per_update=50,new_updates=100,initial_completed_updates=2500,final_completed_updates=2600,
        checkpoints=CHECKS,environment=env,scales=model.scoring_contract(),
        step_log_semantics='loss/Edge metrics before update; displacement and Adam steps after update',
        code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},terminal_ffn=CONTRACT,
        terminal_ffn_parameter_count=sum(p.numel() for p in ffn.parameters()),
        terminal_ffn_optimizer='fresh Adam group; LR3e-5; original groups/states preserved',
        rng_rule='restore parent python/numpy/torch/CUDA RNG after FFN initialization and startup checks',
        author_model='gpt-6-astra',reasoning_effort='xhigh')
    write(ROOT/'config.json',config)
    write(ROOT/'restore_verification.json',dict(original_model_exact=True,original_optimizer_exact=True,rng_exact=True,adam_steps=group_steps(opt),
        checkpoint_sha256=SOURCE_SHA,actual_lrs=LRS,original_parameter_groups_exact=True,data_pool_verified=True,
        original_parameter_identity=inheritance,new_group_state_empty=True,terminal_ffn=CONTRACT))
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
        # Zero-step helper checks every saved prediction array against B2500.
        # Compare OLD gradients BEFORE global clipping. The new branch belongs
        # in global clipping, so old post-clip updates need not match control.
        scoring.set_mode(False)
        rows,loss,parts=objective('teacher_cad50_20')
        grad=T.autograd.grad(loss,active)
        device_check=dict(uid='teacher_cad50_20',full_mode_loss=float(loss.detach()),
            full_gradient_sha256=tensor_hash((str(i),v) for i,v in enumerate(grad[:len(old_active)])),gradient_norm=norm(grad[:len(old_active)]))
        assert device_check==baseline_device_check
        assert all(T.equal(a.detach().cpu(),b) for a,b in zip(grad[:len(old_active)],baseline_grad))
        new_grad=dict(zip(groups['terminal_ffn'],grad[len(old_active):]))
        for name,value in new_grad.items():
            assert T.isfinite(value).all(),name
            if not name.startswith('terminal_ffn.out_proj.'):
                assert T.count_nonzero(value)==0,name
        assert T.count_nonzero(new_grad['terminal_ffn.out_proj.weight'])>0
        initial_ffn_grad={n:dict(norm=norm([v]),nonzero=int(T.count_nonzero(v))) for n,v in new_grad.items()}
        del new_grad,grad,baseline_grad,rows,loss;capture.clear();opt.zero_grad(set_to_none=True)
        restore_rng(cp['rng']);model.eval()
        assert equal(rng(),cp['rng']) and verify_original_state(model,opt,groups,cp)==inheritance
        write(ROOT/'startup_gate.json',dict(passed=True,zero_step_all50_arrays_exact=True,
            old_parameter_gradients_before_clip_exact=True,initial_ffn_gradients=initial_ffn_grad,
            old_optimizer_state_by_name_exact=True,new_optimizer_state_empty=True,
            parent_rng_restored=True,optimizer_updates=0))
        write(ROOT/'ready.json',dict(passed=True,source_sha256=SOURCE_SHA,baseline=summary['counts'],device_check=device_check,
            original_model_optimizer_rng_exact=True,zero_step_all50_arrays_exact=True,
            old_gradients_before_clip_exact=True,new_ffn_initial_gradients=initial_ffn_grad,
            terminal_ffn=CONTRACT,optimizer_updates=0))
        print('READY',ROOT.name,device_check,flush=True)
        while not (ROOT.parent/'release.json').exists():
            if (ROOT.parent/'barrier_failure.json').exists():raise RuntimeError('Paired baseline/device check failed')
            time.sleep(2)
        release=json.loads((ROOT.parent/'release.json').read_text())
        assert release['passed'] and release['source_sha256']==SOURCE_SHA and release['new_updates']==100
        with (ROOT/'updates.jsonl').open('x',buffering=1) as log:
            for target in range(1,101):
                started_step=time.monotonic();opt.zero_grad(set_to_none=True);details=[]
                assert scoring.STOP_WEIGHT_GRAD==(mode=='stopgrad')
                assert {g['name']:g['lr'] for g in opt.param_groups}==LRS
                for uid in uids:
                    rows,loss,parts=objective(uid);observation=edge_observation(rows,pools[uid],scales)
                    (loss/50).backward();details.append(dict(uid=uid,**parts,**observation))
                    del rows,loss;capture.clear()
                assert all(p.grad is not None and T.isfinite(p.grad).all() for p in active)
                group_norms={g:norm(p.grad for p in ps.values()) for g,ps in groups.items()}
                ffn_grad=gradient_details(ffn)
                if target==1:
                    assert ffn_grad['out_proj.weight']['nonzero']>0
                    assert all(v['nonzero']==0 for n,v in ffn_grad.items() if not n.startswith('out_proj.'))
                if target==2:
                    assert ffn_grad['in_proj.weight']['nonzero']>0
                    assert ffn_grad['norm.weight']['nonzero']>0
                gn=float(T.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                before={g:[p.detach().clone() for p in ps.values()] for g,ps in groups.items()}
                opt.step();completed=target;updates={}
                assert all(T.isfinite(p).all() for p in active)
                for g,ps in groups.items():
                    delta=norm(p.detach()-v for p,v in zip(ps.values(),before[g]));base=norm(before[g])
                    updates[g]=dict(delta_l2=delta,relative_l2=delta/max(base,1e-30))
                del before
                assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
                steps=group_steps(opt)
                assert all(steps[k]==2500+completed for k in LRS if k!='terminal_ffn')
                assert steps['terminal_ffn']==completed
                losses={k:sum(x[k] for x in details)/50 for k in ['edge','face']}
                row=dict(new_update=completed,completed_updates=2500+completed,state_before=2499+completed,
                    timing='loss and per-UID Edge counts BEFORE update; delta and Adam counters AFTER',meshes=details,
                    losses_before=losses,total_loss_before=losses['edge']+losses['face'],
                    edge_perfect_uids_before=[x['uid'] for x in details if x['edge_perfect']],
                    gradient_norms=group_norms,total_grad_norm=gn,clip_coefficient=min(1.,1/(gn+1e-6)),
                    terminal_ffn_gradients_before_clip=ffn_grad,
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
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['terminal_ffn'],required=True)
    main(p.parse_args().mode)
