"""CAD50 paired continuation: same step2000 model/Adam/RNG, LR is the only intervention."""
import argparse, fcntl, json, random, shutil, sys, time, traceback
from pathlib import Path
from runtime import ROOT, BASE, T, np, c, setup, write, sha, norm, tensor_hash
from evaluate import evaluate_mesh

PARENT=BASE/'diagnostics/teacher_cad50_fresh512_20260921'
SOURCE=PARENT/'run/checkpoint-step2000.pt'
SOURCE_SHA='ae7c2835fe7ad9da72edd413f66373b415721b2cfac72df8c6b3fc141b987358'
CHECKS=list(range(0,501,50))
SELECTED=[f'teacher_cad50_{i:02d}' for i in [0,2,3,13,20,24,34,39]]
LRS={'A_control_lr1':dict(encoder_mu=1e-5,decoder=1e-4,edge_head=1e-4,face_head=1e-4),
     'B_lr03':dict(encoder_mu=3e-6,decoder=3e-5,edge_head=3e-5,face_head=3e-5)}

def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=T.get_rng_state(),cuda=T.cuda.get_rng_state_all())

def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);T.set_rng_state(x['torch']);T.cuda.set_rng_state_all(x['cuda'])

def equal(a,b):
    if T.is_tensor(a):return T.equal(a.detach().cpu(),b.detach().cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def group_steps(opt):
    ans={}
    for g in opt.param_groups:
        values={int(opt.state[p]['step']) for p in g['params']}
        assert len(values)==1
        ans[g['name']]=values.pop()
    return ans

@T.no_grad()
def edge_observation(rows,pool,scales,arrays=False):
    n=len(pool['vertices']);keys=T.as_tensor(np.unique(pool['edges'].T@np.array([n,1])),device='cuda')
    ids=T.triu_indices(n,n,offset=1,device='cuda').T
    logits=c.edge_logits(rows[2][0].detach(),ids,scales)
    assert T.isfinite(logits).all()
    y=T.isin(ids[:,0]*n+ids[:,1],keys);positive=logits>0
    tp=int((positive&y).sum());fp=int((positive&~y).sum());fn=int((~positive&y).sum())
    result=dict(tp=tp,fp=fp,fn=fn,edge_perfect=fp==fn==0)
    if arrays:return result,ids.cpu().numpy().astype(np.int32),logits.cpu().numpy(),y.cpu().numpy()
    return result

def keep_encoder_output(model,captured):
    def hook(module,inputs,output):
        captured['encoder_vertex_hidden']=output.detach().clone()
        return None
    return model.autoencoder.encoder_output_norm.register_forward_hook(hook)

def restore_parent():
    assert sha(SOURCE)==SOURCE_SHA
    parent_cfg=json.loads((PARENT/'config.json').read_text())
    for filename,digest in parent_cfg['entry_sha256'].items():assert sha(PARENT/filename)==digest,filename
    for filename,digest in parent_cfg['effective_source_sha256'].items():assert sha(filename)==digest,filename
    assert sha(ROOT/'data/manifest.json')==parent_cfg['data_manifest_sha256']
    assert sha(ROOT/'pool_manifest.json')==parent_cfg['pool_manifest_sha256']
    for filename in ['runtime.py','loader.py','evaluate.py','effective_loss_and_scoring.py','construction_args.json']:
        assert sha(ROOT/filename)==sha(PARENT/filename),filename
    model,opt,groups,uids,pools,forward,objective,capture,args,batches=setup()
    cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
    assert cp['config']==parent_cfg and cp['completed_updates']==2000
    assert list(cp['participation'])==uids and set(cp['participation'].values())=={2000}
    assert {k:list(v) for k,v in groups.items()}==parent_cfg['trainable_groups']
    model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
    assert equal(model.state_dict(),cp['model']) and equal(opt.state_dict(),cp['optimizer']) and equal(rng(),cp['rng'])
    assert group_steps(opt)=={k:2000 for k in groups}
    for g in opt.param_groups:
        assert g['lr']==LRS['A_control_lr1'][g['name']]
        assert tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
    assert not model.training and not model.autoencoder.training
    assert not T.backends.cuda.matmul.allow_tf32 and not T.backends.cudnn.allow_tf32
    totals=dict(meshes=len(uids),vertices=sum(len(p['vertices']) for p in pools.values()),
        edges=sum(p['edges'].shape[1] for p in pools.values()),faces=sum(len(p['positive']) for p in pools.values()),
        pairs=sum(len(p['vertices'])*(len(p['vertices'])-1)//2 for p in pools.values()),
        face_pool=sum(len(p['positive'])+len(p['mixed']) for p in pools.values()))
    assert totals==dict(meshes=50,vertices=2872,edges=8364,faces=5576,pairs=235741,face_pool=13946),totals
    parent_eval=json.loads((PARENT/'run/eval-step2000.json').read_text())
    success=parent_eval['perfect_uids'];failed=[u for u in uids if u not in success]
    assert len(success)==30 and len(failed)==20
    if (ROOT/'partition.json').exists():assert json.loads((ROOT/'partition.json').read_text())==dict(success30=success,failed20=failed)
    else:write(ROOT/'partition.json',dict(success30=success,failed20=failed))
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if raw and raw.endswith('.py') and str(BASE) in raw and Path(raw).is_file():
            p=Path(raw)
            if ROOT in p.parents:continue
            dest=ROOT/'source_archive'/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    verification=dict(checkpoint=str(SOURCE),sha256=SOURCE_SHA,model_exact=True,optimizer_exact=True,rng_exact=True,
        completed_updates=2000,adam_steps=group_steps(opt),totals=totals,parameter_groups={k:list(v) for k,v in groups.items()},
        parent_config_sha256=sha(PARENT/'config.json'),data_manifest_sha256=parent_cfg['data_manifest_sha256'],
        pool_manifest_sha256=parent_cfg['pool_manifest_sha256'],scales=model.scoring_contract(),model_mode='eval with gradients enabled',
        unchanged_runtime_files={n:sha(ROOT/n) for n in ['runtime.py','loader.py','evaluate.py','effective_loss_and_scoring.py']})
    return model,opt,groups,uids,pools,objective,capture,args,batches,cp,parent_eval,verification

def summarize(rows,parent_success):
    totals={kind:{k:sum(x[kind][k] for x in rows) for k in ['tp','fp','fn','tn']} for kind in ['edge','face']}
    for d in totals.values():d['micro_f1']=2*d['tp']/max(2*d['tp']+d['fp']+d['fn'],1)
    edge=[x['uid'] for x in rows if x['edge_perfect']];face=[x['uid'] for x in rows if x['face_perfect']]
    joint=[x['uid'] for x in rows if x['joint_perfect']]
    return dict(counts=totals,edge_perfect=len(edge),face_perfect=len(face),joint_perfect=len(joint),
        edge_perfect_uids=edge,face_perfect_uids=face,perfect_uids=joint,
        retained=sorted(set(joint)&set(parent_success)),lost=sorted(set(parent_success)-set(joint)),new=sorted(set(joint)-set(parent_success)),
        losses={k:sum(x['parts'][k] for x in rows)/len(rows) for k in ['edge','face']},
        face_fn_missing=sum(x['missing_gt_face_candidates'] for x in rows),
        face_fn_present=sum(x['face']['fn_present_but_negative'] for x in rows),
        face_fp_inside_pool=sum(x['face']['actual_fp_inside_training_pool'] for x in rows),
        face_fp_outside_pool=sum(x['face']['actual_fp_outside_training_pool'] for x in rows))

def evaluate(model,uids,pools,objective,capture,batches,out,new_step,checkpoint,parent_eval,representations=False):
    params_before=tensor_hash(model.named_parameters());rng_before=rng();scales=model.scoring_contract()
    dest=out/f'predictions-new{new_step:04d}';dest.mkdir(exist_ok=False)
    repdir=out/f'representations-new{new_step:04d}'
    if representations:repdir.mkdir(exist_ok=False)
    records=[];rep_index=[];started=time.monotonic()
    for uid in uids:
        captured={};hook=None
        if representations and uid in SELECTED:hook=keep_encoder_output(model,captured)
        rows,loss,parts=objective(uid)
        if hook:hook.remove()
        edge_check=edge_observation(rows,pools[uid],scales)
        detached=tuple(tuple(x.detach() for x in group) for group in rows)
        if captured:
            batch=batches[uid];n=len(pools[uid]['vertices'])
            _,pair,edge_logits,labels=edge_observation(rows,pools[uid],scales,True)
            gt_faces=np.sort(pools[uid]['positive'],axis=1).astype(np.int32)
            with T.no_grad():fl=c.face_logits(detached[3][0],T.as_tensor(gt_faces,device='cuda'),scales).cpu().numpy()
            tensors=dict(vertices=batch.vertices[0,:n].detach().cpu().numpy(),vertex_mask=batch.vertex_mask[0].cpu().numpy(),
                local_vertex_indices=np.arange(n,dtype=np.int32),encoder_vertex_hidden=captured['encoder_vertex_hidden'].cpu().numpy(),
                mu=detached[0][0].cpu().numpy(),decoder_hidden=capture['hidden'].detach().cpu().numpy(),
                edge_embedding=detached[2][0].cpu().numpy(),face_embedding=detached[3][0].cpu().numpy(),
                edge_pair_ids=pair,edge_pair_logits=edge_logits,edge_pair_gt=labels,gt_face_ids=gt_faces,gt_face_logits=fl)
            assert np.array_equal(tensors['vertices'],pools[uid]['vertices'])
            p=repdir/f'{uid}.npz';np.savez_compressed(p,**tensors)
            rep_index.append(dict(uid=uid,path=str(p.relative_to(ROOT)),sha256=sha(p),shapes={k:list(v.shape) for k,v in tensors.items()},
                positions=dict(encoder_vertex_hidden='encoder_output_norm output: vertex nodes only, before mu projection',
                    mu='mu head output; exactly decoder latent input',decoder_hidden='decoder_output_norm output, after all16 blocks',
                    edge_embedding='actual per-mesh centered scoring representation',face_embedding='actual per-mesh centered scoring representation'),
                coordinates='unchanged source FP32; original cached local node numbering',mask='microbatch1, all real vertices true'))
        del rows,loss;capture.clear();captured.clear()
        p=dest/f'{uid}.npz';m=evaluate_mesh(detached,pools[uid],scales,p);del detached
        assert m['face']['complete']
        assert all(edge_check[k]==m['edge'][k] for k in ['tp','fp','fn'])
        m['face']['actual_fp_inside_training_pool']=m['face']['fp']-m['face']['actual_fp_outside_training_pool']
        m['face']['fn_missing_candidate']=m['missing_gt_face_candidates']
        m['face']['fn_present_but_negative']=m['face']['fn']-m['missing_gt_face_candidates']
        records.append(dict(uid=uid,vertices=len(pools[uid]['vertices']),parts=parts,
            prediction_path=str(p.relative_to(ROOT)),prediction_sha256=sha(p),**m))
    summary=summarize(records,parent_eval['perfect_uids'])
    summary.update(new_step=new_step,completed_updates=2000+new_step,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        timing='frozen state after completed_updates optimizer updates',seconds=time.monotonic()-started,meshes=records)
    summary['size_groups']={}
    for name,lo,hi in [('8_vertices',8,8),('12_to_16',12,16),('66_to_274',66,274)]:
        subset=[x for x in records if lo<=x['vertices']<=hi]
        subset_uids={x['uid'] for x in subset}
        subset_parent=[u for u in parent_eval['perfect_uids'] if u in subset_uids]
        summary['size_groups'][name]=dict(meshes=len(subset),**summarize(subset,subset_parent))
    assert sum(x['meshes'] for x in summary['size_groups'].values())==50
    if new_step==0:
        assert summary['counts']==parent_eval['counts']
        assert summary['perfect_uids']==parent_eval['perfect_uids']
        assert summary['edge_perfect']==summary['face_perfect']==summary['joint_perfect']==30
        for actual,reference in zip(records,parent_eval['meshes']):
            assert actual['uid']==reference['uid'] and actual['parts']==reference['parts']
            for kind in ['edge','face']:
                for k in ['tp','fp','fn','tn']:assert actual[kind][k]==reference[kind][k]
            with np.load(PARENT/reference['prediction_path']) as old,np.load(ROOT/actual['prediction_path']) as now:
                assert old.files==now.files and all(np.array_equal(old[k],now[k]) for k in old.files)
    assert tensor_hash(model.named_parameters())==params_before and equal(rng_before,rng())
    assert all(p.grad is None for p in model.autoencoder.log_variance.parameters())
    write(out/f'eval-new{new_step:04d}.json',summary)
    if representations:write(repdir/'manifest.json',rep_index)
    print('EVAL',out.name,new_step,summary['joint_perfect'],summary['counts'],flush=True)
    return summary

def check_hooks(model,groups,objective,capture):
    active=[p for g in groups.values() for p in g.values()];state=rng();results=[]
    for uid in SELECTED:
        rows,loss,parts=objective(uid);base=tuple(tuple(x.detach().clone() for x in row) for row in rows)
        reference=T.autograd.grad(loss,active);grad_hash=tensor_hash((str(i),g) for i,g in enumerate(reference))
        del reference,rows,loss;capture.clear()
        captured={};hook=keep_encoder_output(model,captured)
        rows,loss,other=objective(uid);hook.remove()
        assert other==parts and all(T.equal(a,b) for row,ref in zip(rows,base) for a,b in zip(row,ref))
        current=T.autograd.grad(loss,active)
        assert tensor_hash((str(i),g) for i,g in enumerate(current))==grad_hash
        assert equal(state,rng())
        results.append(dict(uid=uid,outputs_exact=True,full_gradient_exact=True,rng_unchanged=True))
        del current,rows,loss,base;capture.clear();captured.clear()
    return results

def save_checkpoint(path,model,opt,config,cp,new_step,updates_applied=None):
    tmp=path.with_suffix('.tmp')
    T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),config=config,
        completed_updates=2000+new_step,new_updates=new_step,participation={u:2000+new_step for u in cp['participation']},
        parent_checkpoint=str(SOURCE),parent_sha256=SOURCE_SHA,updates_applied=updates_applied),tmp)
    tmp.replace(path)
    return path

def main(branch,precheck):
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,opt,groups,uids,pools,objective,capture,args,batches,cp,parent_eval,verification=restore_parent()
    frozen=tensor_hash(model.autoencoder.log_variance.named_parameters());scales=model.scoring_contract()
    if precheck:
        out=ROOT/'common';out.mkdir(exist_ok=False)
        summary=evaluate(model,uids,pools,objective,capture,batches,out,0,SOURCE,parent_eval,True)
        hook_check=check_hooks(model,groups,objective,capture)
        assert equal(model.state_dict(),cp['model']) and equal(opt.state_dict(),cp['optimizer']) and equal(rng(),cp['rng'])
        verification.update(passed=True,optimizer_updates=0,baseline_joint_perfect=summary['joint_perfect'],hooks=hook_check)
        write(ROOT/'precheck.json',verification);print('PRECHECK_PASSED',flush=True);return
    assert json.loads((ROOT/'precheck.json').read_text())['passed']
    assert branch in LRS
    out=ROOT/branch;out.mkdir(exist_ok=False)
    for g in opt.param_groups:g['lr']=LRS[branch][g['name']]
    for a,z in zip(opt.state_dict()['param_groups'],cp['optimizer']['param_groups']):
        assert {k:v for k,v in a.items() if k!='lr'}=={k:v for k,v in z.items() if k!='lr'}
    assert equal(opt.state_dict()['state'],cp['optimizer']['state']) and equal(rng(),cp['rng'])
    config=dict(branch=branch,lrs=LRS[branch],lr_schedule='constant; no warmup or per-step overwrite',
        parent=verification,uids=uids,trainable_groups={g:list(ps) for g,ps in groups.items()},
        objective='mean50(fully-differentiable EdgeSoft4 + FaceSoft4); no extra outer0.25',
        effective_backend='math00 FP32 MATH; deterministic Graph; TF32/autocast off',model_mode='eval with gradients enabled',
        dropout='unchanged eval behavior; attention dropout0',sampling=False,KL=0,logvar_frozen=True,
        betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,microbatch=1,meshes_per_update=50,
        new_updates=500,final_completed_updates=2500,checkpoints=CHECKS,representation_uids=SELECTED,
        step_log_semantics='loss and Edge metrics at state_before=completed_updates-1; gradient before clipping; actual FP32 displacement and Adam steps after update',
        stability='max and P95/P99 over the same500 pre-update records; top10 total-loss records and per-UID loss jumps (descriptive, no adaptive training)',
        code_sha256={p.name:sha(p) for p in ROOT.glob('*.py')})
    write(out/'config.json',config);write(out/'restore_verification.json',dict(**verification,branch=branch,only_change='param_group lr',actual_lrs=LRS[branch]))
    active=[p for g in groups.values() for p in g.values()];completed=0;best=-1;first=None;started=time.monotonic()
    def checkpoint():return save_checkpoint(out/f'checkpoint-new{completed:04d}-step{2000+completed:04d}.pt',model,opt,config,cp,completed)
    def assess(path):
        nonlocal best,first
        e=evaluate(model,uids,pools,objective,capture,batches,out,completed,path,parent_eval,completed==500)
        if e['joint_perfect']>best:
            best=e['joint_perfect'];write(out/'best.json',dict(new_step=completed,completed_updates=2000+completed,joint_perfect=best,checkpoint=str(path),sha256=e['checkpoint_sha256']))
        if e['joint_perfect']==50 and first is None:
            first=completed;write(out/'first50perfect.json',dict(new_step=completed,checkpoint=str(path),sha256=e['checkpoint_sha256']))
        return e
    try:
        summary=assess(checkpoint())
        with (out/'updates.jsonl').open('x',buffering=1) as log:
            for target in range(1,501):
                started_step=time.monotonic();opt.zero_grad(set_to_none=True);details=[]
                assert {g['name']:g['lr'] for g in opt.param_groups}==LRS[branch]
                for uid in uids:
                    rows,loss,parts=objective(uid)
                    observation=edge_observation(rows,pools[uid],scales)
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
                steps=group_steps(opt);assert set(steps.values())=={2000+completed}
                losses={k:sum(x[k] for x in details)/50 for k in ['edge','face']}
                row=dict(new_update=completed,completed_updates=2000+completed,state_before=1999+completed,
                    timing='loss and per-UID Edge counts are BEFORE this update; delta and Adam counters AFTER',
                    meshes=details,losses_before=losses,total_loss_before=losses['edge']+losses['face'],
                    edge_perfect_uids_before=[x['uid'] for x in details if x['edge_perfect']],
                    gradient_norms=group_norms,total_grad_norm=gn,clip_coefficient=min(1.,1/(gn+1e-6)),
                    lrs={g['name']:g['lr'] for g in opt.param_groups},actual_updates=updates,adam_steps=steps,
                    participation_per_mesh=2000+completed,seconds=time.monotonic()-started_step)
                log.write(json.dumps(row,allow_nan=False)+'\n')
                write(ROOT/'status.json',dict(state='training',branch=branch,new_updates=completed,completed_updates=2000+completed,
                    branch_budget=500,total_loss_before=row['total_loss_before'],last_update_seconds=row['seconds']))
                if completed<=3 or completed%10==0:print('UPDATE',branch,completed,row['total_loss_before'],row['seconds'],flush=True)
                if completed in CHECKS:
                    assert tensor_hash(model.autoencoder.log_variance.named_parameters())==frozen
                    summary=assess(checkpoint())
        assert completed==500 and set(group_steps(opt).values())=={2500}
        full=out/'model-step2500-inference.pt'
        T.save(dict(model=model.state_dict(),args=args,config=config,completed_updates=2500),full)
        done=dict(state='complete',branch=branch,new_updates=500,completed_updates=2500,stopped_at_budget=True,
            summary={k:v for k,v in summary.items() if k!='meshes'},best=json.loads((out/'best.json').read_text()),
            first50perfect_new_step=first,full_model=str(full),full_model_sha256=sha(full),seconds=time.monotonic()-started,
            frozen_logvar_unchanged=tensor_hash(model.autoencoder.log_variance.named_parameters())==frozen)
        write(out/'complete.json',done);write(ROOT/'status.json',done);print('BRANCH_COMPLETE',branch,flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(branch=branch,completed_new_updates=completed,error=str(exc),traceback=traceback.format_exc()))
        save_checkpoint(out/'failure-state.pt',model,opt,config,cp,completed)
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--branch',choices=list(LRS));p.add_argument('--precheck',action='store_true')
    a=p.parse_args();main(a.branch,a.precheck)
