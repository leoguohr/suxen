"""Same-parent A/B training: only historical-success Edge coefficient changes."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,fcntl,json,math,random,shutil,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'decoder_last2_joint_fixed100_20260920'
GRAD=ROOT.parent/'decoder_last2_gradient_groups_20260920'
FINITE=ROOT.parent/'decoder_last2_adam_displacement_20260921'
CFG=json.loads((ROOT/'config.json').read_text())
PARTS=['success_edge','success_face','failed_edge','failed_face']
LIVE={}


def main():
    lock=(ROOT/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'complete.json').exists(), 'Completed experiment cannot be overwritten'
    assert not any((ROOT/n).exists() for n in CFG['branches']), 'Existing branch state: inspect, do not restart'
    for name in ['READY.json','overfit100_manifest.csv','data_manifest.csv','selection.json',
                 'pool_provenance.json','pools','construction_args.json']:
        target=(SOURCE/name).resolve();assert target.exists(),target
        if not (ROOT/name).exists(): (ROOT/name).symlink_to(target)
        assert (ROOT/name).resolve()==target
    if not (ROOT/'effective_code').exists(): shutil.copytree(SOURCE/'effective_code',ROOT/'effective_code')
    from runtime import T,np,b,c,setup,sha,write,tensor_hash
    import prior_core as core
    import joint_core as joint
    import evaluate
    evaluate.FACE_SECONDS=float('inf')
    LIVE.update(T=T,write=write)
    started=time.monotonic()

    def same(x,y):
        if T.is_tensor(x): assert T.is_tensor(y) and x.dtype==y.dtype and T.equal(x.cpu(),y.cpu())
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x: same(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            assert len(x)==len(y)
            for a,z in zip(x,y): same(a,z)
        else: assert x==y,(x,y)

    protected={CFG['source_full']:CFG['source_full_sha256'],CFG['source_checkpoint']:CFG['source_checkpoint_sha256'],
               CFG['preserve74']:CFG['preserve74_sha256']}
    for path,digest in protected.items(): assert sha(path)==digest,path
    partition=json.loads((GRAD/'partition.json').read_text())
    assert sha(partition['source'])==partition['source_sha256']
    success,failed=set(partition['success71']),set(partition['failed29'])
    assert len(success)==71 and len(failed)==29 and not success&failed and 'nexus_2k_000520' in failed
    write(ROOT/'partition.json',partition)
    model,unused,groups,uids,_,forward,capture,args=setup();assert unused is None;del groups
    T.use_deterministic_algorithms(True)
    full=T.load(CFG['source_full'],map_location='cpu',mmap=True,weights_only=False)
    cp=T.load(CFG['source_checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(full['model'],strict=True);model.requires_grad_(False)
    a=model.autoencoder
    tail=T.nn.ModuleDict(dict(block=a.decoder_blocks[15],final_norm=a.decoder_output_norm,
        head=a.edge_embedding,face=a.face_embedding,penultimate=a.decoder_blocks[14]))
    tail.requires_grad_(True)
    named=list(tail.named_parameters());params=tuple(p for n,p in named);pnames=[n for n,p in named]
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in params}
    same(tail.state_dict(),cp['tail'])
    assert cp['completed_updates']==1500 and cp['cumulative_joint_updates']==1000 and cp['new_decoder14_updates']==500
    prefixes=('autoencoder.decoder_blocks.14.','autoencoder.decoder_blocks.15.',
              'autoencoder.decoder_output_norm.','autoencoder.edge_embedding.','autoencoder.face_embedding.')
    def frozen_hash():
        state=model.state_dict()
        return dict(tensors=tensor_hash((k,v) for k,v in state.items() if T.is_tensor(v) and not k.startswith(prefixes)),
                    metadata=json.dumps({k:v for k,v in state.items() if not T.is_tensor(v)},sort_keys=True))
    frozen0=frozen_hash()
    meta=json.loads((SOURCE/'cache/manifest.json').read_text())
    assert sha(SOURCE/'cache/manifest.json')==cp['cache_manifest_sha256']
    poolmeta=json.loads((SOURCE/'source_manifest.json').read_text())
    data=[]
    for uid,row in zip(uids,meta['meshes']):
        assert uid==row['uid'] and uid in success|failed
        path=SOURCE/'cache'/f'{uid}.npz';assert sha(path)==row['sha256']
        with np.load(path) as q:
            x=T.from_numpy(q['block14_input'].copy()).cuda();gt=T.from_numpy(q['edges'].copy()).cuda()
        n=len(x);pairs=T.triu_indices(n,n,1,device='cuda').T
        keys=gt[:,0]*n+gt[:,1];query=pairs[:,0]*n+pairs[:,1];at=T.searchsorted(keys,query)
        labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==query)
        path=SOURCE/'augmented_pools'/f'{uid}_pool.npz';assert sha(path)==poolmeta['augmented_pool_sha256'][uid]
        with np.load(path) as q: pool={k:q[k].copy() for k in ['vertices','edges','positive','mixed']}
        tris=T.as_tensor(np.concatenate([pool['positive'],pool['mixed']]),device='cuda')
        yl=T.cat([T.ones(len(pool['positive']),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')])
        data.append(dict(uid=uid,x=x,pairs=pairs,labels=labels,face_tris=tris,face_labels=yl,pool=pool))
    assert len(data)==100 and sum(len(d['pairs']) for d in data)==CFG['edge_pair_count']
    assert sum(len(d['face_tris']) for d in data)==CFG['face_pool_count']
    sc,scales=core.scoring_module(),model.scoring_contract()
    initial_scalar=json.loads((SOURCE/'run/checkpoint-new0500-tail1500.json').read_text())['meshes']
    original_actual={r['uid']:r for r in json.loads((SOURCE/'run/final_real_network.json').read_text())['meshes']}
    original72={uid for uid,r in original_actual.items() if r['joint_perfect']}
    assert len(original72)==72
    assert sha(GRAD/'gradient_vectors.pt')=='09d1b99493391317fa6661c6626d0282d9bb5ff23602418ab954a7f93749d9b3'
    vectors=T.load(GRAD/'gradient_vectors.pt',map_location='cpu',weights_only=False)
    assert vectors['parameter_names']==pnames
    finite_plus=T.load(FINITE/'points/plus1/tail-fp32.pt',map_location='cpu',weights_only=False)
    finite_result=json.loads((FINITE/'points/plus1/result.json').read_text())
    assert sha(FINITE/'points/plus1/tail-fp32.pt')==finite_result['state_sha256']
    finite_actual={r['uid']:r for r in map(json.loads,(FINITE/'points/plus1/meshes.jsonl').read_text().splitlines())}
    python0=random.getstate();numpy0=np.random.get_state()

    def rng():
        return dict(rng=T.get_rng_state().clone(),cuda_rng=[s.clone() for s in T.cuda.get_rng_state_all()],
                    python_rng=random.getstate(),numpy_rng=np.random.get_state())
    def rng_hash():
        r=rng()
        return dict(torch=tensor_hash([('cpu',r['rng'])]+[(str(i),s) for i,s in enumerate(r['cuda_rng'])]),
                    python=repr(r['python_rng']),numpy=repr(r['numpy_rng']))
    def group_steps(opt):
        return {g['name']:sorted({int(opt.state[p]['step']) for p in g['params']}) for g in opt.param_groups}
    def dot(x,y): return sum(float((a.double()*z.double()).sum()) for a,z in zip(x,y))
    def norm(xs): return math.sqrt(max(0.,dot(xs,xs)))
    def four(rows):
        out={p:0. for p in PARTS}
        for row in rows:
            prefix='success' if row['uid'] in success else 'failed'
            out[prefix+'_edge']+=row['edge_soft4']/100;out[prefix+'_face']+=row['face_soft4']/100
        return out
    def cycle(coefficient,backward):
        rows=[]
        for d in data:
            h,ev,fv=joint.score(tail,d,meta['scale'],sc,scales)
            row=core.metrics(d,ev);row['face_soft4']=float(fv[3].detach());rows.append(row)
            # Control preserves the exact original arithmetic/accumulation order.
            if coefficient==1. or d['uid'] in failed: loss=(ev[3]+fv[3])/100
            else: loss=(coefficient*ev[3]+fv[3])/100
            assert bool(T.isfinite(loss)),d['uid']
            if backward: loss.backward()
            del h,ev,fv,loss;capture.clear()
        return rows

    write(ROOT/'protocol_verification.json',dict(source_hashes=protected,partition_sha256=sha(ROOT/'partition.json'),
        uids=uids,selection_sha256=sha(ROOT/'selection.json'),data_manifest_sha256=sha(ROOT/'data_manifest.csv'),
        cache_manifest_sha256=sha(SOURCE/'cache/manifest.json'),augmented_pool_sha256=poolmeta['augmented_pool_sha256'],
        face_training_candidates=750215,all_edge_pairs=84669234,trainable_names=pnames,frozen_hash=frozen0,
        original_joint72=sorted(original72),historical_partition='71/29 fixed; 000520 in F',
        source_rng_fields=['rng','cuda_rng'],additional_rng_note='Source lacks Python/NumPy RNG fields. Both branches use identical captured startup states; these streams do not participate in deterministic mu training.',
        full_real_verification='all100 at all eight checkpoints per branch',config=CFG))
    print('PREPARED: source/cache/pools verified, 750215 Face candidates, 84669234 pairs',flush=True)
    baseline_hashes=[];branch_completions={}
    for branch,coefficient in CFG['branches'].items():
        out=ROOT/branch;out.mkdir();(out/'run').mkdir();LIVE.pop('optimizer',None);LIVE.update(branch=branch,step=0,tail=tail,out=out)
        model.load_state_dict(full['model'],strict=True);same(tail.state_dict(),cp['tail']);assert frozen_hash()==frozen0
        optimizer_groups=[dict(params=list(tail['block'].parameters())+list(tail['final_norm'].parameters()),name='decoder_tail'),
            dict(params=list(tail['head'].parameters()),name='edge_head'),dict(params=list(tail['face'].parameters()),name='face_head'),
            dict(params=list(tail['penultimate'].parameters()),name='decoder14')]
        opt=T.optim.Adam(optimizer_groups);opt.load_state_dict(copy.deepcopy(cp['optimizer']));same(opt.state_dict(),cp['optimizer'])
        LIVE['optimizer']=opt
        for g in opt.param_groups:
            assert g['lr']==CFG['lr'][g['name']] and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
            assert not g['amsgrad'] and not g['maximize'] and not g['capturable'] and not g['differentiable']
        assert group_steps(opt)==dict(decoder_tail=[1500],edge_head=[1500],face_head=[1500],decoder14=[500])
        T.set_rng_state(cp['rng'].cpu());T.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
        random.setstate(python0);np.random.set_state(numpy0)
        tail.zero_grad(set_to_none=True)
        start_hash=dict(model=tensor_hash((n,t) for n,t in model.state_dict().items() if T.is_tensor(t)),rng=rng_hash())
        baseline_hashes.append(start_hash)
        if len(baseline_hashes)>1: assert baseline_hashes[0]==baseline_hashes[-1]
        write(out/'restore_verification.json',dict(weights_buffers_source_exact=True,adam_all_moments_steps_groups_exact=True,
            restored_group_steps=group_steps(opt),actual_groups=[{k:v for k,v in g.items() if k!='params'} for g in opt.param_groups],
            baseline_state_hashes=start_hash,success_edge_coefficient=coefficient,independent_source_restore=True))
        first_audit=None;last_eval=None
        logs=(out/'run/updates.jsonl').open('w',buffering=1)
        traces=(out/'run/training_trace.jsonl').open('w',buffering=1)
        for step in range(501):
            LIVE['step']=step
            tail.zero_grad(set_to_none=True)
            tick=time.monotonic();rows=cycle(coefficient,step<500)
            losses=four(rows);original=sum(losses.values());objective=original+(coefficient-1)*losses['success_edge']
            if step==0: assert rows==initial_scalar
            trace=dict(branch=branch,new_step=step,components=losses,original_objective=original,
                optimized_objective=objective,success_edge_coefficient=coefficient,meshes=rows)
            traces.write(json.dumps(trace,allow_nan=False)+'\n')
            if step in CFG['checkpoints']:
                # Freeze a resumable snapshot before all-checkpoint evaluation.
                state=dict(tail={n:p.detach().cpu().clone() for n,p in tail.state_dict().items()},
                    optimizer=copy.deepcopy(opt.state_dict()),**rng(),new_updates=step,completed_updates=1500+step,
                    cumulative_joint_updates=1000+step,decoder14_adam_step=500+step,config=CFG,branch=branch,
                    success_edge_coefficient=coefficient,cache_manifest_sha256=cp['cache_manifest_sha256'],
                    per_uid_new_training_participations={u:step for u in uids})
                path=out/'run'/f'checkpoint-new{step:04d}.pt';tmp=path.with_suffix('.tmp')
                T.save(state,tmp);tmp.replace(path)
                same(tail.state_dict(),state['tail']);same(opt.state_dict(),state['optimizer']);del state
                before_eval_rng=rng_hash();predictions=out/'run'/f'predictions-new{step:04d}';predictions.mkdir()
                actual=[];prediction_files=[]
                for index,(d,expected_row) in enumerate(zip(data,rows)):
                    real=forward(d['uid']);real_hidden=capture['hidden'].clone()
                    h,ev,fv=joint.score(tail,d,meta['scale'],sc,scales)
                    assert T.equal(h.detach(),real_hidden) and T.equal(real[2][0],ev[1]) and T.equal(real[3][0],fv[1]),d['uid']
                    measured=core.metrics(d,ev);measured['face_soft4']=float(fv[3].detach());assert measured==expected_row
                    npz=predictions/f"{d['uid']}.npz"
                    r=evaluate.evaluate_mesh(real,d['pool'],scales,prediction_path=npz)
                    assert r['face']['complete'] and all(r['edge'][k]==expected_row[k] for k in ['tp','fp','fn','tn'])
                    r.update(uid=d['uid'],vertices=len(d['x']),fixed_group='S' if d['uid'] in success else 'F',
                        edge_soft4=expected_row['edge_soft4'],face_soft4=expected_row['face_soft4'],
                        face_training_pool=c.metrics(d['face_labels'].cpu().numpy().astype(bool),fv[2].detach().cpu().numpy()))
                    if step==0 or (branch=='A_control' and step==1):
                        ref=original_actual[d['uid']] if step==0 else finite_actual[d['uid']]
                        for key in ['edge','face','gt_face_candidates','missing_gt_face_candidates','edge_perfect','face_perfect','joint_perfect','margins']:
                            assert r[key]==ref[key],(branch,step,d['uid'],key)
                    prediction_files.append(dict(uid=d['uid'],path=str(npz.relative_to(ROOT)),sha256=sha(npz),bytes=npz.stat().st_size))
                    actual.append(r)
                    if (index+1)%25==0:print('ACTUAL',branch,step,index+1,'/100',flush=True)
                    write(ROOT/'status.json',dict(stage='actual100_real_network',branch=branch,new_updates=step,evaluated_meshes=index+1,budget=500))
                    del real,real_hidden,h,ev,fv;capture.clear()
                assert before_eval_rng==rng_hash()
                summary={}
                for group in ['all','S','F']:
                    rs=[r for r in actual if group=='all' or r['fixed_group']==group]
                    summary[group]=dict(meshes=len(rs),edge_perfect=sum(r['edge_perfect'] for r in rs),
                        joint_perfect=sum(r['joint_perfect'] for r in rs),
                        **{kind:{k:sum(r[kind][k] for r in rs) for k in ['tp','fp','fn','tn']} for kind in ['edge','face']},
                        missing_gt_face_candidates=sum(r['missing_gt_face_candidates'] for r in rs))
                perfect={r['uid'] for r in actual if r['joint_perfect']}
                result=dict(branch=branch,new_updates=step,components=losses,original_objective=original,
                    optimized_objective=objective,success_edge_coefficient=coefficient,summary=summary,meshes=actual,
                    joint_perfect_uids=sorted(perfect),retained_source72=sorted(perfect&original72),
                    lost_source72=sorted(original72-perfect),new_over_source72=sorted(perfect-original72),
                    large_meshes=[r for r in actual if r['vertices']>1500],prediction_files=prediction_files,
                    checkpoint_sha256=sha(path),adam_steps=group_steps(opt),
                    verification=dict(all100_real_vs_cached_hidden_edge_face_bitwise=True,all100_actual_face_complete=True,evaluation_rng_unchanged=True))
                write(out/'run'/f'actual-new{step:04d}.json',result)
                if step==0: assert summary['all']['edge_perfect']==summary['all']['joint_perfect']==72 and perfect==original72
                if step==1:
                    first_audit['components_after']=losses
                    first_audit['actual_component_changes']={k:losses[k]-first_audit['components_before'][k] for k in PARTS}
                    first_audit['actual100_summary_after']=summary
                    if branch=='A_control':
                        assert losses==finite_result['losses']
                        first_audit['matches_saved_lambda_plus1_all_parameters_losses_counts_margins']=True
                    write(out/'first_step_audit.json',first_audit)
                last_eval=result
                print('EVAL_COMPLETE',branch,step,json.dumps(dict(summary=summary['all'],losses=losses)),flush=True)
                assert frozen_hash()==frozen0
            if step==500: break
            if branch=='A_control' and step==0:
                assert all(T.equal(p.grad.detach().cpu(),v) for p,v in zip(params,vectors['original_total_gradient']))
            before=[p.detach().clone() for p in params]
            gradient_groups={k:norm([p.grad for p in m.parameters()]) for k,m in tail.items()}
            raw_first=[p.grad.detach().cpu().clone() for p in params] if step==0 else None
            total=T.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            clip=min(1.,1./(float(total)+1e-6))
            opt.step();LIVE['step']=step+1
            assert all(bool(T.isfinite(p).all()) for p in params)
            delta=[p.detach()-q for p,q in zip(params,before)]
            updates={}
            for module in tail:
                ids=[i for i,(n,p) in enumerate(named) if n.startswith(module+'.')]
                dn=norm([delta[i] for i in ids]);base=norm([before[i] for i in ids])
                updates[module]=dict(delta_norm=dn,relative=dn/base)
                assert dn>0,(module,step)
            expected_steps=dict(decoder_tail=[1501+step],edge_head=[1501+step],face_head=[1501+step],decoder14=[501+step])
            assert group_steps(opt)==expected_steps
            record=dict(branch=branch,update=step+1,mesh_count=100,components_before=losses,
                original_objective_before=original,optimized_objective_before=objective,success_edge_coefficient=coefficient,
                gradient_norms=gradient_groups,global_gradient_norm=float(total),clip_coefficient=clip,
                actual_updates=updates,actual_delta_norm=norm(delta),actual_relative_delta=norm(delta)/norm(before),
                adam_steps=expected_steps,seconds=time.monotonic()-tick)
            logs.write(json.dumps(record,allow_nan=False)+'\n')
            if step==0:
                actual=[q.detach().cpu() for q in delta]
                if branch=='A_control':
                    same(tail.state_dict(),finite_plus)
                    assert all(T.equal(q,z) for q,z in zip(actual,vectors['cloned_adam_actual_fp32_displacement']))
                first_audit=dict(**record,parameter_names=pnames,
                    unweighted_gradient_dot_actual_displacement={k:dot(vectors['gradients'][k],actual) for k in PARTS},
                    saved_unweighted_gradients_source=str(GRAD/'gradient_vectors.pt'),
                    source_moments_restored=True,new_weighted_gradient_used_in_actual_adam=True)
                # Full four unweighted gradients refer to the identical verified source point.
                T.save(dict(parameter_names=pnames,raw_objective_gradient=raw_first,actual_fp32_displacement=actual,
                    theta_before={n:q.cpu() for (n,p),q in zip(named,before)},theta_after={n:p.detach().cpu() for n,p in named}),out/'first_step_tensors.pt')
                write(out/'first_step_audit_pending.json',first_audit)
            write(ROOT/'status.json',dict(stage='training',branch=branch,new_updates=step+1,budget=500,
                components_before=losses,edge_fp=sum(r['fp'] for r in rows),edge_fn=sum(r['fn'] for r in rows)))
            if (step+1)%25==0: print('UPDATE',branch,step+1,original,flush=True)
            del before,delta
        logs.close();traces.close();tail.zero_grad(set_to_none=True)
        final=out/'run/checkpoint-new0500.pt'
        reloaded=T.load(final,map_location='cpu',weights_only=False)
        same(tail.state_dict(),reloaded['tail']);same(opt.state_dict(),reloaded['optimizer'])
        assert frozen_hash()==frozen0 and all(p.grad is None for p in model.parameters())
        final_model=out/'run/model-new0500-inference.pt'
        T.save(dict(model={k:v.detach().cpu() if T.is_tensor(v) else copy.deepcopy(v) for k,v in model.state_dict().items()},
            args=full['args'],inference_only=True,source=CFG['source_full'],source_sha256=CFG['source_full_sha256'],
            branch=branch,success_edge_coefficient=coefficient,new_updates=500,cumulative_joint_updates=1500,
            optimizer_checkpoint=str(final),optimizer_checkpoint_sha256=sha(final)),final_model)
        branch_completions[branch]=dict(state='complete',new_updates=500,stopped_at_budget=True,full_model=str(final_model),
            full_model_sha256=sha(final_model),checkpoint=str(final),checkpoint_sha256=sha(final),
            final_summary=last_eval['summary'],final_components=last_eval['components'],final_adam_steps=group_steps(opt),
            all8_checkpoints_all100_real_network_verified=True,frozen_state_unchanged=True)
        write(out/'complete.json',branch_completions[branch]);print('BRANCH_COMPLETE',branch,flush=True)
        del reloaded,opt
    for path,digest in protected.items(): assert sha(path)==digest,path
    for row in meta['meshes']: assert sha(SOURCE/'cache'/f"{row['uid']}.npz")==row['sha256']
    for uid,digest in poolmeta['augmented_pool_sha256'].items(): assert sha(SOURCE/'augmented_pools'/f'{uid}_pool.npz')==digest
    write(ROOT/'complete.json',dict(state='complete',branches=branch_completions,total_branch_optimizer_steps=1000,
        per_branch_updates=500,independent_identical_source_states=True,source_files_unchanged=True,
        all_pools_cache_unchanged=True,seconds=time.monotonic()-started))
    write(ROOT/'status.json',dict(state='complete',branches=branch_completions,total_branch_optimizer_steps=1000))
    print('COMPLETE BOTH BRANCHES',flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc())
        if 'optimizer' in LIVE:
            T=LIVE['T'];out=LIVE['out']
            T.save(dict(tail=LIVE['tail'].state_dict(),optimizer=LIVE['optimizer'].state_dict(),
                rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),branch=LIVE['branch'],
                completed_updates_at_failure=LIVE['step']),out/'failure-state.pt')
        raise
