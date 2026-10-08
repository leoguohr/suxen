"""Paired two-block Joint diagnostic: restore three Adam groups and add empty decoder14 state."""
import fcntl,json,shutil,time,traceback
from pathlib import Path
from runtime import ROOT,T,np,b,c,setup,write,sha,tensor_hash
import prior_core as core
import evaluate
import joint_core as joint
import face_core
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])

def same(a,z):
    if T.is_tensor(a):assert T.is_tensor(z) and a.dtype==z.dtype and T.equal(a.cpu(),z.cpu())
    elif isinstance(a,dict):
        assert a.keys()==z.keys()
        for k in a:same(a[k],z[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(z)
        for x,y in zip(a,z):same(x,y)
    else:assert a==z,(a,z)

def frozen_hash(model):
    prefixes=('autoencoder.decoder_blocks.14.','autoencoder.decoder_blocks.15.','autoencoder.decoder_output_norm.','autoencoder.edge_embedding.','autoencoder.face_embedding.')
    state=model.state_dict()
    return dict(tensors=tensor_hash((k,v) for k,v in state.items() if T.is_tensor(v) and not k.startswith(prefixes)),metadata=json.dumps({k:v for k,v in state.items() if not T.is_tensor(v)},sort_keys=True))

def face_summary(rows,step):
    return dict(step=step,cumulative_joint_updates=step+500,cumulative_tail_updates=step+1000,new_decoder14_updates=step,meshes=rows,**{k:sum(r[k] for r in rows) for k in ['edge_perfect','face_perfect','joint_perfect','missing_gt_face_candidates']},**{kind+'_'+k:sum(r[kind][k] for r in rows) for kind in ['edge','face'] for k in ['tp','fp','fn']})

def actual_one(uid,hidden,edge,pool,scales,face_head):
    with T.no_grad():
        raw=face_head(hidden);face=raw-raw.mean(0,keepdim=True)
        result=evaluate.evaluate_mesh(((),(),(edge,),(face,)),pool,scales)
        tri=T.as_tensor(np.concatenate([pool['positive'],pool['mixed']]),device='cuda')
        y=T.cat([T.ones(len(pool['positive']),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')])
        logits=c.face_logits(face,tri,scales);w=b.full_weights(logits,y);ns,ms=c.soft(logits,y,w)
        loss=float((ns/(ms+1e-8)).mean())
    assert result['face']['complete']
    return dict(uid=uid,vertices=len(hidden),gt_faces=len(pool['positive']),face_pool_soft4=loss,training_pool=c.metrics(y.cpu().numpy().astype(bool),logits.cpu().numpy()),**result),face

def prepare():
    assert sha(CFG['source_checkpoint'])==CFG['source_sha256']
    assert sha(CFG['parent_checkpoint'])==CFG['parent_checkpoint_sha256']
    assert sha(CFG['preserve_joint74'])==CFG['preserve_joint74_sha256']
    assert sha(CFG['preserve_joint72'])==CFG['preserve_joint72_sha256']
    for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json','pool_provenance.json','data_validation.json','pools','source_archive','review_runtime','augmented_pools','mining_complete.json']:
        target=(PARENT/name).resolve();assert target.exists()
        (ROOT/name).symlink_to(target)
    (ROOT/'cache').mkdir(exist_ok=False)
    shutil.copytree(PARENT/'effective_code',ROOT/'effective_code')
    for src,name in [(PARENT/'run/checkpoint-new0500-tail1000.json','parent_joint_step500.json'),(PARENT/'run/final_real_network.json','parent_actual_baseline.json'),(PARENT/'source_manifest.json','source_manifest.json')]:shutil.copyfile(src,ROOT/name)
    model,unused,groups,uids,base_pools,forward,capture,args=setup();assert unused is None;del groups
    T.use_deterministic_algorithms(True)
    full=T.load(CFG['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert full['inference_only'] and full['new_joint_updates']==500 and full['cumulative_tail_updates']==1000
    model.load_state_dict(full['model'],strict=True);saved_args=full['args'];del full
    model.requires_grad_(False);a=model.autoencoder
    tail=T.nn.ModuleDict(dict(block=a.decoder_blocks[15],final_norm=a.decoder_output_norm,head=a.edge_embedding,face=a.face_embedding,penultimate=a.decoder_blocks[14]));tail.requires_grad_(True)
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in tail.parameters()}
    before=frozen_hash(model)
    sc=core.scoring_module()
    cp=T.load(CFG['parent_checkpoint'],map_location='cpu',weights_only=False)
    assert cp['completed_updates']==1000 and cp['new_updates']==500
    same({k:v for k,v in tail.state_dict().items() if not k.startswith('penultimate.')},cp['tail'])
    # All weights already come from the same full parent, including the newly unfrozen block.
    groups=[dict(params=list(tail['block'].parameters())+list(tail['final_norm'].parameters()),lr=1e-5,name='decoder_tail'),dict(params=list(tail['head'].parameters()),lr=1e-4,name='edge_head'),dict(params=list(tail['face'].parameters()),lr=1e-4,name='face_head')]
    opt=T.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0);opt.load_state_dict(cp['optimizer'])
    same(opt.state_dict(),cp['optimizer'])
    assert {g['name']:g['lr'] for g in opt.param_groups}=={'decoder_tail':1e-5,'edge_head':1e-4,'face_head':1e-4}
    assert all(float(state['step'])==1000 for state in opt.state.values())
    opt.add_param_group(dict(params=list(tail['penultimate'].parameters()),lr=1e-5,name='decoder14'))
    assert all(p not in opt.state for p in tail['penultimate'].parameters())
    restored=opt.state_dict();same(restored['state'],cp['optimizer']['state']);same(restored['param_groups'][:3],cp['optimizer']['param_groups'])
    T.set_rng_state(cp['rng'].cpu());T.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']])
    same(T.get_rng_state(),cp['rng']);same(T.cuda.get_rng_state_all(),cp['cuda_rng'])
    write(ROOT/'restore_verification.json',dict(source_checkpoint_sha256=CFG['source_sha256'],parent_checkpoint_sha256=CFG['parent_checkpoint_sha256'],existing_four_modules_match_both_full_model_and_parent=True,new_block_weights_from_same_parent=True,adam_all_moments_steps_and_groups_exact=True,adam_initial_step=1000,available_cpu_cuda_rng_restored=True,all_three_groups_restored_together=True,new_decoder14_adam_empty=True,lrs={g['name']:g['lr'] for g in opt.param_groups},trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],frozen_state_hash=before,cache_boundary=CFG['cache_boundary']))
    del cp
    pools={};manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for uid in uids:
        f=ROOT/'augmented_pools'/f'{uid}_pool.npz';assert sha(f)==manifest['augmented_pool_sha256'][uid]
        with np.load(f) as p:pools[uid]={k:p[k] for k in ['vertices','edges','positive','mixed']}
    data=[];records=[];meta=dict(scale=model.scoring_contract()['edge_logit_scale'])
    scales=model.scoring_contract()
    # Export only the frozen prefix output: before the newly trainable block14.
    grab={};hook=a.decoder_blocks[14].norm.register_forward_pre_hook(lambda m,x:grab.update(x=x[0].detach()))
    started=time.monotonic()
    for i,uid in enumerate(uids):
        rows=forward(uid);fullhidden=capture['hidden'];fullx=grab['x'];n=len(fullx)
        pool=pools[uid];pairs=T.triu_indices(n,n,1,device='cuda').T
        gt=np.unique(np.sort(pool['edges'].T,axis=1),axis=0)
        keys=T.as_tensor(gt[:,0]*n+gt[:,1],device='cuda');q=pairs[:,0]*n+pairs[:,1];at=T.searchsorted(keys,q)
        labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q)
        assert fullx.shape==(n,1024) and fullx.dtype==T.float32 and not fullx.requires_grad
        d=dict(uid=uid,x=fullx,pairs=pairs,labels=labels,face_tris=T.as_tensor(np.concatenate([pool['positive'],pool['mixed']]),device='cuda'),face_labels=T.cat([T.ones(len(pool['positive']),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')]))
        data.append(d)
        h,v,fv=joint.score(tail,d,meta['scale'],sc,scales)
        full_edge_logits=sc.first_order_interval(rows[2][0][pairs[:,0]],rows[2][0][pairs[:,1]])*meta['scale']
        ns,ms=sc.soft4_sums(full_edge_logits,labels);full_edge=(ns/(ms+1e-8)).mean()
        full_face,_=face_core.objective(rows[3][0],d['face_tris'],d['face_labels'],scales)
        full_grads=T.autograd.grad((full_edge+full_face)/100,tuple(tail.parameters()))
        cached_grads=T.autograd.grad((v[3]+fv[3])/100,tuple(tail.parameters()))
        assert T.equal(full_edge,v[3]) and T.equal(full_face,fv[3])
        assert all(T.equal(x,y) for x,y in zip(full_grads,cached_grads)),uid
        assert T.equal(fullhidden,h) and T.equal(rows[2][0],v[1]),uid
        with T.no_grad():raw=a.face_embedding(h);f=raw-raw.mean(0,keepdim=True)
        assert T.equal(rows[3][0],f)
        path=ROOT/'cache'/f'{uid}.npz'
        np.savez(path,block14_input=fullx.cpu().numpy(),original_hidden=fullhidden.cpu().numpy(),original_edge_raw=v[0].detach().cpu().numpy(),original_face_raw=raw.cpu().numpy(),edges=gt,faces=pool['positive'],vertices=pool['vertices'])
        records.append(dict(uid=uid,vertices=n,pairs=len(pairs),gt_edges=len(gt),sha256=sha(path),input_bytes=fullx.numel()*4,full_vs_cached_forward_gradient_bitwise=True))
        del rows,fullhidden,fullx,h,v,raw,f,fv,full_edge_logits,full_edge,full_face,full_grads,cached_grads,ns,ms;capture.clear();grab.clear()
        write(ROOT/'status.json',dict(stage='cache_export_before_block14',meshes=i+1,optimizer_updates=0))
        if (i+1)%20==0:print('BASELINE_FORWARD',i+1,flush=True)
    assert len(data)==100 and sum(len(d['pairs']) for d in data)==84669234
    meta.update(meshes=records,cache_boundary=CFG['cache_boundary'],source_sha256=CFG['source_sha256'],cache_input_bytes=sum(m['input_bytes'] for m in records),export_seconds=time.monotonic()-started)
    write(ROOT/'cache/manifest.json',meta)
    write(ROOT/'freeze_contract.json',dict(cache_boundary=CFG['cache_boundary'],trainable=[n for n,p in model.named_parameters() if p.requires_grad],parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in tail.items()},frozen_state_hash=before))
    hook.remove();assert frozen_hash(model)==before
    write(ROOT/'cache_gradient_verification.json',dict(all100_real_vs_cached_combined_gradient_bitwise=True,all100_forward_embeddings_bitwise=True,optimizer_updates=0))
    return model,tail,opt,data,sc,meta,pools,forward,capture,saved_args,before

def main():
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evaluate.FACE_SECONDS=float('inf')
    model,tail,opt,data,sc,meta,pools,forward,capture,saved_args,rest=prepare()
    scales=model.scoring_contract();face_head=model.autoencoder.face_embedding
    run=ROOT/'run';run.mkdir(exist_ok=False)
    parent=json.loads((ROOT/'parent_joint_step500.json').read_text())
    sourceface=json.loads((ROOT/'parent_actual_baseline.json').read_text())
    initial_rows=joint.cycle(tail,data,meta['scale'],sc,scales,False)
    assert initial_rows==parent['meshes']
    initial_opt=opt.state_dict();initial_rng=T.get_rng_state().clone();initial_cuda=T.cuda.get_rng_state_all()
    gradients=[];times=[];T.cuda.reset_peak_memory_stats()
    for _ in range(2):
        tail.zero_grad(set_to_none=True);t=time.monotonic();rr=joint.cycle(tail,data,meta['scale'],sc,scales,True);T.cuda.synchronize();times.append(time.monotonic()-t)
        assert rr==initial_rows;gradients.append([p.grad.detach().clone() for p in tail.parameters()])
    assert all(T.equal(x,y) for x,y in zip(*gradients));del gradients
    same(initial_opt,opt.state_dict());same(initial_rng,T.get_rng_state());same(initial_cuda,T.cuda.get_rng_state_all())
    tail.zero_grad(set_to_none=True)
    write(ROOT/'benchmark.json',dict(all100_forward_backward_seconds=times,repeated_gradients_bitwise=True,optimizer_updates=0,weights_adam_rng_unchanged=True,cache_boundary=CFG['cache_boundary'],peak_allocated_bytes=T.cuda.max_memory_allocated(),peak_reserved_bytes=T.cuda.max_memory_reserved(),cache_input_bytes=meta['cache_input_bytes'],parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in tail.items()}))
    logs=(run/'updates.jsonl').open('w',buffering=1);trace=(run/'edge_trace.jsonl').open('w',buffering=1)
    started=time.monotonic();joint_checks=[];checks=set(CFG['checkpoint_steps'])
    for step in range(501):
        tail.zero_grad(set_to_none=True);rows=joint.cycle(tail,data,meta['scale'],sc,scales,step<500)
        edge=joint.summary(rows,step);edge['cumulative_tail_updates']=step+1000;edge['cumulative_joint_updates']=step+500;edge['new_decoder14_updates']=step
        if step==0:assert rows==initial_rows
        trace.write(json.dumps(edge,allow_nan=False)+'\n')
        if step in checks:
            p=run/f'checkpoint-new{step:04d}-tail{step+1000:04d}.pt'
            T.save(dict(completed_updates=step+1000,new_updates=step,cumulative_joint_updates=step+500,new_decoder14_updates=step,tail=tail.state_dict(),optimizer=opt.state_dict(),rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),config=CFG,cache_manifest_sha256=sha(ROOT/'cache/manifest.json')),p)
            write(p.with_suffix('.json'),dict(**edge,sha256=sha(p)))
            diagnostic=[]
            for d,expected in zip(data,rows):
                with T.enable_grad():h,v=core.score(tail,d['x'],d['pairs'],d['labels'],meta['scale'],sc)
                r,f=actual_one(d['uid'],h.detach(),v[1].detach(),pools[d['uid']],scales,face_head)
                assert all(r['edge'][k]==expected[k] for k in ['tp','fp','fn','tn'])
                assert r['face_pool_soft4']==expected['face_soft4']
                r['edge_soft4']=expected['edge_soft4']
                diagnostic.append(r);del h,v,f
            result=face_summary(diagnostic,step);joint_checks.append(result)
            if step==0:
                assert all(result[k]==v for k,v in CFG['expected_start'].items())
                for x,y in zip(result['meshes'],sourceface['meshes']):
                    assert x['uid']==y['uid'] and x['edge']==y['edge']
                    assert all(x['face'][k]==y['face'][k] for k in ['tp','fp','fn','tn','scored_candidates'])
                    assert x['face_pool_soft4']==y['face_pool_soft4'] and x['edge_soft4']==y['edge_soft4']
                    assert x['training_pool']==y['training_pool'] and x['margins']==y['margins']
            write(run/f'actual-new{step:04d}.json',result)
            with (run/'actual_evaluations.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            print('ACTUAL',step,{k:v for k,v in result.items() if k!='meshes'},flush=True)
        write(ROOT/'status.json',dict(stage='training' if step<500 else 'final_real_network',completed_new_updates=step,cumulative_tail_updates=step+1000,budget_new_updates=500,edge_perfect=edge['edge_perfect'],fp=edge['total_fp'],fn=edge['total_fn'],objective=edge['objective']))
        if step%25==0:print('EDGE',step,edge['edge_perfect'],edge['total_fp'],edge['total_fn'],edge['objective'],flush=True)
        if step==500:break
        before={n:p.detach().clone() for n,p in tail.named_parameters()}
        gn={k:float(sum(p.grad.double().square().sum() for p in m.parameters()).sqrt()) for k,m in tail.items()}
        total=T.nn.utils.clip_grad_norm_(tuple(tail.parameters()),1.0,error_if_nonfinite=True);coef=min(1.,1./(float(total)+1e-6));opt.step()
        changes={}
        for k,m in tail.items():
            delta=sum((p.detach()-before[k+'.'+n]).double().square().sum() for n,p in m.named_parameters()).sqrt();base=sum(before[k+'.'+n].double().square().sum() for n,p in m.named_parameters()).sqrt()
            changes[k]=dict(delta_norm=float(delta),relative=float(delta/base),parameter_norm=core.norm(T.cat([p.detach().flatten() for p in m.parameters()])))
        assert all(v['delta_norm']>0 for v in changes.values())
        logs.write(json.dumps(dict(update=step+1,cumulative_tail_update=step+1001,decoder14_update=step+1,mesh_count=100,objective_before=edge['objective'],edge_loss_before=edge['edge_objective'],face_loss_before=edge['face_objective'],gradient_norms=gn,total_grad_norm=float(total),clip_coefficient=coef,actual_updates=changes,lrs={g['name']:g['lr'] for g in opt.param_groups}),allow_nan=False)+'\n')
    logs.close();trace.close();tail.zero_grad(set_to_none=True)
    cp=T.load(run/'checkpoint-new0500-tail1500.pt',map_location='cuda',weights_only=False);tail.load_state_dict(cp['tail']);same(opt.state_dict(),cp['optimizer'])
    assert all(float(opt.state[p]['step'])==1500 for g in opt.param_groups[:3] for p in g['params'])
    assert all(float(opt.state[p]['step'])==500 for p in opt.param_groups[3]['params']);del cp
    assert frozen_hash(model)==rest and all(p.grad is None for p in model.parameters())
    actual=[];out=run/'final_outputs';out.mkdir()
    for i,d in enumerate(data):
        rows=forward(d['uid']);hidden=capture['hidden'];h,v=core.score(tail,d['x'],d['pairs'],d['labels'],meta['scale'],sc)
        assert T.equal(hidden,h) and T.equal(rows[2][0],v[1])
        r,f=actual_one(d['uid'],hidden,rows[2][0].detach(),pools[d['uid']],scales,face_head)
        r['edge_soft4']=float(v[3].detach())
        assert T.equal(f,rows[3][0]);expected=joint_checks[-1]['meshes'][i]
        assert {k:v for k,v in r.items() if k!='face_seconds'}=={k:v for k,v in expected.items() if k!='face_seconds'}
        actual.append(r)
        np.savez_compressed(out/f"{d['uid']}.npz",hidden=hidden.cpu().numpy(),edge_embedding=v[1].detach().cpu().numpy(),face_embedding=f.cpu().numpy(),edge_logits=v[2].detach().cpu().numpy(),vertices=pools[d['uid']]['vertices'],edges=np.unique(np.sort(pools[d['uid']]['edges'].T,axis=1),axis=0),faces=pools[d['uid']]['positive'])
        print('REAL_NETWORK',i+1,d['uid'],r['joint_perfect'],flush=True)
        del rows,hidden,h,v,f;capture.clear()
    assert frozen_hash(model)==rest
    assert sha(CFG['source_checkpoint'])==CFG['source_sha256'] and sha(CFG['parent_checkpoint'])==CFG['parent_checkpoint_sha256']
    assert sha(CFG['preserve_joint74'])==CFG['preserve_joint74_sha256']
    assert sha(CFG['preserve_joint72'])==CFG['preserve_joint72_sha256']
    for r in meta['meshes']:assert sha(ROOT/'cache'/f"{r['uid']}.npz")==r['sha256']
    final_model=run/'model-last2-joint1000-tail1500-block14new500-inference.pt'
    T.save(dict(model={k:v.detach().cpu() if T.is_tensor(v) else v for k,v in model.state_dict().items()},args=saved_args,inference_only=True,source_checkpoint=CFG['source_checkpoint'],source_sha256=CFG['source_sha256'],tail_checkpoint=str(run/'checkpoint-new0500-tail1500.pt'),tail_checkpoint_sha256=sha(run/'checkpoint-new0500-tail1500.pt'),new_joint_updates=500,cumulative_tail_updates=1500,cumulative_face_updates=1500,cumulative_joint_updates=1000,new_decoder14_updates=500,optimizer_checkpoint=str(run/'checkpoint-new0500-tail1500.pt')),final_model)
    write(run/'final_real_network.json',face_summary(actual,500))
    write(run/'verification.json',dict(new_updates=500,cumulative_tail_updates=1500,original_three_adam_groups_final1500=True,new_decoder14_adam_final500=True,endpoint_tail_checkpoint_reloaded=True,all100_real_network_matches_cache=True,frozen_parameters_buffers_metadata_unchanged=True,face_head_trainable=True,face_adam_final_step1500=True,parent_joint_full_model_unchanged=True,parent_joint_adam_checkpoint_unchanged=True,joint72_baseline_unchanged=True,joint74_baseline_unchanged=True,cached_pre_block14_inputs_unchanged=True,face_candidates_reenumerated_from_each_predicted_edge_graph=True,all_actual_face_evaluations_complete=True,attention_audit=b.AUDIT))
    complete=dict(state='complete',new_updates=500,cumulative_tail_updates=1500,cumulative_joint_updates=1000,new_decoder14_updates=500,stopped_at_budget=True,seconds=time.monotonic()-started,initial=joint_checks[0],final=joint_checks[-1],initial_edge=joint.summary(initial_rows,0),final_edge=edge,inference_copy=str(final_model),inference_sha256=sha(final_model),source_baseline_sha256=CFG['source_sha256'])
    write(run/'complete.json',complete);write(ROOT/'status.json',dict(state='complete',new_updates=500,cumulative_tail_updates=1500,cumulative_joint_updates=1000,edge_perfect=edge['edge_perfect'],joint_perfect=joint_checks[-1]['joint_perfect']))
    print('COMPLETE',edge['edge_perfect'],joint_checks[-1]['joint_perfect'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc());raise
