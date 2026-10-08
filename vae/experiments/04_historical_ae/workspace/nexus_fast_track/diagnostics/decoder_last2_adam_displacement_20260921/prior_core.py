"""Bounded original-architecture decoder-tail diagnostic; no upstream updates."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import copy,fcntl,importlib.util,json,shutil,time
from pathlib import Path
from runtime import ROOT,T,np,b,c,setup,sha,write,tensor_hash
from head_core import metrics,summary,norm
CFG=json.loads((ROOT/'config.json').read_text());CONTROL=Path(CFG['control'])

def score(tail,x,pairs,labels,scale,scoring):
    cu=T.tensor([0,len(x)],dtype=T.int32,device=x.device)
    # Exact original sampling_forward.py decoder loop, including residual order.
    h=x
    for name in ['penultimate','block']:
        block=tail[name]
        h=h+b.unified_attention(block.attention,block.norm(h),cu,len(h))
    h=tail['final_norm'](h)
    raw=tail['head'](h);center=raw-raw.mean(0,keepdim=True)
    logits=scoring.first_order_interval(center[pairs[:,0]],center[pairs[:,1]])*scale
    ns,ms=scoring.soft4_sums(logits,labels)
    return h,(raw,center,logits,(ns/(ms+1e-8)).mean())

def scoring_module():
    p=ROOT/'effective_code/effective_loss_and_scoring.py'
    spec=importlib.util.spec_from_file_location('tail_scoring',p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prep():
    parent=Path(CFG['source_checkpoint']).parent.parent
    for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json','pool_provenance.json','data_validation.json','pools','source_archive','review_runtime']:
        p=(parent/name).resolve();assert p.exists(),p
        dest=ROOT/name
        if not dest.exists():dest.symlink_to(p)
        assert dest.resolve()==p
    assert sha(CFG['source_checkpoint'])==CFG['source_sha256']
    assert sha(CONTROL/'run/checkpoint-step0500.pt')==CFG['control_step500_sha256']
    for name in ['head_core.py','runtime.py','pipeline.py','config.json','construction_args.json']:
        if name.endswith('.py'):compile((ROOT/name).read_text(),str(ROOT/name),'exec')
    if not (ROOT/'effective_code').exists():shutil.copytree(CONTROL/'effective_code',ROOT/'effective_code')
    for name in ['checkpoint-step0000.json','checkpoint-step0500.json']:
        shutil.copyfile(CONTROL/'run'/name,ROOT/('control-'+name))

def export():
    model,opt,groups,uids,pools,forward,capture,args=setup();del opt,groups
    T.use_deterministic_algorithms(True)
    cp=T.load(CFG['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert cp['epoch']==900 and cp['completed_updates']==22500 and uids==cp['manifest']['uids']
    model.load_state_dict(cp['model'],strict=True);del cp
    model.requires_grad_(False);a=model.autoencoder
    assert len(a.decoder_blocks)==16 and not a.normalize_spacetime_embeddings
    for module in [a.decoder_blocks[-1],a.decoder_output_norm,a.edge_embedding]:module.requires_grad_(True)
    allowed={id(p) for m in [a.decoder_blocks[-1],a.decoder_output_norm,a.edge_embedding] for p in m.parameters()}
    assert allowed=={id(p) for p in model.parameters() if p.requires_grad}
    before=tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
    tail=T.nn.ModuleDict(dict(block=a.decoder_blocks[-1],final_norm=a.decoder_output_norm,head=a.edge_embedding))
    b.ATTENTION_NAMES[id(tail['block'].attention)]='decoder_15/attention'
    cache=ROOT/'cache';cache.mkdir(exist_ok=False)
    template=copy.deepcopy(tail).cpu()
    for module in template.modules():
        module._forward_hooks.clear();module._forward_pre_hooks.clear();module._backward_hooks.clear()
    T.save(template,cache/'initial_module.pt');del template
    state={k:v.detach().cpu().clone() for k,v in tail.state_dict().items()}
    T.save(state,cache/'initial_state.pt')
    grab={};hook=a.decoder_blocks[-1].norm.register_forward_pre_hook(lambda m,x:grab.update(x=x[0].detach()))
    control=json.loads((CONTROL/'cache/manifest.json').read_text());ref={x['uid']:x for x in control['meshes']}
    sc=scoring_module();scale=model.scoring_contract()['edge_logit_scale'];records=[];started=time.monotonic()
    g=c.m.h.soft4_loss.__globals__;old=g['soft4_sums'];g['soft4_sums']=b.full_sums
    try:
        for i,uid in enumerate(uids):
            rows=forward(uid);x=grab['x'];n=len(x);h=capture['hidden'].detach().clone();center=rows[2][0]
            pairs=T.triu_indices(n,n,1,device='cuda').T
            edges=np.unique(np.sort(pools[uid]['edges'].T,axis=1),axis=0)
            keys=T.as_tensor(edges@np.array([n,1]),device='cuda');q=pairs[:,0]*n+pairs[:,1]
            at=T.searchsorted(keys,q);labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q)
            loss,_=c.m.h.soft4_loss(center,keys,args['pair_chunk_size'],scale)
            fullgrad=T.autograd.grad(loss/100,tuple(tail.parameters()))
            hidden,v=score(tail,x,pairs,labels,scale,sc)
            cachegrad=T.autograd.grad(v[3]/100,tuple(tail.parameters()))
            assert T.equal(hidden,h) and T.equal(v[1],center) and T.equal(loss,v[3]),uid
            assert all(T.equal(g1,g2) for g1,g2 in zip(fullgrad,cachegrad)),uid
            r=metrics(dict(uid=uid,labels=labels),v)
            assert r['edge_soft4']==ref[uid]['edge_loss'] and all(r[k]==ref[uid]['counts'][k] for k in ['tp','fp','fn','tn'])
            with np.load(CONTROL/'cache'/f'{uid}.npz') as oldcache:
                assert np.array_equal(h.cpu().numpy(),oldcache['hidden']) and np.array_equal(v[0].detach().cpu().numpy(),oldcache['original_edge_raw']),uid
            p=cache/f'{uid}.npz'
            np.savez(p,last_block_input=x.cpu().numpy(),original_hidden=h.cpu().numpy(),original_edge_raw=v[0].detach().cpu().numpy(),edges=edges,faces=pools[uid]['positive'],vertices=pools[uid]['vertices'])
            records.append(dict(uid=uid,vertices=n,pairs=len(pairs),gt_edges=len(edges),sha256=sha(p),baseline=r,full_vs_cached_forward_gradient_bitwise=True,input_bytes=x.numel()*4))
            write(ROOT/'status.json',dict(stage='export',meshes=i+1,optimizer_updates=0))
            print('EXPORT',i+1,uid,flush=True)
            del rows,x,h,hidden,center,loss,v,fullgrad,cachegrad;capture.clear();grab.clear()
    finally:hook.remove();g['soft4_sums']=old
    assert before==tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
    write(cache/'manifest.json',dict(meshes=records,scale=scale,source_sha256=CFG['source_sha256'],source_unchanged=True,all100_forward_backward_bitwise=True,seconds=time.monotonic()-started,cache_input_bytes=sum(r['input_bytes'] for r in records),initial_module_sha256=sha(cache/'initial_module.pt')))
    write(ROOT/'freeze_contract.json',dict(trainable=[n for n,p in model.named_parameters() if p.requires_grad],trainable_count=sum(p.numel() for p in tail.parameters()),source_state_hash=before,cache_boundary='input to decoder_blocks.15.norm, before last attention/residual',frozen='Encoder/mu/logvar/Decoder blocks0..14/Face head; absent from optimizer and training graph'))
    write(ROOT/'export_complete.json',dict(meshes=100,optimizer_updates=0,seconds=time.monotonic()-started))
    del model,tail;T.cuda.empty_cache()

def load():
    meta=json.loads((ROOT/'cache/manifest.json').read_text())
    assert sha(ROOT/'cache/initial_module.pt')==meta['initial_module_sha256']
    tail=T.load(ROOT/'cache/initial_module.pt',map_location='cuda',weights_only=False).float().eval().requires_grad_(True)
    b.ATTENTION_NAMES[id(tail['block'].attention)]='decoder_15/attention'
    data=[]
    for row in meta['meshes']:
        p=ROOT/'cache'/f"{row['uid']}.npz";assert sha(p)==row['sha256']
        with np.load(p) as f:x=T.from_numpy(f['last_block_input'].copy()).cuda();gt=T.from_numpy(f['edges'].copy()).cuda()
        n=len(x);pairs=T.triu_indices(n,n,1,device='cuda').T;keys=gt[:,0]*n+gt[:,1];q=pairs[:,0]*n+pairs[:,1];at=T.searchsorted(keys,q)
        labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q)
        assert int(labels.sum())==row['gt_edges'] and not x.requires_grad
        data.append(dict(uid=row['uid'],x=x,pairs=pairs,labels=labels,reference=row))
    assert len(data)==100 and sum(len(d['pairs']) for d in data)==84669234
    return meta,tail,data,scoring_module()

def cycle(tail,data,sc,scale,backward):
    rows=[]
    for d in data:
        with T.enable_grad():
            hidden,v=score(tail,d['x'],d['pairs'],d['labels'],scale,sc)
            rows.append(metrics(d,v))
            if backward:(v[3]/100).backward()
        del hidden,v
    return rows

def benchmark(meta,tail,data,sc):
    start=time.monotonic();rows=cycle(tail,data,sc,meta['scale'],False);T.cuda.synchronize()
    elapsed=time.monotonic()-start
    for d,r in zip(data,rows):assert r==d['reference']['baseline'],r['uid']
    times=[];grads=[]
    T.cuda.reset_peak_memory_stats()
    for repeat in range(2):
        tail.zero_grad(set_to_none=True);start=time.monotonic()
        rows2=cycle(tail,data,sc,meta['scale'],True);T.cuda.synchronize();times.append(time.monotonic()-start)
        assert rows==rows2
        grads.append([p.grad.detach().clone() for p in tail.parameters()])
    assert all(T.equal(a,bv) for a,bv in zip(*grads))
    result=dict(full100_forward_seconds=elapsed,full100_backward_seconds=times,estimated500_seconds=500*sum(times)/2,baseline=summary(rows,0),all100_repeat_forward_backward_bitwise=True,peak_allocated=T.cuda.max_memory_allocated(),peak_reserved=T.cuda.max_memory_reserved(),parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in tail.items()},gradient_norms={k:float(sum(p.grad.double().square().sum() for p in m.parameters()).sqrt()) for k,m in tail.items()},optimizer_updates=0)
    write(ROOT/'benchmark.json',result);tail.zero_grad(set_to_none=True)
    print('BENCHMARK',json.dumps({k:v for k,v in result.items() if k!='baseline'}),flush=True)
    return rows

def train(meta,tail,data,sc,baseline):
    run=ROOT/'run';run.mkdir(exist_ok=False)
    groups=[dict(params=list(tail['block'].parameters())+list(tail['final_norm'].parameters()),lr=CFG['tail_lr'],name='decoder_tail'),dict(params=list(tail['head'].parameters()),lr=CFG['head_lr'],name='edge_head')]
    opt=T.optim.Adam(groups,betas=tuple(CFG['betas']),eps=CFG['eps'],weight_decay=0)
    assert not opt.state
    named=dict(tail.named_parameters());checkpoints=set(CFG['checkpoint_steps']);start=time.monotonic()
    logs=[(run/n).open('w',buffering=1) for n in ['updates.jsonl','evaluations.jsonl']]
    def save(step,rows):
        p=run/f'checkpoint-step{step:04d}.pt';temp=p.with_suffix('.tmp')
        T.save(dict(completed_updates=step,tail=tail.state_dict(),optimizer=opt.state_dict(),rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),config=CFG,cache_manifest_sha256=sha(ROOT/'cache/manifest.json')),temp);temp.replace(p)
        record=summary(rows,step);record['checkpoint_sha256']=sha(p);write(p.with_suffix('.json'),record)
    try:
        for step in range(CFG['updates']+1):
            tail.zero_grad(set_to_none=True)
            rows=cycle(tail,data,sc,meta['scale'],step<CFG['updates'])
            if step==0:assert rows==baseline
            result=summary(rows,step);logs[1].write(json.dumps(result,allow_nan=False)+'\n')
            if step in checkpoints:save(step,rows)
            write(ROOT/'status.json',dict(stage='training' if step<500 else 'verifying',completed_updates=step,budget=500,fp=result['total_fp'],fn=result['total_fn'],strict=result['edge_perfect'],objective=result['objective'],seconds=time.monotonic()-start))
            if step%10==0:print('STEP',step,result['objective'],result['edge_perfect'],result['total_fp'],result['total_fn'],flush=True)
            if step==CFG['updates']:break
            before={n:p.detach().clone() for n,p in named.items()}
            gn={k:float(sum(p.grad.double().square().sum() for p in m.parameters()).sqrt()) for k,m in tail.items()}
            total=T.nn.utils.clip_grad_norm_(tuple(tail.parameters()),1.0,error_if_nonfinite=True);coef=min(1.,1./(float(total)+1e-6))
            opt.step()
            updates={}
            for k,m in tail.items():
                delta=sum((p.detach()-before[k+'.'+n]).double().square().sum() for n,p in m.named_parameters()).sqrt()
                base=sum(before[k+'.'+n].double().square().sum() for n,p in m.named_parameters()).sqrt()
                updates[k]=dict(delta_norm=float(delta),relative=float(delta/base),parameter_norm=norm(T.cat([p.detach().flatten() for p in m.parameters()])))
            assert all(v['delta_norm']>0 for v in updates.values())
            logs[0].write(json.dumps(dict(update=step+1,mesh_count=100,objective_before=result['objective'],gradient_norms=gn,total_grad_norm=float(total),clip_coefficient=coef,actual_updates=updates,lrs={g['name']:g['lr'] for g in opt.param_groups}),allow_nan=False)+'\n')
    finally:
        for f in logs:f.close()
    # Reload saved endpoint before final verification/export.
    final=T.load(run/'checkpoint-step0500.pt',map_location='cuda',weights_only=False)
    tail.load_state_dict(final['tail']);verified=cycle(tail,data,sc,meta['scale'],False);assert verified==rows
    out=run/'final_outputs';out.mkdir()
    with T.enable_grad():
        for d in data:
            h,v=score(tail,d['x'],d['pairs'],d['labels'],meta['scale'],sc)
            np.savez_compressed(out/f"{d['uid']}.npz",edge_head_raw=v[0].detach().cpu().numpy(),edge_embedding=v[1].detach().cpu().numpy(),logits=v[2].detach().cpu().numpy())
    control=json.loads((ROOT/'control-checkpoint-step0500.json').read_text())
    write(run/'comparison.json',dict(control_step500=control,new_step500=summary(rows,500),same_source_sha256=CFG['source_sha256'],same_original_head=True,face_evaluated=False))
    assert sha(CFG['source_checkpoint'])==CFG['source_sha256']
    assert sha(CONTROL/'run/checkpoint-step0500.pt')==CFG['control_step500_sha256']
    for row in meta['meshes']:assert sha(ROOT/'cache'/f"{row['uid']}.npz")==row['sha256']
    write(run/'verification.json',dict(saved_endpoint_reloaded_exact=True,original_source_and_control_unchanged=True,cache_unchanged=True,only_tail_and_head_in_optimizer=True,all500_groups_updated=True,updates=500,evaluations=501,face_not_evaluated=True,attention_audit=b.AUDIT))
    write(run/'complete.json',dict(state='complete',completed_updates=500,seconds=time.monotonic()-start,final=summary(rows,500),stopped_at_budget=True))
    write(ROOT/'status.json',dict(state='complete',completed_updates=500,strict=result['edge_perfect'],fp=result['total_fp'],fn=result['total_fn']))
    print('COMPLETE',result['edge_perfect'],result['total_fp'],result['total_fn'],flush=True)

if __name__=='__main__':
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    T.set_num_threads(1);T.manual_seed(CFG['seed']);T.cuda.set_device(0);T.use_deterministic_algorithms(False)
    T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False;T.set_float32_matmul_precision('highest')
    prep()
    if not (ROOT/'export_complete.json').exists():export()
    T.use_deterministic_algorithms(True)
    meta,tail,data,sc=load();baseline=benchmark(meta,tail,data,sc)
    train(meta,tail,data,sc,baseline)
