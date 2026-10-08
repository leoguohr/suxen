"""500 Face-head updates on frozen installed-network features; fixed real Edge candidates."""
import fcntl,inspect,json,shutil,time,traceback
from pathlib import Path
from runtime import ROOT,T,np,b,c,setup,write,sha,tensor_hash
from face_core import center,objective,train_cycle,evaluate_one,summary
CFG=json.loads((ROOT/'config.json').read_text());PARENT=Path(CFG['parent'])

def frozen_hash(model):
    state=model.state_dict()
    return dict(tensors=tensor_hash((k,v) for k,v in state.items() if T.is_tensor(v) and not k.startswith('autoencoder.face_embedding.')),metadata=json.dumps({k:v for k,v in state.items() if not T.is_tensor(v)},sort_keys=True))

def prepare():
    assert sha(Path(CFG['source_checkpoint']))==CFG['source_sha256']
    for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json','pool_provenance.json','data_validation.json','pools','source_archive','review_runtime','augmented_pools','mining_complete.json']:
        source=(PARENT/name).resolve();assert source.exists(),source
        target=ROOT/name
        if not target.exists():target.symlink_to(source)
        assert target.resolve()==source
    for old,new in [('source_manifest.json','source_manifest.json'),('run/final_real_network.json','baseline_reference.json'),('run/complete.json','parent_complete.json')]:shutil.copyfile(PARENT/old,ROOT/new)
    model,opt,groups,uids,base_pools,forward,capture,args=setup();assert opt is None;del groups
    T.use_deterministic_algorithms(True)
    cp=T.load(CFG['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert cp['inference_only'] and cp['source_sha256']=='8de6074700fafa02b3adcfafd352a4555ce0cbc88421667261f29495faa69640' and cp['cumulative_tail_updates']==1000 and cp['face_head_updates']==0
    assert sha(cp['tail_checkpoint'])==cp['tail_checkpoint_sha256']
    model.load_state_dict(cp['model'],strict=True);saved_args=cp['args'];del cp
    model.requires_grad_(False);head=model.autoencoder.face_embedding;head.requires_grad_(True)
    assert {n for n,p in model.named_parameters() if p.requires_grad}=={'autoencoder.face_embedding.weight','autoencoder.face_embedding.bias'}
    assert head.weight.shape==(32,1024) and not model.autoencoder.normalize_spacetime_embeddings
    scales=model.scoring_contract();manifest=json.loads((ROOT/'source_manifest.json').read_text());assert scales==manifest['scales']
    original_rest=frozen_hash(model)
    cache=ROOT/'cache';cache.mkdir(exist_ok=False)
    T.save(head.state_dict(),cache/'face_head_initial.pt')
    records=[];data=[];ref={r['uid']:r for r in json.loads((ROOT/'baseline_reference.json').read_text())['meshes']}
    for i,uid in enumerate(uids):
        poolfile=ROOT/'augmented_pools'/f'{uid}_pool.npz';digest=sha(poolfile);assert digest==manifest['augmented_pool_sha256'][uid]
        with np.load(poolfile) as f:pool={k:f[k] for k in ['vertices','edges','positive','mixed']}
        assert np.array_equal(pool['vertices'],base_pools[uid]['vertices']) and np.array_equal(pool['positive'],base_pools[uid]['positive'])
        rows=forward(uid);h=capture['hidden'].detach();e=rows[2][0].detach();f=rows[3][0];n=len(h)
        previous_output=PARENT/'run/final_outputs'/f'{uid}.npz'
        with np.load(previous_output) as previous:
            assert np.array_equal(h.cpu().numpy(),previous['hidden']) and np.array_equal(e.cpu().numpy(),previous['edge_embedding']) and np.array_equal(f.detach().cpu().numpy(),previous['face_embedding']),uid
        tr=np.concatenate([pool['positive'],pool['mixed']]);gt=np.sort(pool['positive'],axis=1)
        gk=c.m.probe.keys(gt,n);tk=c.m.probe.keys(tr,n)
        assert not np.isin(c.m.probe.keys(pool['mixed'],n),gk).any()
        train_tris=T.as_tensor(tr,device='cuda',dtype=T.long);train_labels=T.cat([T.ones(len(gt),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')])
        full_loss,_=objective(f,train_tris,train_labels,scales);full_grad=T.autograd.grad(full_loss/100,tuple(head.parameters()))
        raw,fc=center(head,h);loss,_=objective(fc,train_tris,train_labels,scales);cached_grad=T.autograd.grad(loss/100,tuple(head.parameters()))
        assert T.equal(f,fc) and T.equal(full_loss,loss) and all(T.equal(a,z) for a,z in zip(full_grad,cached_grad)),uid
        assert float(loss)==ref[uid]['face_pool_soft4'],uid
        pairs=T.triu_indices(n,n,1,device='cuda').T
        with T.no_grad():el=c.edge_logits(e,pairs,scales);pred=el>0
        edges=np.unique(np.sort(pool['edges'].T,axis=1),axis=0);keys=edges@np.array([n,1]);pn=pairs.cpu().numpy();ey=np.isin(pn@np.array([n,1]),keys)
        edge_labels=T.from_numpy(ey).cuda();adj=np.zeros((n,n),dtype=bool);pe=pn[pred.cpu().numpy()];adj[pe[:,0],pe[:,1]]=True
        actual=[]
        for a in range(n):
            for z in np.flatnonzero(adj[a]):actual.extend((a,int(z),int(k)) for k in np.flatnonzero(adj[a]&adj[z]))
        actual=np.asarray(actual,dtype=np.int64).reshape(-1,3);ak=c.m.probe.keys(actual,n)
        assert len(np.unique(ak))==len(ak) and len(actual)==ref[uid]['face']['scored_candidates']
        covered=adj[gt[:,0],gt[:,1]]&adj[gt[:,0],gt[:,2]]&adj[gt[:,1],gt[:,2]]
        d=dict(uid=uid,hidden=h,edge_embedding=e,pairs=pairs,edge_prediction=pred,edge_labels=edge_labels,
               train_tris=train_tris,train_labels=train_labels,gt_faces=T.from_numpy(gt).cuda(),gt_covered=T.from_numpy(covered).cuda(),actual_tris=T.from_numpy(actual).cuda(),actual_labels=T.from_numpy(np.isin(ak,gk)).cuda(),actual_in_training_pool=T.from_numpy(np.isin(ak,tk)).cuda(),reference=ref[uid])
        baseline,_=evaluate_one(head,d,scales)
        for k in ['tp','fp','fn','tn']:
            assert baseline['edge'][k]==ref[uid]['edge'][k] and baseline['face'][k]==ref[uid]['face'][k],(uid,k)
        assert baseline['face_soft4']==ref[uid]['face_pool_soft4'] and baseline['joint_perfect']==ref[uid]['joint_perfect']
        assert baseline['margins']['face_gt_all']==ref[uid]['margins']['face_gt_all'] and baseline['margins']['face_non_gt_actual']==ref[uid]['margins']['face_non_gt_scored']
        p=cache/f'{uid}.npz'
        np.savez_compressed(p,hidden=h.cpu().numpy(),edge_embedding=e.cpu().numpy(),predicted_edges=pe,train_triangles=tr,train_labels=train_labels.cpu().numpy(),actual_triangles=actual,actual_labels=np.isin(ak,gk),actual_in_training_pool=np.isin(ak,tk),gt_faces=gt,gt_covered=covered,vertices=pool['vertices'],edges=edges,initial_face_embedding=fc.detach().cpu().numpy())
        record=dict(uid=uid,vertices=n,train_candidates=len(tr),actual_candidates=len(actual),gt_faces=len(gt),pool_sha256=digest,cache_sha256=sha(p),parent_outputs_sha256=sha(previous_output),baseline=baseline,
                    actual_candidates_sha256=tensor_hash([('actual_triangles',d['actual_tris'])]),hidden_sha256=tensor_hash([('hidden',h)]),full_vs_cached_face_loss_gradient_bitwise=True)
        records.append(record);data.append(d)
        write(ROOT/'status.json',dict(stage='cache_and_baseline',completed=i+1,uid=uid,optimizer_updates=0));print('BASELINE',i+1,uid,baseline['face']['fp'],baseline['face']['fn'],flush=True)
        del rows,raw,fc,loss,full_loss,full_grad,cached_grad,el;capture.clear()
    assert original_rest==frozen_hash(model) and all(p.grad is None for p in model.parameters())
    baseline=summary([r['baseline'] for r in records],0)
    assert {k:baseline[v] for k,v in [('perfect','edge_perfect'),('fp','edge_fp'),('fn','edge_fn')]}==CFG['expected_edge']
    assert all(baseline[k]==v for k,v in CFG['expected_step0'].items())
    cachemeta=dict(source_checkpoint=CFG['source_checkpoint'],source_sha256=CFG['source_sha256'],meshes=records,scales=scales,hidden_total_bytes=sum(d['hidden'].numel()*4 for d in data),train_candidates=sum(r['train_candidates'] for r in records),actual_candidates=sum(r['actual_candidates'] for r in records),frozen_state_hash=original_rest)
    write(cache/'manifest.json',cachemeta);write(ROOT/'baseline.json',baseline)
    code='\n\n'.join(inspect.getsource(x) for x in [c.face_logits,c.m.topology._split_spacetime,c.m.topology._squared_parallelogram_area,c.m.topology.second_order_interval,c.m.topology.face_interval_logits,c.soft])
    (ROOT/'effective_face_scoring.py.txt').write_text(code+'\n\n# Fully-diff membership runtime source:\n'+(ROOT/'loss_changes_exact_runtime.txt').read_text())
    write(ROOT/'freeze_contract.json',dict(trainable=['autoencoder.face_embedding.weight','autoencoder.face_embedding.bias'],trainable_parameters=sum(p.numel() for p in head.parameters()),hidden='installed network final hidden, verified against real forward and prior installation hashes',edge='frozen installed-network prediction; not GT graph',pool='unchanged round2 augmented pool; positive followed by mixed, no reorder',frozen_state_hash=original_rest))
    return model,head,data,scales,forward,capture,cachemeta,saved_args

def main():
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model,head,data,scales,forward,capture,meta,saved_args=prepare()
    # A zero-update full100 benchmark and repeatability gate precede fresh Adam.
    times=[];grads=[]
    for repeat in range(2):
        head.zero_grad(set_to_none=True);T.cuda.synchronize();started=time.monotonic()
        losses=train_cycle(head,data,scales,True);T.cuda.synchronize();times.append(time.monotonic()-started)
        grads.append([p.grad.detach().clone() for p in head.parameters()])
        assert losses==[r['baseline']['face_soft4'] for r in meta['meshes']]
    assert all(T.equal(a,z) for a,z in zip(*grads));head.zero_grad(set_to_none=True)
    write(ROOT/'benchmark.json',dict(full100_forward_backward_seconds=times,repeated_gradient_bitwise=True,optimizer_updates=0,peak_allocated=T.cuda.max_memory_allocated(),peak_reserved=T.cuda.max_memory_reserved()))
    print('BENCHMARK',times,flush=True)
    run=ROOT/'run';run.mkdir(exist_ok=False)
    opt=T.optim.Adam(head.parameters(),lr=CFG['lr'],betas=tuple(CFG['betas']),eps=CFG['eps'],weight_decay=0);assert not opt.state
    checks=set(CFG['checks']);logs=(run/'updates.jsonl').open('w',buffering=1);evals=[];started=time.monotonic()
    for step in range(CFG['updates']+1):
        if step in checks:
            rows=[]
            for d in data:
                row,saved=evaluate_one(head,d,scales,save_logits=step==500);rows.append(row)
                if saved is not None:
                    folder=run/'final_outputs';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f"{d['uid']}.npz",**saved)
            result=summary(rows,step);assert {k:result[v] for k,v in [('perfect','edge_perfect'),('fp','edge_fp'),('fn','edge_fn')]}==CFG['expected_edge']
            if step==0:assert result==json.loads((ROOT/'baseline.json').read_text())
            evals.append(result);write(run/f'evaluation-step{step:04d}.json',result)
            with (run/'evaluations.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            p=run/f'checkpoint-step{step:04d}.pt';tmp=p.with_suffix('.tmp')
            T.save(dict(step=step,face_head=head.state_dict(),optimizer=opt.state_dict(),rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),config=CFG,cache_manifest_sha256=sha(ROOT/'cache/manifest.json')),tmp);tmp.replace(p)
            write(p.with_suffix('.json'),dict(step=step,sha256=sha(p),face_perfect=result['face_perfect'],joint_perfect=result['joint_perfect'],face_fp=result['face_fp'],face_fn=result['face_fn']))
            print('EVAL',step,result['joint_perfect'],result['face_fp'],result['face_fn'],flush=True)
        if step==CFG['updates']:break
        head.zero_grad(set_to_none=True);losses=train_cycle(head,data,scales,True)
        before={n:p.detach().clone() for n,p in head.named_parameters()};gn={n:float(p.grad.detach().double().norm()) for n,p in head.named_parameters()}
        total=T.nn.utils.clip_grad_norm_(tuple(head.parameters()),1.0,error_if_nonfinite=True);coefficient=min(1.,1./(float(total)+1e-6));opt.step()
        updates={n:dict(delta_norm=float((p.detach()-before[n]).double().norm()),relative=float((p.detach()-before[n]).double().norm()/before[n].double().norm()),norm=float(p.detach().double().norm()),changed_elements=int((p.detach()!=before[n]).sum())) for n,p in head.named_parameters()}
        assert updates['weight']['changed_elements']>0
        row=dict(update=step+1,mesh_count=100,mean_face_soft4_before=sum(losses)/100,per_mesh_losses=losses,gradient_norms=gn,total_grad_norm=float(total),clip_coefficient=coefficient,parameter_updates=updates,lr=opt.param_groups[0]['lr'])
        logs.write(json.dumps(row,allow_nan=False)+'\n')
        write(ROOT/'status.json',dict(stage='training',completed_updates=step+1,budget=500,mean_face_soft4=row['mean_face_soft4_before']))
        if (step+1)%25==0:print('UPDATE',step+1,row['mean_face_soft4_before'],flush=True)
    logs.close()
    verify_and_finish(model,head,data,scales,forward,capture,meta,saved_args,evals,started)

def verify_and_finish(model,head,data,scales,forward,capture,meta,saved_args,evals,started):
    run=ROOT/'run';write(ROOT/'status.json',dict(stage='final_real_network_verification',completed_updates=500))
    cp=T.load(run/'checkpoint-step0500.pt',map_location='cuda',weights_only=False);assert cp['step']==500
    head.load_state_dict(cp['face_head'],strict=True);del cp
    # Frozen weights, features, pools and candidate membership must survive every update.
    assert frozen_hash(model)==meta['frozen_state_hash']
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('autoencoder.face_embedding.'))
    for d,record in zip(data,meta['meshes']):
        assert sha(ROOT/'cache'/f"{d['uid']}.npz")==record['cache_sha256']
        assert sha(ROOT/'augmented_pools'/f"{d['uid']}_pool.npz")==record['pool_sha256']
        assert tensor_hash([('actual_triangles',d['actual_tris'])])==record['actual_candidates_sha256']
        assert tensor_hash([('hidden',d['hidden'])])==record['hidden_sha256']
    verified=[]
    for i,d in enumerate(data):
        rows=forward(d['uid']);h=capture['hidden'].detach()
        assert T.equal(h,d['hidden']) and T.equal(rows[2][0],d['edge_embedding'])
        with T.no_grad():raw,f=center(head,d['hidden'])
        assert T.equal(f,rows[3][0]),d['uid']
        row,_=evaluate_one(head,d,scales,embedding=rows[3][0].detach())
        assert row==evals[-1]['meshes'][i],d['uid'];verified.append(row)
        print('REAL_NETWORK',i+1,d['uid'],row['joint_perfect'],flush=True)
        del rows,h,raw,f;capture.clear()
    write(run/'final_real_network.json',summary(verified,500))
    assert frozen_hash(model)==meta['frozen_state_hash'] and sha(Path(CFG['source_checkpoint']))==CFG['source_sha256']
    installed=run/'model-tail1000-face1000-inference.pt'
    T.save(dict(model={k:v.detach().cpu() if T.is_tensor(v) else v for k,v in model.state_dict().items()},args=saved_args,inference_only=True,source_checkpoint=CFG['source_checkpoint'],source_sha256=CFG['source_sha256'],face_checkpoint=str(run/'checkpoint-step0500.pt'),face_checkpoint_sha256=sha(run/'checkpoint-step0500.pt'),new_face_updates=500,cumulative_face_updates=1000,frozen_tail_updates=1000,other_parameter_updates=0),installed)
    initial={r['uid'] for r in evals[0]['meshes'] if r['joint_perfect']};retention=[]
    for state in evals:
        current={r['uid'] for r in state['meshes'] if r['joint_perfect']}
        retention.append(dict(step=state['step'],retained=sorted(initial&current),lost=sorted(initial-current),gained=sorted(current-initial),success_uids=sorted(current)))
    verification=dict(updates=500,each_update_meshes=100,face_only_parameters=32800,source_file_unchanged=True,frozen_weights_buffers_metadata_unchanged=True,all_cached_hidden_unchanged=True,all_actual_candidates_unchanged=True,all_pool_files_unchanged=True,all_checks_edge_state=CFG['expected_edge'],final_head_checkpoint_reloaded=True,all100_real_network_matches_final_cache=True,attention_audit=b.AUDIT)
    write(run/'verification.json',verification)
    complete=dict(state='complete',updates=500,budget_stopped=True,seconds=time.monotonic()-started,initial=evals[0],final=evals[-1],retention=retention,inference_copy=str(installed),inference_sha256=sha(installed),source_checkpoint=CFG['source_checkpoint'],source_sha256=CFG['source_sha256'])
    write(run/'complete.json',complete);write(ROOT/'status.json',dict(state='complete',completed_updates=500,joint_perfect=evals[-1]['joint_perfect'],face_fp=evals[-1]['face_fp'],face_fn=evals[-1]['face_fn']))
    print('COMPLETE',evals[-1]['joint_perfect'],evals[-1]['face_fp'],evals[-1]['face_fn'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'failure.txt').write_text(traceback.format_exc());raise
