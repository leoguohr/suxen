"""Recover the interrupted task from new100; exact replay verification before continuing to new500."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('last2_task_entry',Path(__file__).with_name('run.py'))
_entry=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_entry)
globals().update({k:v for k,v in vars(_entry).items() if not k.startswith('_')})

def prepare_resume():
    assert not (ROOT/'run/complete.json').exists() and not (ROOT/'interrupted_attempt0').exists()
    assert sha(CFG['source_checkpoint'])==CFG['source_sha256'] and sha(CFG['parent_checkpoint'])==CFG['parent_checkpoint_sha256']
    model,unused,groups,uids,base_pools,forward,capture,args=setup();assert unused is None;del groups
    T.use_deterministic_algorithms(True)
    full=T.load(CFG['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(full['model'],strict=True);saved_args=full['args'];del full
    model.requires_grad_(False);a=model.autoencoder
    tail=T.nn.ModuleDict(dict(block=a.decoder_blocks[15],final_norm=a.decoder_output_norm,head=a.edge_embedding,face=a.face_embedding,penultimate=a.decoder_blocks[14]));tail.requires_grad_(True)
    path=ROOT/'run/checkpoint-new0100-tail1100.pt';digest=sha(path)
    assert digest==json.loads(path.with_suffix('.json').read_text())['sha256']
    cp=T.load(path,map_location='cpu',weights_only=False);assert cp['new_updates']==100 and cp['new_decoder14_updates']==100
    tail.load_state_dict(cp['tail'],strict=True);same(tail.state_dict(),cp['tail'])
    groups=[dict(params=list(tail['block'].parameters())+list(tail['final_norm'].parameters()),lr=1e-5,name='decoder_tail'),dict(params=list(tail['head'].parameters()),lr=1e-4,name='edge_head'),dict(params=list(tail['face'].parameters()),lr=1e-4,name='face_head'),dict(params=list(tail['penultimate'].parameters()),lr=1e-5,name='decoder14')]
    opt=T.optim.Adam(groups,betas=(.9,.999),eps=1e-8,weight_decay=0);opt.load_state_dict(cp['optimizer']);same(opt.state_dict(),cp['optimizer'])
    assert all(float(opt.state[p]['step'])==1100 for g in opt.param_groups[:3] for p in g['params']) and all(float(opt.state[p]['step'])==100 for p in opt.param_groups[3]['params'])
    before=frozen_hash(model);assert before==json.loads((ROOT/'restore_verification.json').read_text())['frozen_state_hash']
    assert {id(p) for p in model.parameters() if p.requires_grad}=={id(p) for p in tail.parameters()}
    meta=json.loads((ROOT/'cache/manifest.json').read_text());assert sha(ROOT/'cache/manifest.json')==cp['cache_manifest_sha256']
    manifest=json.loads((ROOT/'source_manifest.json').read_text());pools={};data=[];sc=core.scoring_module()
    for uid,row in zip(uids,meta['meshes']):
        assert uid==row['uid'];path=ROOT/'cache'/f'{uid}.npz';assert sha(path)==row['sha256']
        with np.load(path) as p:x=T.from_numpy(p['block14_input'].copy()).cuda();gt=T.from_numpy(p['edges'].copy()).cuda()
        n=len(x);pairs=T.triu_indices(n,n,1,device='cuda').T;keys=gt[:,0]*n+gt[:,1];q=pairs[:,0]*n+pairs[:,1];at=T.searchsorted(keys,q);labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q)
        path=ROOT/'augmented_pools'/f'{uid}_pool.npz';assert sha(path)==manifest['augmented_pool_sha256'][uid]
        with np.load(path) as p:pool={k:p[k] for k in ['vertices','edges','positive','mixed']}
        pools[uid]=pool;data.append(dict(uid=uid,x=x,pairs=pairs,labels=labels,face_tris=T.as_tensor(np.concatenate([pool['positive'],pool['mixed']]),device='cuda'),face_labels=T.cat([T.ones(len(pool['positive']),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')])))
    T.set_rng_state(cp['rng'].cpu());T.cuda.set_rng_state_all([s.cpu() for s in cp['cuda_rng']]);same(T.get_rng_state(),cp['rng']);same(T.cuda.get_rng_state_all(),cp['cuda_rng']);del cp
    write(ROOT/'resume_restore_verification.json',dict(resumed_from_new_step=100,checkpoint_sha256=digest,weights_all_four_adam_groups_and_rng_exact=True,original_groups_step1100=True,decoder14_step100=True,frozen_state_and_cache_exact=True,previous_logged_updates179=True,replay_steps101_to179=True))
    print('RESUME_RESTORED',100,digest,flush=True)
    return model,tail,opt,data,sc,meta,pools,forward,capture,saved_args,before

def main():
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    evaluate.FACE_SECONDS=float('inf')
    model,tail,opt,data,sc,meta,pools,forward,capture,saved_args,rest=prepare_resume()
    scales=model.scoring_contract();face_head=model.autoencoder.face_embedding
    run=ROOT/'run'
    parent=json.loads((run/'checkpoint-new0100-tail1100.json').read_text())
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
    write(ROOT/'resume_benchmark.json',dict(all100_forward_backward_seconds=times,repeated_gradients_bitwise=True,optimizer_updates=0,weights_adam_rng_unchanged=True,cache_boundary=CFG['cache_boundary'],peak_allocated_bytes=T.cuda.max_memory_allocated(),peak_reserved_bytes=T.cuda.max_memory_reserved(),cache_input_bytes=meta['cache_input_bytes'],parameter_counts={k:sum(p.numel() for p in m.parameters()) for k,m in tail.items()}))
    archive=ROOT/'interrupted_attempt0';archive.mkdir(exist_ok=False)
    for n in ['run/updates.jsonl','run/edge_trace.jsonl','run/actual_evaluations.jsonl','run/actual-new0100.json','status.json','launcher.json','finalize_launcher.json','console.log']:
        dst=archive/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
    replay_updates=[json.loads(s) for s in (archive/'run/updates.jsonl').read_text().splitlines()]
    replay_trace=[json.loads(s) for s in (archive/'run/edge_trace.jsonl').read_text().splitlines()]
    old_actual=[json.loads(s) for s in (archive/'run/actual_evaluations.jsonl').read_text().splitlines()]
    assert len(replay_updates)==179 and replay_updates[-1]['update']==179 and replay_trace[-1]['step']==178
    (run/'updates.jsonl').write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in replay_updates[:100]))
    (run/'edge_trace.jsonl').write_text(''.join(json.dumps(e,allow_nan=False)+'\n' for e in replay_trace[:100]))
    (run/'actual_evaluations.jsonl').write_text(json.dumps(old_actual[0],allow_nan=False)+'\n')
    logs=(run/'updates.jsonl').open('a',buffering=1);trace=(run/'edge_trace.jsonl').open('a',buffering=1)
    replay_checked=[];started=time.monotonic();joint_checks=[old_actual[0]];checks=set(CFG['checkpoint_steps'])
    for step in range(100,501):
        tail.zero_grad(set_to_none=True);rows=joint.cycle(tail,data,meta['scale'],sc,scales,step<500)
        edge=joint.summary(rows,step);edge['cumulative_tail_updates']=step+1000;edge['cumulative_joint_updates']=step+500;edge['new_decoder14_updates']=step
        if step==100:assert rows==initial_rows
        if step<len(replay_trace):assert edge==replay_trace[step],('replay_forward_diff',step)
        trace.write(json.dumps(edge,allow_nan=False)+'\n')
        if step in checks:
            p=run/f'checkpoint-new{step:04d}-tail{step+1000:04d}.pt'
            if step!=100:T.save(dict(completed_updates=step+1000,new_updates=step,cumulative_joint_updates=step+500,new_decoder14_updates=step,tail=tail.state_dict(),optimizer=opt.state_dict(),rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),config=CFG,cache_manifest_sha256=sha(ROOT/'cache/manifest.json')),p)
            if step!=100:write(p.with_suffix('.json'),dict(**edge,sha256=sha(p)))
            diagnostic=[]
            for d,expected in zip(data,rows):
                with T.enable_grad():h,v=core.score(tail,d['x'],d['pairs'],d['labels'],meta['scale'],sc)
                r,f=actual_one(d['uid'],h.detach(),v[1].detach(),pools[d['uid']],scales,face_head)
                assert all(r['edge'][k]==expected[k] for k in ['tp','fp','fn','tn'])
                assert r['face_pool_soft4']==expected['face_soft4']
                r['edge_soft4']=expected['edge_soft4']
                diagnostic.append(r);del h,v,f
            result=face_summary(diagnostic,step);joint_checks.append(result)
            if step==100:
                sourceface=old_actual[1]
                assert all(result[k]==sourceface[k] for k in CFG['expected_start'])
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
        record=dict(update=step+1,cumulative_tail_update=step+1001,decoder14_update=step+1,mesh_count=100,objective_before=edge['objective'],edge_loss_before=edge['edge_objective'],face_loss_before=edge['face_objective'],gradient_norms=gn,total_grad_norm=float(total),clip_coefficient=coef,actual_updates=changes,lrs={g['name']:g['lr'] for g in opt.param_groups})
        if step<len(replay_updates):
            assert record==replay_updates[step],('replay_update_diff',step+1)
            replay_checked.append(step+1)
            if step+1==179:write(ROOT/'resume_replay_verification.json',dict(replayed_updates=replay_checked,all_forward_rows_gradients_clipping_and_actual_updates_exact=True,extra_replayed_optimizer_executions=79,unique_trajectory_target=500))
        logs.write(json.dumps(record,allow_nan=False)+'\n')
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
    complete=dict(state='complete',new_updates=500,cumulative_tail_updates=1500,cumulative_joint_updates=1000,new_decoder14_updates=500,stopped_at_budget=True,seconds=time.monotonic()-started,resumed_from_new_step=100,extra_replayed_optimizer_executions=79,total_optimizer_executions_including_lost_attempt=579,initial=joint_checks[0],final=joint_checks[-1],initial_edge=replay_trace[0],final_edge=edge,inference_copy=str(final_model),inference_sha256=sha(final_model),source_baseline_sha256=CFG['source_sha256'])
    write(run/'complete.json',complete);write(ROOT/'status.json',dict(state='complete',new_updates=500,cumulative_tail_updates=1500,cumulative_joint_updates=1000,edge_perfect=edge['edge_perfect'],joint_perfect=joint_checks[-1]['joint_perfect']))
    print('COMPLETE',edge['edge_perfect'],joint_checks[-1]['joint_perfect'],flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        (ROOT/'resume_failure.txt').write_text(traceback.format_exc());raise
