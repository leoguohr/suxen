"""Explicit 204-to-500 recovery; executed in unchanged pipeline module globals."""
def train():
    pre=json.loads((ROOT/'preflight_complete.json').read_text());assert pre['status']=='passed' and pre['new_optimizer_updates']==0
    assert not (ROOT/'Treatment_last3').exists()
    assert sha(ROOT/'Control_last2/run/latest.pt')==RECOVERY_SHA
    ctx=setup_all();meta,data=load_data(ctx);assert sha(ROOT/'cache/manifest.json')==pre['cache_manifest_sha256']
    for path,digest in json.loads((ROOT/'repro_outputs/loaded_source_manifest.json').read_text()).items():assert sha(path)==digest,path
    parent_edge={r['uid'] for r in ctx['baseline']['meshes'] if r['edge_perfect']};assert len(parent_edge)==72
    completions={}
    for branch,treatment in CFG['branches'].items():
        out=ROOT/branch/'run';resuming=branch=='Control_last2';start_step=204 if resuming else 0
        if not resuming:out.mkdir(parents=True)
        tail,opt,named,mnames=restore_branch(ctx['model'],ctx['full'],ctx['cp'],treatment)
        if resuming:
            tail.load_state_dict(RECOVERY_CP['tail'],strict=True);opt.load_state_dict(copy.deepcopy(RECOVERY_CP['optimizer']))
            same(tail.state_dict(),RECOVERY_CP['tail']);same(opt.state_dict(),RECOVERY_CP['optimizer']);restore_rng(RECOVERY_CP)
            assert [json.loads(x)['update'] for x in (out/'updates.jsonl').read_text().splitlines()]==list(range(1,205))
            assert [json.loads(x)['after_new_updates'] for x in (out/'step_records.jsonl').read_text().splitlines()]==list(range(204))
        params=tuple(p for _,p in named);frozen=fingerprint(ctx['model'],mnames);previous=set(parent_edge);last_update=None;last_eval=None
        if resuming:
            assert frozen==json.loads((out/'restore_verification.json').read_text())['frozen_state']
            previous=set(json.loads((out/'step_records.jsonl').read_text().splitlines()[-1])['edge_success_uids'])
            last_update=json.loads((out/'updates.jsonl').read_text().splitlines()[-1]);last_eval=json.loads((out/'actual-new0200.json').read_text())
        write(out/('resume_verification-0204.json' if resuming else 'restore_verification.json'),dict(full_parent_sha256=CFG['source_full_sha256'],adam_parent_sha256=RECOVERY_SHA if resuming else CFG['source_checkpoint_sha256'],inherited_adam_moments_steps_exact=True,all_four_rng_exact=True,initial_adam_steps=steps(opt),trainable_names=mnames,decoder_layers=16,trainable_tensors=len(named),trainable_parameters=sum(p.numel() for p in params),frozen_state=frozen))
        LIVE.update(branch=branch,tail=tail,opt=opt,step=start_step,model=ctx['model'])
        traces=(out/'step_records.jsonl').open('a' if resuming else 'w',buffering=1);updates=(out/'updates.jsonl').open('a' if resuming else 'w',buffering=1)
        def checkpoint(step,path):
            disk_save(path,dict(tail=cpu_state(tail),optimizer=copy.deepcopy(opt.state_dict()),**rng(),branch=branch,new_updates=step,source_full=CFG['source_full'],source_full_sha256=CFG['source_full_sha256'],source_checkpoint_sha256=CFG['source_checkpoint_sha256'],trainable_names=mnames,per_uid_new_training_participations={u:step for u in ctx['uids']},config=CFG,cache_manifest_sha256=pre['cache_manifest_sha256']))
        for step in range(start_step,501):
            LIVE['step']=step;tail.zero_grad(set_to_none=True);tick=time.monotonic();rows=[]
            for d in data:
                h,ev,fv=ctx['core'].score(tail,d,ctx['sc'],ctx['scales']);row=ctx['head'].metrics(d,ev);row['face_soft4']=float(fv[3].detach());rows.append(row)
                loss=(ev[3]+fv[3])/100;assert bool(T.isfinite(loss))
                if step<500:loss.backward()
                del h,ev,fv,loss;ctx['capture'].clear()
            current={r['uid'] for r in rows if r['perfect']};components=four(rows)
            record=dict(branch=branch,after_new_updates=step,metric_state='after exactly this many completed optimizer updates',edge_success_count=len(current),edge_success_uids=sorted(current),edge_fp=sum(r['fp'] for r in rows),edge_fn=sum(r['fn'] for r in rows),lost_vs_A500=sorted(parent_edge-current),lost_vs_previous_update=sorted(previous-current),gained_vs_A500=sorted(current-parent_edge),gained_vs_previous_update=sorted(current-previous),components=components,original_objective=sum(components.values()),meshes=rows,update_that_produced_this_state=last_update)
            traces.write(json.dumps(record,allow_nan=False)+'\n');previous=current
            if step in CFG['checkpoints'] or step in CFG['checkpoint_extra_steps']:
                cp=out/f'checkpoint-new{step:04d}.pt';checkpoint(step,cp)
                if step in CFG['checkpoints']:last_eval=evaluate_full(ctx,tail,data,out,step,rows,cp)
                assert fingerprint(ctx['model'],mnames)==frozen
            write(ROOT/'status.json',dict(stage='training' if step<500 else 'verifying',branch=branch,new_updates=step,budget=500,edge_success=len(current),edge_fp=record['edge_fp'],edge_fn=record['edge_fn'],lost_vs_A500=record['lost_vs_A500'],last_complete_face_evaluation_step=last_eval['new_updates'] if last_eval else None,last_joint_success=last_eval['summary']['joint_perfect'] if last_eval else None))
            if step==500:break
            before=[p.detach().clone() for p in params]
            gradients={module:norm([p.grad for n,p in named if n.startswith(module+'.')]) for module in tail if any(n.startswith(module+'.') for n,p in named)}
            total=T.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();LIVE['step']=step+1
            assert all(bool(T.isfinite(p).all()) for p in params)
            delta=[p.detach()-q for p,q in zip(params,before)]
            per_group={}
            for module in gradients:
                ids=[i for i,(n,p) in enumerate(named) if n.startswith(module+'.')]
                size=norm([delta[i] for i in ids]);base=norm([before[i] for i in ids]);per_group[module]=dict(delta_norm=size,relative_delta=size/base)
            expected=dict(decoder_tail=[2001+step],edge_head=[2001+step],face_head=[2001+step],decoder14=[1001+step])
            if treatment:expected['decoder13']=[step+1]
            assert steps(opt)==expected
            last_update=dict(branch=branch,update=step+1,mesh_count=100,components_before=components,global_gradient_norm=float(total),clip_coefficient=min(1.,1./(float(total)+1e-6)),gradient_norms=gradients,actual_updates=per_group,actual_delta_norm=norm(delta),actual_relative_delta=norm(delta)/norm(before),adam_steps=expected,seconds=time.monotonic()-tick)
            checkpoint(step+1,out/'latest.pt')
            updates.write(json.dumps(last_update,allow_nan=False)+'\n')
            print('UPDATE',branch,'step='+str(step+1),'edge_before='+str(len(current)),'delta='+str(last_update['actual_delta_norm']),flush=True)
            del before,delta
        traces.close();updates.close();tail.zero_grad(set_to_none=True)
        final=T.load(out/'checkpoint-new0500.pt',map_location='cpu',weights_only=False);same(tail.state_dict(),final['tail']);same(opt.state_dict(),final['optimizer']);assert fingerprint(ctx['model'],mnames)==frozen
        full_path=out/'model-new0500-inference.pt';disk_save(full_path,dict(model={n:v.detach().cpu() if T.is_tensor(v) else copy.deepcopy(v) for n,v in ctx['model'].state_dict().items()},args=ctx['full']['args'],inference_only=True,branch=branch,new_updates=500,source_sha256=CFG['source_full_sha256'],optimizer_checkpoint=str(out/'checkpoint-new0500.pt'),optimizer_checkpoint_sha256=sha(out/'checkpoint-new0500.pt')))
        completions[branch]=dict(new_updates=500,stopped_at_budget=True,full_model=str(full_path),full_model_sha256=sha(full_path),checkpoint=str(out/'checkpoint-new0500.pt'),checkpoint_sha256=sha(out/'checkpoint-new0500.pt'),final=last_eval,frozen_state_unchanged=True,adam_steps=steps(opt))
        write(ROOT/branch/'complete.json',completions[branch]);print('BRANCH_COMPLETE',branch,flush=True);del opt,final
    verify_assets();write(ROOT/'complete.json',dict(state='complete',per_branch_updates=500,total_experiment_updates=1000,branches=completions,source_assets_unchanged=True));write(ROOT/'status.json',dict(state='complete',per_branch_updates=500,total_experiment_updates=1000,final_joint={k:v['final']['summary']['joint_perfect'] for k,v in completions.items()}));print('COMPLETE BOTH BRANCHES; BUDGET CLOSED',flush=True)
