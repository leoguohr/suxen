"""Fixed-budget A500 last2/last3 comparison, with every-update Edge evidence."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['RIGORPILOT_LESSONS']='0'
import argparse,copy,csv,fcntl,hashlib,importlib.util,json,math,random,subprocess,sys,time,traceback,types
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'decoder_last2_success_edge_half_pair_20260921'
SOURCE=ROOT.parent/'decoder_last2_joint_fixed100_20260920'
CFG=json.loads((ROOT/'config.json').read_text())
PARTS=['success_edge','success_face','failed_edge','failed_face']
LIVE={}

def same(a,b):
    if T.is_tensor(a):assert T.is_tensor(b) and a.dtype==b.dtype and T.equal(a.cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)

def rng():
    return dict(rng=T.get_rng_state().clone(),cuda_rng=[x.clone() for x in T.cuda.get_rng_state_all()],python_rng=random.getstate(),numpy_rng=np.random.get_state())

def restore_rng(cp):
    T.set_rng_state(cp['rng'].cpu());T.cuda.set_rng_state_all([x.cpu() for x in cp['cuda_rng']]);random.setstate(cp['python_rng']);np.random.set_state(cp['numpy_rng'])
    for k,v in rng().items():same(v,cp[k])

def disk_save(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');T.save(obj,temp);os.replace(temp,path)

def cpu_state(tail):return {k:v.detach().cpu().clone() for k,v in tail.state_dict().items()}

def four(rows):
    out={k:0. for k in PARTS}
    for row in rows:
        prefix='success' if row['uid'] in SUCCESS else 'failed'
        for name in ['edge','face']:out[prefix+'_'+name]+=row[name+'_soft4']/100
    return out

def steps(opt):return {g['name']:sorted({int(opt.state[p]['step']) for p in g['params']}) if all(p in opt.state and 'step' in opt.state[p] for p in g['params']) else [0] for g in opt.param_groups}

def norm(xs):return math.sqrt(sum(float(x.detach().double().square().sum()) for x in xs))

def fingerprint(model,trainable):
    state=model.state_dict();names=set(trainable)
    return {'tensors':tensor_hash((k,v) for k,v in state.items() if T.is_tensor(v) and k not in names),
            'metadata':json.dumps({k:v for k,v in state.items() if not T.is_tensor(v)},sort_keys=True)}

def restore_branch(model,full,cp,treatment):
    model.load_state_dict(full['model'],strict=True);model.requires_grad_(False);a=model.autoencoder
    assert len(a.decoder_blocks)==16
    tail=T.nn.ModuleDict(dict(block=a.decoder_blocks[15],final_norm=a.decoder_output_norm,head=a.edge_embedding,face=a.face_embedding,penultimate=a.decoder_blocks[14],third=a.decoder_blocks[13]))
    for name,module in tail.items():module.requires_grad_(name!='third' or treatment)
    for k,v in cp['tail'].items():same(tail.state_dict()[k],v)
    groups=[dict(params=list(tail['block'].parameters())+list(tail['final_norm'].parameters()),name='decoder_tail'),dict(params=list(tail['head'].parameters()),name='edge_head'),dict(params=list(tail['face'].parameters()),name='face_head'),dict(params=list(tail['penultimate'].parameters()),name='decoder14')]
    opt=T.optim.Adam(groups);opt.load_state_dict(copy.deepcopy(cp['optimizer']));same(opt.state_dict(),cp['optimizer'])
    if treatment:opt.add_param_group(dict(params=list(tail['third'].parameters()),name='decoder13',lr=CFG['lr']['decoder13'],betas=(.9,.999),eps=1e-8,weight_decay=0,amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None))
    for g in opt.param_groups:
        assert g['lr']==CFG['lr'][g['name']] and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and g['weight_decay']==0
    expected={'decoder_tail':[2000],'edge_head':[2000],'face_head':[2000],'decoder14':[1000]}
    if treatment:expected['decoder13']=[0]
    assert steps(opt)==expected
    named=[(n,p) for n,p in tail.named_parameters() if p.requires_grad]
    mnames=[n for n,p in model.named_parameters() if p.requires_grad]
    assert len(named)==(24 if treatment else 18)
    assert {id(p) for _,p in named}=={id(p) for p in model.parameters() if p.requires_grad}
    restore_rng(cp);tail.zero_grad(set_to_none=True)
    return tail,opt,named,mnames

def verify_assets():
    refs={CFG['source_full']:CFG['source_full_sha256'],CFG['source_checkpoint']:CFG['source_checkpoint_sha256'],CFG['preserve74']:CFG['preserve74_sha256']}
    prior=T.load(SOURCE/'run/checkpoint-new0500-tail1500.pt',map_location='cpu',mmap=True,weights_only=False)['config']
    refs.update({prior['preserve_joint72']:prior['preserve_joint72_sha256'],str(SOURCE/'run/model-last2-joint1000-tail1500-block14new500-inference.pt'):'82cf33fbc0151d72332218c5e447f7bf84e8934947e4c70148205c6a308db884'})
    for p,digest in refs.items():assert sha(p)==digest,p
    pm=json.loads((SOURCE/'source_manifest.json').read_text())
    for row in csv.DictReader((ROOT/'overfit100_manifest.csv').open()):
        for kind in ['mesh','topology']:assert sha(row[kind+'_path'])==row[kind+'_sha256']
    for uid,digest in pm['augmented_pool_sha256'].items():assert sha(SOURCE/'augmented_pools'/f'{uid}_pool.npz')==digest
    return refs,pm

def setup_all():
    global T,np,b,c,sha,write,tensor_hash,SUCCESS
    for name in ['READY.json','overfit100_manifest.csv','data_manifest.csv','selection.json','pool_provenance.json','pools','construction_args.json']:
        target=(OLD/name).resolve();dest=ROOT/name
        if not dest.exists():dest.symlink_to(target)
        assert dest.resolve()==target
    import runtime
    T,np,b,c,sha,write,tensor_hash=runtime.T,runtime.np,runtime.b,runtime.c,runtime.sha,runtime.write,runtime.tensor_hash
    import core,head_core,evaluate
    evaluate.FACE_SECONDS=float('inf')
    refs,pm=verify_assets()
    full=T.load(CFG['source_full'],map_location='cpu',mmap=True,weights_only=False);cp=T.load(CFG['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    assert cp['new_updates']==500 and cp['branch']=='A_control' and cp['success_edge_coefficient']==1
    model,unused,_,uids,_,forward,capture,args=runtime.setup();assert unused is None
    T.use_deterministic_algorithms(True);model.load_state_dict(full['model'],strict=True);model.requires_grad_(False)
    part=json.loads((ROOT/'partition.json').read_text());SUCCESS=set(part['success71']);assert len(SUCCESS)==71 and len(part['failed29'])==29 and 'nexus_2k_000520' in part['failed29']
    scpath=ROOT/'effective_code/effective_loss_and_scoring.py';spec=importlib.util.spec_from_file_location('sc',scpath);sc=importlib.util.module_from_spec(spec);spec.loader.exec_module(sc)
    scales=model.scoring_contract();same(scales,{k:v for k,v in full['model']['_extra_state'].items() if k!='format_version'})
    baseline=json.loads((OLD/'A_control/run/actual-new0500.json').read_text());assert [r['uid'] for r in baseline['meshes']]==uids
    assert baseline['summary']['all']['joint_perfect']==72
    return dict(model=model,full=full,cp=cp,uids=uids,forward=forward,capture=capture,sc=sc,scales=scales,baseline=baseline,refs=refs,pm=pm,core=core,head=head_core,evaluate=evaluate,args=args)

def load_data(ctx):
    meta=json.loads((ROOT/'cache/manifest.json').read_text());data=[]
    for uid,record in zip(ctx['uids'],meta['meshes']):
        assert uid==record['uid'];p=ROOT/'cache'/f'{uid}.npz';assert sha(p)==record['sha256']
        with np.load(p) as q:x=T.from_numpy(q['block13_input'].copy()).cuda()
        p=SOURCE/'augmented_pools'/f'{uid}_pool.npz';assert sha(p)==ctx['pm']['augmented_pool_sha256'][uid]
        with np.load(p) as q:pool={k:q[k].copy() for k in ['vertices','edges','positive','mixed']}
        n=len(x);pairs=T.triu_indices(n,n,1,device='cuda').T
        edges=np.unique(np.sort(pool['edges'].T,axis=1),axis=0);keys=T.as_tensor(edges[:,0]*n+edges[:,1],device='cuda');q=pairs[:,0]*n+pairs[:,1];at=T.searchsorted(keys,q);labels=(at<len(keys))&(keys[at.clamp_max(len(keys)-1)]==q)
        tris=T.as_tensor(np.concatenate([pool['positive'],pool['mixed']]),device='cuda');yl=T.cat([T.ones(len(pool['positive']),device='cuda'),T.zeros(len(pool['mixed']),device='cuda')])
        data.append(dict(uid=uid,x=x,pairs=pairs,labels=labels,face_tris=tris,face_labels=yl,pool=pool))
    assert len(data)==100 and sum(len(d['pairs']) for d in data)==84669234 and sum(len(d['face_tris']) for d in data)==750215
    return meta,data

def original_metrics(ctx,d,rows):
    e=rows[2][0];p=d['pairs'];s=ctx['sc'].first_order_interval(e[p[:,0]],e[p[:,1]])*ctx['scales']['edge_logit_scale'];ns,ms=ctx['sc'].soft4_sums(s,d['labels']);le=(ns/(ms+1e-8)).mean()
    import face_core
    lf,fl=face_core.objective(rows[3][0],d['face_tris'],d['face_labels'],ctx['scales'])
    return le,lf,s,fl

def preflight():
    assert not (ROOT/'preflight_complete.json').exists() and not (ROOT/'cache').exists(),'Do not repeat completed/partial preflight blindly'
    ctx=setup_all();model=ctx['model'];tail,opt,named,mnames=restore_branch(model,ctx['full'],ctx['cp'],False)
    before_rng=rng();initial_hash=tensor_hash((n,t) for n,t in model.state_dict().items() if T.is_tensor(t))
    cache=ROOT/'cache';cache.mkdir();grab={};hook=model.autoencoder.decoder_blocks[13].norm.register_forward_pre_hook(lambda module,x:grab.update(x=x[0].detach()))
    records=[]
    # math00's nonreentrant checkpoint context requires the original enabled
    # grad mode, even when upstream parameters are frozen. No backward/step here.
    with T.enable_grad():
        for i,uid in enumerate(ctx['uids']):
            real=ctx['forward'](uid);p=cache/f'{uid}.npz';np.savez(p,block13_input=grab['x'].cpu().numpy())
            records.append(dict(uid=uid,vertices=len(grab['x']),sha256=sha(p)));ctx['capture'].clear();grab.clear();del real
            if (i+1)%25==0:print('CACHE13',i+1,'/100',flush=True)
    hook.remove();same(before_rng,rng())
    write(cache/'manifest.json',dict(source_full_sha256=CFG['source_full_sha256'],boundary='input to decoder13; before norm/attention',meshes=records))
    meta,data=load_data(ctx);results=[]
    for branch,treatment in CFG['branches'].items():
        tail,opt,named,mnames=restore_branch(model,ctx['full'],ctx['cp'],treatment);parameters=tuple(p for _,p in named);start_rng=rng();details=[]
        for i,d in enumerate(data):
            real=ctx['forward'](d['uid']);realhidden=ctx['capture']['hidden'].clone();le,lf,elog,flog=original_metrics(ctx,d,real)
            fullgrad=T.autograd.grad((le+lf)/100,parameters)
            h,ev,fv=ctx['core'].score(tail,d,ctx['sc'],ctx['scales']);cachedgrad=T.autograd.grad((ev[3]+fv[3])/100,parameters)
            assert T.equal(h.detach(),realhidden) and T.equal(ev[1],real[2][0]) and T.equal(fv[1],real[3][0]),d['uid']
            assert T.equal(le,ev[3]) and T.equal(lf,fv[3]) and T.equal(elog,ev[2]) and T.equal(flog,fv[2]),d['uid']
            assert all(T.equal(a,bv) for a,bv in zip(fullgrad,cachedgrad)),d['uid']
            expected=ctx['baseline']['meshes'][i];m=ctx['head'].metrics(d,ev)
            assert all(m[k]==expected['edge'][k] for k in ['tp','fp','fn','tn']) and m['edge_soft4']==expected['edge_soft4'] and float(fv[3])==expected['face_soft4']
            if not treatment:
                p=ROOT/'repro_outputs/preflight_predictions'/f"{d['uid']}.npz";p.parent.mkdir(exist_ok=True)
                actual=ctx['evaluate'].evaluate_mesh(real,d['pool'],ctx['scales'],prediction_path=p)
                for k in ['edge','face','gt_face_candidates','missing_gt_face_candidates','joint_perfect','margins']:assert actual[k]==expected[k],(d['uid'],k)
                with np.load(p) as new,np.load(OLD/'A_control/run/predictions-new0500'/p.name) as old:
                    assert set(new.files)==set(old.files) and all(np.array_equal(new[k],old[k]) for k in new.files),d['uid']
            details.append(dict(uid=d['uid'],full_cached_hidden_edge_face_logits_loss_gradient_bitwise=True,trainable_tensors=len(parameters)))
            del real,realhidden,le,lf,elog,flog,h,ev,fv,fullgrad,cachedgrad;ctx['capture'].clear()
            if (i+1)%25==0:print('PREFLIGHT',branch,i+1,'/100',flush=True)
        same(start_rng,rng());assert initial_hash==tensor_hash((n,t) for n,t in model.state_dict().items() if T.is_tensor(t))
        results.append(dict(branch=branch,meshes=details,adam_steps=steps(opt),trainable_names=mnames,trainable_parameters=sum(p.numel() for _,p in named),rng_unchanged=True))
        del opt
    modules={};pending=list(sys.modules.values());seen=set()
    while pending:
        mod=pending.pop()
        if not isinstance(mod,types.ModuleType) or id(mod) in seen:continue
        seen.add(id(mod))
        fn=getattr(mod,'__file__',None)
        if fn and str(fn).startswith('/guohaoran/nexus_fast_track') and Path(fn).suffix=='.py' and Path(fn).is_file():
            modules[str(Path(fn).resolve())]=sha(fn)
            pending.extend(x for x in vars(mod).values() if isinstance(x,types.ModuleType))
    # Snapshot the actually imported project dependency closure without changing sources.
    for path,digest in modules.items():
        dest=ROOT/'repro_outputs/loaded_sources'/Path(path).relative_to('/guohaoran/nexus_fast_track');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(Path(path).read_bytes());assert sha(dest)==digest
    write(ROOT/'repro_outputs/loaded_source_manifest.json',modules)
    verify_assets();write(ROOT/'preflight_complete.json',dict(status='passed',new_optimizer_updates=0,parent_actual100_arrays_bitwise=True,both_branches_all100_full_cached_forward_gradient_bitwise=True,branches=results,cache_manifest_sha256=sha(cache/'manifest.json'),source_files_unchanged=True))
    print('PREFLIGHT COMPLETE optimizer_updates=0',flush=True)

def evaluate_full(ctx,tail,data,out,step,rows,checkpoint):
    before=rng();actual=[];preds=out/f'predictions-new{step:04d}';preds.mkdir()
    for i,(d,expected) in enumerate(zip(data,rows)):
        real=ctx['forward'](d['uid']);h0=ctx['capture']['hidden'].clone();h,ev,fv=ctx['core'].score(tail,d,ctx['sc'],ctx['scales'])
        assert T.equal(h.detach(),h0) and T.equal(ev[1],real[2][0]) and T.equal(fv[1],real[3][0])
        p=preds/f"{d['uid']}.npz";a=ctx['evaluate'].evaluate_mesh(real,d['pool'],ctx['scales'],prediction_path=p)
        assert a['face']['complete'] and all(a['edge'][k]==expected[k] for k in ['tp','fp','fn','tn'])
        a.update(uid=d['uid'],vertices=len(d['x']),edge_soft4=expected['edge_soft4'],face_soft4=expected['face_soft4'],prediction_sha256=sha(p),prediction_path=str(p.relative_to(ROOT)))
        actual.append(a);del real,h0,h,ev,fv;ctx['capture'].clear()
        if (i+1)%25==0:print('ACTUAL',out.parent.name,step,i+1,'/100',flush=True)
    same(before,rng());perfect=sorted(a['uid'] for a in actual if a['joint_perfect']);parent=set(ctx['baseline']['joint_perfect_uids'])
    summary={kind:{key:sum(a[kind][key] for a in actual) for key in ['tp','fp','fn','tn']} for kind in ['edge','face']}
    summary.update(edge_perfect=sum(a['edge_perfect'] for a in actual),joint_perfect=len(perfect),meshes=100)
    result=dict(branch=out.parent.name,new_updates=step,checkpoint_sha256=sha(checkpoint),components=four(rows),summary=summary,meshes=actual,joint_perfect_uids=perfect,lost_A500=sorted(parent-set(perfect)),new_over_A500=sorted(set(perfect)-parent),verification=dict(all100_real_vs_cached_bitwise=True,all100_face_enumeration_complete=True,evaluation_rng_unchanged=True),progress=dict(exceeds_branch72=len(perfect)>72,exceeds_historical74=len(perfect)>74,main_goal_100=len(perfect)==100))
    write(out/f'actual-new{step:04d}.json',result);print('EVAL_COMPLETE',out.parent.name,step,json.dumps(summary),flush=True)
    return result

def train():
    pre=json.loads((ROOT/'preflight_complete.json').read_text());assert pre['status']=='passed' and pre['new_optimizer_updates']==0
    assert not any((ROOT/br).exists() for br in CFG['branches']),'Existing state: inspect; no implicit restart or resume'
    ctx=setup_all();meta,data=load_data(ctx);assert sha(ROOT/'cache/manifest.json')==pre['cache_manifest_sha256']
    for path,digest in json.loads((ROOT/'repro_outputs/loaded_source_manifest.json').read_text()).items():assert sha(path)==digest,path
    parent_edge={r['uid'] for r in ctx['baseline']['meshes'] if r['edge_perfect']};assert len(parent_edge)==72
    completions={}
    for branch,treatment in CFG['branches'].items():
        out=ROOT/branch/'run';out.mkdir(parents=True);tail,opt,named,mnames=restore_branch(ctx['model'],ctx['full'],ctx['cp'],treatment)
        params=tuple(p for _,p in named);frozen=fingerprint(ctx['model'],mnames);previous=set(parent_edge);last_update=None;last_eval=None
        write(out/'restore_verification.json',dict(full_parent_sha256=CFG['source_full_sha256'],adam_parent_sha256=CFG['source_checkpoint_sha256'],inherited_adam_moments_steps_exact=True,all_four_rng_exact=True,initial_adam_steps=steps(opt),trainable_names=mnames,decoder_layers=16,trainable_tensors=len(named),trainable_parameters=sum(p.numel() for p in params),frozen_state=frozen))
        LIVE.update(branch=branch,tail=tail,opt=opt,step=0,model=ctx['model'])
        traces=(out/'step_records.jsonl').open('w',buffering=1);updates=(out/'updates.jsonl').open('w',buffering=1)
        def checkpoint(step,path):
            disk_save(path,dict(tail=cpu_state(tail),optimizer=copy.deepcopy(opt.state_dict()),**rng(),branch=branch,new_updates=step,source_full=CFG['source_full'],source_full_sha256=CFG['source_full_sha256'],source_checkpoint_sha256=CFG['source_checkpoint_sha256'],trainable_names=mnames,per_uid_new_training_participations={u:step for u in ctx['uids']},config=CFG,cache_manifest_sha256=pre['cache_manifest_sha256']))
        for step in range(501):
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

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','train']);args=parser.parse_args()
    lock=(ROOT/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        {'preflight':preflight,'train':train}[args.stage]()
    except BaseException:
        (ROOT/f'{args.stage}_failure.txt').write_text(traceback.format_exc())
        if LIVE:
            disk_save(ROOT/LIVE['branch']/'failure-state.pt',dict(tail=cpu_state(LIVE['tail']),optimizer=LIVE['opt'].state_dict(),**rng(),branch=LIVE['branch'],new_updates=LIVE['step']))
        raise
