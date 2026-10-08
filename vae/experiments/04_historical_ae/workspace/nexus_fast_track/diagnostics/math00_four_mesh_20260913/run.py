"""Four complete meshes from Hold: fixed beta1e-4, 1000-update observation budget."""
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--preflight",action="store_true");args=parser.parse_args()
import backend00 as b
import fcntl
import json
import time
import inspect
import hashlib
from pathlib import Path

torch=b.torch; np=b.np; c=b.c; ROOT=b.ROOT
PREVIOUS=ROOT.parent/'math00_kl_hold_20260912'
SELECTION=json.loads((ROOT/'selection.json').read_text())
c.UIDS=SELECTION['uids'];c.m.UIDS=c.UIDS;c.SEEDS=[970000,970001,970002,970003]
M=len(c.UIDS);assert M==4
STEPS=1000
START=PREVIOUS/'checkpoint-update0200.pt'
CHECKS={0,1,10,20,50,100,200,400,600,800,1000}
NOISE_CHECKS={0,200,400,600,800,1000}
SEEDS=json.loads((ROOT/'evaluation_seeds.json').read_text())
assert SEEDS['training']==c.SEEDS
assert len(SEEDS['evaluation'])==50
assert not set(sum(SEEDS['evaluation'],[])) & set(c.SEEDS)
lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'updates.jsonl').exists()
torch.set_num_threads(1);torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
c.m.probe.UIDS=c.UIDS
cp,model,batch=c.m.probe.setup_model(START)
assert cp['completed_updates']==200
assert c.m.probe.digest(ROOT/'backend00.py')==cp['diagnostic_manifest']['backend_sha']
for p,sha in cp['diagnostic_manifest']['source_sha256'].items():assert c.m.probe.digest(p)==sha
start_sha=c.m.probe.digest(START)
baseline=dict(name='math00_kl_hold_step200',checkpoint=str(START),sha256=start_sha,adam_steps=dict(encoder_decoder_heads=800,logvar=600),backend_sha=cp['diagnostic_manifest']['backend_sha'],verification=str(PREVIOUS/'complete.json'))
c.write(ROOT/'math00_kl_hold_step200.json',baseline)
runtime=c.m.install_clamp(model,-20.)
assert runtime.count('noise = torch.randn_like(sample_mu)')==1
runtime=runtime.replace('noise = torch.randn_like(sample_mu)', 'noise = torch.randn(sample_mu.shape, dtype=sample_mu.dtype, device=sample_mu.device, generator=autoencoder.diagnostic_train_rng)\n                autoencoder.diagnostic_train_draws += 1')
exec(compile(runtime,str(ROOT/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__)
(ROOT/'sampling_forward.py').write_text(runtime)
train_rng=torch.Generator(device='cuda');train_rng.set_state(cp['training_rng_state'])
DRAW_OFFSET=cp['training_noise_draws'];assert DRAW_OFFSET==1200
model.autoencoder.diagnostic_train_rng=train_rng
model.autoencoder.diagnostic_train_draws=DRAW_OFFSET

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

# The metric/scoring code is unchanged; monitoring sets omit only bulky NPZ serialization.
capture_source=inspect.getsource(c.capture)
assert capture_source.count('baseline=None):')==1
assert capture_source.count("np.savez_compressed(out/f'{uid}.npz',**table);tables.append(table)")==1
capture_source=capture_source.replace('baseline=None):','baseline=None,save_npz=True):').replace("  np.savez_compressed(out/f'{uid}.npz',**table);tables.append(table)","  if save_npz:np.savez_compressed(out/f'{uid}.npz',**table)\n  tables.append(table)")
ns=dict(c.capture.__wrapped__.__globals__);exec(capture_source,ns);capture=ns['capture']
(ROOT/'capture_metrics.py').write_text(capture_source)
b.install_deterministic(model)
for section,blocks in [('encoder',model.autoencoder.encoder_blocks),('decoder',model.autoencoder.decoder_blocks)]:
    for i,block in enumerate(blocks):
        for name,module in block.named_modules():
            if isinstance(module,torch.nn.MultiheadAttention):b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
assert len(b.ATTENTION_NAMES)==28
model.requires_grad_(True)
named=dict(model.autoencoder.named_parameters())
groups={name:{n:named[n] for n in names} for name,names in cp['diagnostic_manifest']['groups'].items()}
assert list(groups)==['encoder_mu','decoder_body','edge_head','face_head','logvar']
optimizer=torch.optim.Adam([dict(params=list(group.values()),lr=cp['diagnostic_manifest']['lr'][name],name=name) for name,group in groups.items()],weight_decay=0.)
optimizer.load_state_dict(cp['optimizer'])
for group,saved_group in zip(optimizer.param_groups,cp['optimizer']['param_groups']):
    assert {k:v for k,v in group.items() if k!='params'}=={k:v for k,v in saved_group.items() if k!='params'}
    assert group['lr']==(1e-8 if group['name']=='encoder_mu' else 1e-4 if group['name']=='logvar' else 1e-7)
    for p,index in zip(group['params'],saved_group['params']):
        for k,value in cp['optimizer']['state'][index].items():assert torch.equal(optimizer.state[p][k].cpu(),value)
        assert float(optimizer.state[p]['step'])==(600 if group['name']=='logvar' else 800)
active=[p for group in groups.values() for p in group.values()]
assert len(active)==len({id(p) for p in active})==len(list(model.parameters()))
assert {id(p) for p in active}=={id(p) for p in model.parameters()}
assert all(p.requires_grad and p.dtype==torch.float32 for p in active)
assert all(torch.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
initial={n:p.detach().clone() for n,p in named.items()}
scales=model.scoring_contract();c.PAIR_CHUNK=cp['args']['pair_chunk_size']
data=[];pool_hashes={}
for i,uid in enumerate(c.UIDS):
    path=ROOT/(uid+'_pool.npz');pool_hashes[uid]=c.m.probe.digest(path)
    d=np.load(path);data.append(d)
    assert np.array_equal(d['vertices'],batch.vertices[i,:len(d['vertices'])].cpu().numpy())
    assert np.array_equal(d['positive'],batch.face_set[i].cpu().numpy())
manifest=dict(baseline=baseline,backend='unchanged math00; deterministic Graph forward/backward, FP32 MATH on all 28 attention modules; TF32/autocast off',groups={n:list(g) for n,g in groups.items()},lr={g['name']:g['lr'] for g in optimizer.param_groups},optimizer='restore all five Adam groups from Hold step200 exactly',existing_adam_exact=True,logvar_adam_initial='restored Adam step600',betas=optimizer.defaults['betas'],eps=optimizer.defaults['eps'],weight_decay=0.,clip=1.,kl='equal per-mesh diagonal Gaussian KL, scaled by beta_schedule',mode='fresh epsilon each optimizer step',training_rng_seed=SEEDS['training_rng_seed'],fixed_diagnostic_seeds=c.SEEDS,evaluation_seeds=SEEDS['evaluation'],final_unseen_seeds=SEEDS['final_unseen'],noise_check_steps=sorted(NOISE_CHECKS),soft4='fully-diff Edge+Face; equal mean across meshes; tau1 epsilon1e-8; original candidate/reduction',logvar_clamp=[-20.,10.],logvar_reinitialized=False,mu_loss_weight=0.,updates=STEPS,checks=sorted(CHECKS),threshold=0.,source_sha256=cp['diagnostic_manifest']['source_sha256'],pool_sha256=pool_hashes,script_sha=c.m.probe.digest(Path(__file__)),backend_sha=c.m.probe.digest(ROOT/'backend00.py'),gpu=0,device=torch.cuda.get_device_name(),torch_version=torch.__version__)
manifest.update(selected_uids=c.UIDS,mesh_count=M,budget_purpose='1000 updates to observe learning new meshes, not a capacity-failure deadline',beta_target=1e-4,beta_initial=1e-4,beta_schedule='constant 1e-4',training_rng_initial_sha256=tensor_hash(train_rng.get_state()),training_draw_offset=DRAW_OFFSET,kl_definition='equal mesh mean of 0.5 mean_vc(mu^2+exp(logvar)-1-logvar)',kl_gradient_checks=[])
assert pool_hashes==SELECTION['pool_sha256']
for uid in c.UIDS[:2]:assert pool_hashes[uid]==cp['diagnostic_manifest']['pool_sha256'][uid]
assert [int(x.sum()) for x in batch.vertex_mask]==[r['vertices'] for r in SELECTION['records']]
manifest['monitor_npz_serialization']=False
manifest['rng_checkpoint_scope']='attention core only; noise generated once per mesh before decoder; assert generator unchanged in backward and evaluation'
c.write(ROOT/'manifest.json',manifest)

def kl_parts(rows):
    per=[dict(mu=.5*mu.float().square().mean(),sigma=.5*(lv.float().exp()-1-lv.float()).mean(),total=.5*(mu.float().square()+lv.float().exp()-1-lv.float()).mean()) for mu,lv in zip(rows[0],rows[1])]
    kmu=torch.stack([x['mu'] for x in per]).mean();ksigma=torch.stack([x['sigma'] for x in per]).mean()
    kl=torch.stack([x['total'] for x in per]).mean()
    info=dict(total=float(kl.detach()),mu=float(kmu.detach()),sigma=float(ksigma.detach()),per_mesh=[{k:float(v.detach()) for k,v in p.items()} for p in per])
    return kl,kmu,ksigma,info

def beta_at(step):return 1e-4

def forward(mode,seeds=None):
    rows,stats,zs=c.m.forward(model,batch,mode,seeds)
    for mu,z,stat in zip(rows[0],zs,stats):
        perturb=z.detach()-mu.detach()
        stat['actual_noise_rms']=float(perturb.double().square().mean().sqrt())
        stat['actual_noise_max_abs']=float(perturb.abs().max())
        stat['actual_noise_nonzero_elements']=int((perturb!=0).sum())
        if mode=='sample':assert stat['actual_noise_nonzero_elements']>0
    return rows,stats

def compact(rec):return [{k:r[k] for k in ['uid','edge','face','gt_faces_missing_from_edge_candidates','margins']} for r in rec]
def perfect(rec):return all(r[k]['fp']==r[k]['fn']==0 for r in rec for k in ['edge','face'])

def evaluate(rows,stats,step,mode,suffix='',save_npz=True):
    detached=tuple(tuple(x.detach() for x in row) for row in rows)
    value,parts,_=b.full_objective(detached,data,scales)
    _,_,_,kl_info=kl_parts(detached)
    out=ROOT/f'{mode}_step{step:04d}{suffix}'
    rec,tables=capture(detached,data,scales,out,save_npz=save_npz)
    result=dict(step=step,mode=mode,loss=float(value),kl=kl_info,parts=parts,posterior=stats,rec=compact(rec),all_meshes_perfect=perfect(rec),old_meshes_perfect=perfect(rec[:2]),new_meshes_perfect=perfect(rec[2:]))
    c.write(out.with_suffix('.json'),result)
    return result,tables

def detached_forward(mode,seeds=None):
    # Preserve the already verified grad-mode dispatch; release the graph before scoring.
    rng_before=train_rng.get_state().clone()
    rows,stats=forward(mode,seeds)
    assert torch.equal(rng_before,train_rng.get_state())
    return tuple(tuple(x.detach() for x in row) for row in rows),stats

def noise_evaluation(step,final_unseen=False):
    tag='final_unseen' if final_unseen else 'independent'
    seed_pairs=SEEDS['final_unseen'] if final_unseen else SEEDS['evaluation']
    values=[];rng_before=train_rng.get_state().clone()
    with (ROOT/f'{tag}_step{step:04d}.jsonl').open('w',buffering=1) as out:
        for j,seeds in enumerate(seed_pairs):
            rows,stats=detached_forward('sample',seeds)
            eps_hashes=[tensor_hash(x) for x in model.autoencoder._diagnostic_eps]
            assert not set(eps_hashes)&TRAIN_NOISE_HASHES
            result,_=evaluate(rows,stats,step,tag,f'_{j:02d}',save_npz=False)
            result.update(pair=j,seeds=seeds,epsilon_sha256=eps_hashes)
            values.append(result);out.write(json.dumps(result,allow_nan=False)+'\n')
            if (j+1)%10==0:print('NOISE',tag,step,j+1,'perfect',sum(v['all_meshes_perfect'] for v in values),flush=True)
            del rows
    assert torch.equal(rng_before,train_rng.get_state())
    result=dict(step=step,total=50,simultaneous_perfect=sum(v['all_meshes_perfect'] for v in values),seeds=seed_pairs,mean_soft4=float(np.mean([v['loss'] for v in values])),std_soft4=float(np.std([v['loss'] for v in values])),mean_parts=[{k:float(np.mean([v['parts'][i][k] for v in values])) for k in ['edge','face']} for i in range(M)],mean_kl={k:float(np.mean([v['kl'][k] for v in values])) for k in ['total','mu','sigma']},rng_unchanged=True)
    c.write(ROOT/f'{tag}_summary_step{step:04d}.json',result)
    return result

def movement(group,before):
    dsq=torch.zeros((),device='cuda',dtype=torch.float64);tsq=dsq.clone();changed=torch.zeros((),device='cuda',dtype=torch.int64)
    for n,p in group.items():
        delta=p.detach().double()-before[n].double()
        dsq+=delta.square().sum();tsq+=before[n].double().square().sum();changed+=(delta!=0).sum()
    dn=float(dsq.sqrt());tn=float(tsq.sqrt())
    return dict(delta_l2=dn,theta_l2=tn,relative_l2=dn/(tn+1e-30),changed_elements=int(changed))

def save_checkpoint(step):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=step,parent_checkpoint_sha=start_sha,diagnostic_manifest=manifest,args=cp['args'],training_rng_state=train_rng.get_state(),training_noise_draws=model.autoencoder.diagnostic_train_draws),ROOT/f'checkpoint-update{step:04d}.pt')

if args.preflight:
    rng=train_rng.get_state().clone();torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    rows,stats=forward('sample',None)
    rec,parts,_=b.full_objective(rows,data,scales);kl,kmu,ksigma,ki=kl_parts(rows)
    assert abs(float(rec.detach())-sum(x['edge']+x['face'] for x in parts)/M)<1e-5
    assert abs(ki['total']-sum(x['total'] for x in ki['per_mesh'])/M)<1e-5
    value=rec+1e-4*kl;value.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
    memory=dict(peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,total_GiB=torch.cuda.get_device_properties(0).total_memory/2**30)
    c.write(ROOT/'memory_preflight.json',dict(meshes=M,uids=c.UIDS,parts=parts,kl=ki,posterior=stats,reconstruction=float(rec.detach()),memory=memory,all_gradients_finite=True,no_optimizer_update=True,all_meshes_in_single_packed_forward=True))
    train_rng.set_state(rng);model.autoencoder.diagnostic_train_draws=DRAW_OFFSET;optimizer.zero_grad(set_to_none=True)
    assert all(torch.equal(p,initial[n]) for n,p in named.items())
    print('MEMORY_PREFLIGHT',json.dumps(memory),flush=True)
    raise SystemExit(0)

TRAIN_NOISE_HASHES=set()
rows,stats=detached_forward('mu')
mu0,tables=evaluate(rows,stats,0,'mu')
old_mu=json.loads((PREVIOUS/'mu_step0200.json').read_text())
old_mu_baseline=dict(previous_perfect=old_mu['four_way_perfect'] if 'four_way_perfect' in old_mu else old_mu['all_meshes_perfect'],current_perfect=mu0['old_meshes_perfect'],previous_parts=old_mu['parts'],current_parts=mu0['parts'][:2])
previous_mu=[x.detach().clone() for x in rows[0]]
del rows,tables
rows,stats=detached_forward('sample',c.SEEDS)
sample0,tables=evaluate(rows,stats,0,'sample')
del rows,tables
noise_checks={0:noise_evaluation(0)}
new_noise=[json.loads(s) for s in (ROOT/'independent_step0000.jsonl').read_text().splitlines()]
EVAL_NOISE_HASHES={v for row in new_noise for v in row['epsilon_sha256']}
assert all(torch.equal(p,initial[n]) for n,p in named.items())
assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET
c.write(ROOT/'step0_acceptance.json',dict(mu=mu0,fixed_epsilon=sample0,independent=noise_checks[0],old_mu_comparison=old_mu_baseline,new_mesh_errors_are_task_baseline=True,no_updates_yet=True))
save_checkpoint(0)
(ROOT/'training_noise').mkdir(exist_ok=True)
print('BEGIN_TRAINING fresh RNG',SEEDS['training_rng_seed'],flush=True)
started=time.monotonic();evals=[dict(step=0,mu=mu0,sample=sample0)];update_history=[]
with (ROOT/'updates.jsonl').open('w',buffering=1) as log:
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        rng_before=train_rng.get_state().clone()
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+M*(step-1)
        rows,stats=forward('sample',None)
        rng_after=train_rng.get_state().clone()
        assert not torch.equal(rng_before,rng_after)
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+M*step
        eps=[x.detach().clone() for x in model.autoencoder._diagnostic_eps]
        eps_hashes=[tensor_hash(x) for x in eps]
        assert not set(eps_hashes)&(TRAIN_NOISE_HASHES|EVAL_NOISE_HASHES)
        TRAIN_NOISE_HASHES.update(eps_hashes)
        torch.save(dict(update=step,epsilon=[x.cpu() for x in eps],rng_before=rng_before,rng_after=rng_after),ROOT/'training_noise'/f'{step:04d}.pt')
        mu_delta=[b.norm([now.detach()-prev])/(b.norm([prev])+1e-30) for now,prev in zip(rows[0],previous_mu)]
        previous_mu=[x.detach().clone() for x in rows[0]]
        rec_value,parts,_=b.full_objective(rows,data,scales)
        kl,kmu,ksigma,kl_info=kl_parts(rows);beta=beta_at(step)
        assert abs(float(rec_value.detach())-sum(x['edge']+x['face'] for x in parts)/M)<1e-5
        assert abs(kl_info['total']-sum(x['total'] for x in kl_info['per_mesh'])/M)<1e-5
        value=rec_value+beta*kl
        reconstruction_before=float(rec_value.detach())
        value.backward()
        assert torch.equal(rng_after,train_rng.get_state())
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+M*step
        assert all(torch.equal(x,y) for x,y in zip(eps,model.autoencoder._diagnostic_eps))
        assert all(p.grad is not None for p in active)
        grad={name:c.m.gradnorm(group.values()) for name,group in groups.items()}
        preclip=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
        before={n:p.detach().clone() for n,p in named.items()}
        loss_before=float(value.detach())
        del value,rec_value,kl,kmu,ksigma,rows,eps
        optimizer.step()
        updates={name:movement(group,before) for name,group in groups.items()};del before
        record=dict(update=step,beta=beta,reconstruction_before_update=reconstruction_before,kl_before_update=kl_info,parameter_state_for_training_forward=step-1,loss_before_update=loss_before,parts_before_update=parts,posterior_before_update=stats,gradient_norms_preclip=grad,global_norm_preclip=float(preclip),clip_coefficient=min(1.,1./(float(preclip)+1e-6)),actual_updates=updates,mu_change_since_previous_training_forward=mu_delta,epsilon_sha256=eps_hashes,rng_before_sha256=tensor_hash(rng_before),rng_after_sha256=tensor_hash(rng_after),backward_rng_unchanged=True,noise_draws_total=model.autoencoder.diagnostic_train_draws,seconds=time.monotonic()-started)
        if step in CHECKS:
            sr,ss=detached_forward('sample',c.SEEDS);sample,_=evaluate(sr,ss,step,'sample');del sr
            mr,ms=detached_forward('mu');mu,_=evaluate(mr,ms,step,'mu');del mr
            check=dict(step=step,mu=mu,sample=sample);evals.append(check);record['evaluation_after_update']=check
            print('CHECK',step,'mu/fixed perfect',mu['all_meshes_perfect'],sample['all_meshes_perfect'],'sigma p95',[p['std']['p95'] for p in ss],flush=True)
        if step in NOISE_CHECKS:
            save_checkpoint(step)
            noise_checks[step]=noise_evaluation(step)
            record['noise_evaluation_after_update']=noise_checks[step]
        assert torch.equal(rng_after,train_rng.get_state())
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+M*step
        log.write(json.dumps(record,allow_nan=False)+'\n');update_history.append(record)
        if step%10==0:print('UPDATE',step,'sampled loss',loss_before,'logvar delta',updates['logvar']['delta_l2'],flush=True)
training_seconds=time.monotonic()-started
final_mu=evals[-1]['mu'];final_fixed=evals[-1]['sample']
updates_pass=all(r['actual_updates'][name]['delta_l2']>0 for r in update_history for name in groups)
perturbation_pass=all(p['actual_noise_nonzero_elements']>0 for r in update_history for p in r['posterior_before_update'])
final_unseen=noise_evaluation(STEPS,final_unseen=True)
full_acceptance=final_mu['all_meshes_perfect'] and noise_checks[STEPS]['simultaneous_perfect']==50 and final_unseen['simultaneous_perfect']==50 and updates_pass and perturbation_pass
assert b.SEEN==set(b.ATTENTION_NAMES.values())
assert b.AUDIT['dtype_routes']['forward']==b.AUDIT['dtype_routes']['recompute']
assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+M*STEPS and len(TRAIN_NOISE_HASHES)==M*STEPS
c.write(ROOT/'precision_audit.json',dict(**b.AUDIT,attention_modules=sorted(b.SEEN)))
c.write(ROOT/'complete.json',dict(updates=STEPS,initial=evals[0],final=evals[-1],noise_evaluations=noise_checks,acceptance=dict(full_acceptance=full_acceptance,nonzero_updates=updates_pass,nonzero_perturbation=perturbation_pass,scope='Endpoint strict recovery observation; failure within1000 updates is not capacity or VAE failure'),final_unseen=final_unseen,training_rng_draws=M*STEPS,training_rng_draws_cumulative=DRAW_OFFSET+M*STEPS,unique_training_epsilon_hashes=len(TRAIN_NOISE_HASHES),cumulative_parameter_change={name:movement(group,initial) for name,group in groups.items()},seconds_through_monitoring=training_seconds,seconds_including_final_unseen=time.monotonic()-started))
print('COMPLETE',json.dumps(dict(full_acceptance=full_acceptance,final_unseen=final_unseen)),flush=True)
