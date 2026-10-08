"""math00_mu_step200 -> 200 fresh-epsilon, KL-zero sampling updates."""
import argparse
parser=argparse.ArgumentParser();parser.add_argument("--preflight",action="store_true");parser.add_argument("--beta",type=float,default=0.);args=parser.parse_args()
import backend00 as b
import fcntl
import json
import time
import inspect
import hashlib
from pathlib import Path

torch=b.torch; np=b.np; c=b.c; ROOT=b.ROOT
PREVIOUS=ROOT.parent/'math00_fresh_sampling_20260912'
START=PREVIOUS/'checkpoint-update0200.pt'
CHECKS={0,1,10,20,50,100,200}
NOISE_CHECKS={0,20,50,100,200}
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
baseline=dict(name='math00_fresh_eps_step200',checkpoint=str(START),sha256=start_sha,adam_steps=dict(encoder_decoder_heads=400,logvar=200),backend_sha=cp['diagnostic_manifest']['backend_sha'],verification=str(PREVIOUS/'complete.json'))
c.write(ROOT/'math00_fresh_eps_step200.json',baseline)
runtime=c.m.install_clamp(model,-20.)
assert runtime.count('noise = torch.randn_like(sample_mu)')==1
runtime=runtime.replace('noise = torch.randn_like(sample_mu)', 'noise = torch.randn(sample_mu.shape, dtype=sample_mu.dtype, device=sample_mu.device, generator=autoencoder.diagnostic_train_rng)\n                autoencoder.diagnostic_train_draws += 1')
exec(compile(runtime,str(ROOT/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__)
(ROOT/'sampling_forward.py').write_text(runtime)
train_rng=torch.Generator(device='cuda');train_rng.set_state(cp['training_rng_state'])
DRAW_OFFSET=cp['training_noise_draws'];assert DRAW_OFFSET==400
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
        assert float(optimizer.state[p]['step'])==(200 if group['name']=='logvar' else 400)
active=[p for group in groups.values() for p in group.values()]
assert len(active)==len({id(p) for p in active})==len(list(model.parameters()))
assert {id(p) for p in active}=={id(p) for p in model.parameters()}
assert all(p.requires_grad and p.dtype==torch.float32 for p in active)
assert all(torch.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
initial={n:p.detach().clone() for n,p in named.items()}
scales=model.scoring_contract();c.PAIR_CHUNK=cp['args']['pair_chunk_size']
data=[];pool_hashes={}
for i,uid in enumerate(c.UIDS):
    path=c.m.teacher.modules.ab.PREVIOUS/(uid+'_pool.npz');pool_hashes[uid]=c.m.probe.digest(path)
    d=np.load(path);data.append(d)
    assert np.array_equal(d['vertices'],batch.vertices[i,:len(d['vertices'])].cpu().numpy())
    assert np.array_equal(d['positive'],batch.face_set[i].cpu().numpy())
manifest=dict(baseline=baseline,backend='unchanged math00; deterministic Graph forward/backward, FP32 MATH on all 28 attention modules; TF32/autocast off',groups={n:list(g) for n,g in groups.items()},lr={g['name']:g['lr'] for g in optimizer.param_groups},optimizer='restore all five Adam groups from fresh-epsilon step200 exactly',existing_adam_exact=True,logvar_adam_initial='restored step200',betas=optimizer.defaults['betas'],eps=optimizer.defaults['eps'],weight_decay=0.,clip=1.,kl=0.,mode='fresh epsilon each optimizer step',training_rng_seed=SEEDS['training_rng_seed'],fixed_diagnostic_seeds=c.SEEDS,evaluation_seeds=SEEDS['evaluation'],final_unseen_seeds=SEEDS['final_unseen'],noise_check_steps=sorted(NOISE_CHECKS),soft4='fully-diff Edge+Face; equal mean across meshes; tau1 epsilon1e-8; original candidate/reduction',logvar_clamp=[-20.,10.],logvar_reinitialized=False,mu_loss_weight=0.,updates=200,checks=sorted(CHECKS),threshold=0.,source_sha256=cp['diagnostic_manifest']['source_sha256'],pool_sha256=pool_hashes,script_sha=c.m.probe.digest(Path(__file__)),backend_sha=c.m.probe.digest(ROOT/'backend00.py'),gpu=0,device=torch.cuda.get_device_name(),torch_version=torch.__version__)
manifest.update(beta_target=args.beta,beta_schedule='update1=0; update50=target; linear in between; hold through update200',training_rng_initial_sha256=tensor_hash(train_rng.get_state()),training_draw_offset=DRAW_OFFSET,kl_definition='equal mesh mean of 0.5 mean_vc(mu^2+exp(logvar)-1-logvar)',kl_gradient_checks=[0,20,50,100,200])
assert pool_hashes==cp['diagnostic_manifest']['pool_sha256']
manifest['monitor_npz_serialization']=False
manifest['rng_checkpoint_scope']='attention core only; noise generated once per mesh before decoder; assert generator unchanged in backward and evaluation'
c.write(ROOT/'manifest.json',manifest)

def kl_parts(rows):
    per=[dict(mu=.5*mu.float().square().mean(),sigma=.5*(lv.float().exp()-1-lv.float()).mean(),total=.5*(mu.float().square()+lv.float().exp()-1-lv.float()).mean()) for mu,lv in zip(rows[0],rows[1])]
    kmu=torch.stack([x['mu'] for x in per]).mean();ksigma=torch.stack([x['sigma'] for x in per]).mean()
    kl=torch.stack([x['total'] for x in per]).mean()
    info=dict(total=float(kl.detach()),mu=float(kmu.detach()),sigma=float(ksigma.detach()),per_mesh=[{k:float(v.detach()) for k,v in p.items()} for p in per])
    return kl,kmu,ksigma,info

def gradient_probe(rec,kl,kmu,ksigma,beta):
    tensors={}
    for name,loss in [('reconstruction',rec),('kl',kl),('kl_mu',kmu),('kl_sigma',ksigma)]:
        tensors[name]=torch.autograd.grad(loss,active,retain_graph=True,allow_unused=True)
    def norms(gs):
        result={};offset=0
        for name,group in groups.items():
            part=[g for g in gs[offset:offset+len(group)] if g is not None];offset+=len(group)
            result[name]=b.norm(part) if part else 0.
        result['total']=sum(v*v for v in result.values())**.5
        return result
    values={n:norms(gs) for n,gs in tensors.items()}
    weighted={n:beta*v for n,v in values['kl'].items()}
    ratios={n:weighted[n]/max(v,1e-30) for n,v in values['reconstruction'].items()}
    return dict(norms=values,beta=beta,weighted_kl_norms=weighted,weighted_kl_to_reconstruction=ratios)

def beta_at(step):return args.beta*min(1.,max(0.,(step-1)/49.))

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
    result=dict(step=step,mode=mode,loss=float(value),kl=kl_info,parts=parts,posterior=stats,rec=compact(rec),four_way_perfect=perfect(rec))
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
            if (j+1)%10==0:print('NOISE',tag,step,j+1,'perfect',sum(v['four_way_perfect'] for v in values),flush=True)
            del rows
    assert torch.equal(rng_before,train_rng.get_state())
    result=dict(step=step,total=50,simultaneous_perfect=sum(v['four_way_perfect'] for v in values),seeds=seed_pairs,mean_soft4=float(np.mean([v['loss'] for v in values])),std_soft4=float(np.std([v['loss'] for v in values])),mean_parts=[{k:float(np.mean([v['parts'][i][k] for v in values])) for k in ['edge','face']} for i in range(2)],mean_kl={k:float(np.mean([v['kl'][k] for v in values])) for k in ['total','mu','sigma']},rng_unchanged=True)
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
    rng=train_rng.get_state().clone()
    rows,stats=forward('sample',None)
    rec,parts,_=b.full_objective(rows,data,scales);kl,kmu,ksigma,ki=kl_parts(rows)
    probe=gradient_probe(rec,kl,kmu,ksigma,1e-4)
    result=dict(reconstruction=float(rec.detach()),parts=parts,kl=ki,gradient_probe=probe,posterior=stats,epsilon_sha256=[tensor_hash(e) for e in model.autoencoder._diagnostic_eps],initial_rng_sha256=tensor_hash(rng),no_optimizer_updates=True,adam_steps={g['name']:sorted(set(float(optimizer.state[p]['step']) for p in g['params'])) for g in optimizer.param_groups})
    train_rng.set_state(rng);model.autoencoder.diagnostic_train_draws=DRAW_OFFSET
    assert all(torch.equal(p,initial[n]) for n,p in named.items())
    c.write(ROOT/'preflight.json',result);print('PREFLIGHT',json.dumps(result),flush=True)
    raise SystemExit(0)

TRAIN_NOISE_HASHES=set()
rows,stats=detached_forward('mu')
mu0,tables=evaluate(rows,stats,0,'mu')
old=json.loads((PREVIOUS/'mu_step0200.json').read_text())
assert mu0['four_way_perfect'] and mu0['loss']==old['loss'] and mu0['parts']==old['parts']
for uid,table in zip(c.UIDS,tables):
    oldtable=np.load(PREVIOUS/'mu_step0200'/f'{uid}.npz')
    assert all(np.array_equal(table[k],oldtable[k]) for k in oldtable.files)
previous_mu=[x.detach().clone() for x in rows[0]]
del rows,tables
rows,stats=detached_forward('sample',c.SEEDS)
sample0,tables=evaluate(rows,stats,0,'sample')
fixed0=json.loads((PREVIOUS/'sample_step0200.json').read_text())
assert sample0['four_way_perfect'] and all(sample0[k]==fixed0[k] for k in ['loss','parts','rec','posterior'])
for uid,table in zip(c.UIDS,tables):
    oldtable=np.load(PREVIOUS/'sample_step0200'/f'{uid}.npz')
    assert all(np.array_equal(table[k],oldtable[k]) for k in oldtable.files)
del rows,tables
noise_checks={0:noise_evaluation(0)}
old_noise=[json.loads(s) for s in (PREVIOUS/'independent_step0200.jsonl').read_text().splitlines()]
new_noise=[json.loads(s) for s in (ROOT/'independent_step0000.jsonl').read_text().splitlines()]
for old,new in zip(old_noise,new_noise):
    assert all(new[k]==v for k,v in old.items() if k!='step')
assert len(old_noise)==len(new_noise)==50
EVAL_NOISE_HASHES={v for row in new_noise for v in row['epsilon_sha256']}
assert all(torch.equal(p,initial[n]) for n,p in named.items())
assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET
c.write(ROOT/'step0_acceptance.json',dict(mu=mu0,fixed_epsilon=sample0,independent=noise_checks[0],mu_matches_previous_exact=True,entire_fixed_control_step0_exact=True,no_updates_yet=True))
save_checkpoint(0)
(ROOT/'training_noise').mkdir(exist_ok=True)
print('BEGIN_TRAINING fresh RNG',SEEDS['training_rng_seed'],flush=True)
started=time.monotonic();evals=[dict(step=0,mu=mu0,sample=sample0)];update_history=[]
with (ROOT/'updates.jsonl').open('w',buffering=1) as log:
    for step in range(1,201):
        optimizer.zero_grad(set_to_none=True)
        rng_before=train_rng.get_state().clone()
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+2*(step-1)
        rows,stats=forward('sample',None)
        rng_after=train_rng.get_state().clone()
        assert not torch.equal(rng_before,rng_after)
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+2*step
        eps=[x.detach().clone() for x in model.autoencoder._diagnostic_eps]
        eps_hashes=[tensor_hash(x) for x in eps]
        assert not set(eps_hashes)&(TRAIN_NOISE_HASHES|EVAL_NOISE_HASHES)
        TRAIN_NOISE_HASHES.update(eps_hashes)
        torch.save(dict(update=step,epsilon=[x.cpu() for x in eps],rng_before=rng_before,rng_after=rng_after),ROOT/'training_noise'/f'{step:04d}.pt')
        mu_delta=[b.norm([now.detach()-prev])/(b.norm([prev])+1e-30) for now,prev in zip(rows[0],previous_mu)]
        previous_mu=[x.detach().clone() for x in rows[0]]
        rec_value,parts,_=b.full_objective(rows,data,scales)
        kl,kmu,ksigma,kl_info=kl_parts(rows);beta=beta_at(step)
        probe=gradient_probe(rec_value,kl,kmu,ksigma,beta) if step in [1,21,51,101,200] else None
        value=rec_value+beta*kl
        reconstruction_before=float(rec_value.detach())
        value.backward()
        assert torch.equal(rng_after,train_rng.get_state())
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+2*step
        assert all(torch.equal(x,y) for x,y in zip(eps,model.autoencoder._diagnostic_eps))
        assert all(p.grad is not None for p in active)
        grad={name:c.m.gradnorm(group.values()) for name,group in groups.items()}
        preclip=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
        before={n:p.detach().clone() for n,p in named.items()}
        loss_before=float(value.detach())
        del value,rec_value,kl,kmu,ksigma,rows,eps
        optimizer.step()
        updates={name:movement(group,before) for name,group in groups.items()};del before
        record=dict(update=step,beta=beta,reconstruction_before_update=reconstruction_before,kl_before_update=kl_info,independent_gradient_probe=probe,parameter_state_for_training_forward=step-1,loss_before_update=loss_before,parts_before_update=parts,posterior_before_update=stats,gradient_norms_preclip=grad,global_norm_preclip=float(preclip),clip_coefficient=min(1.,1./(float(preclip)+1e-6)),actual_updates=updates,mu_change_since_previous_training_forward=mu_delta,epsilon_sha256=eps_hashes,rng_before_sha256=tensor_hash(rng_before),rng_after_sha256=tensor_hash(rng_after),backward_rng_unchanged=True,noise_draws_total=model.autoencoder.diagnostic_train_draws,seconds=time.monotonic()-started)
        if step in CHECKS:
            sr,ss=detached_forward('sample',c.SEEDS);sample,_=evaluate(sr,ss,step,'sample');del sr
            mr,ms=detached_forward('mu');mu,_=evaluate(mr,ms,step,'mu');del mr
            check=dict(step=step,mu=mu,sample=sample);evals.append(check);record['evaluation_after_update']=check
            print('CHECK',step,'mu/fixed perfect',mu['four_way_perfect'],sample['four_way_perfect'],'sigma p95',[p['std']['p95'] for p in ss],flush=True)
        if step in NOISE_CHECKS:
            save_checkpoint(step)
            noise_checks[step]=noise_evaluation(step)
            record['noise_evaluation_after_update']=noise_checks[step]
        assert torch.equal(rng_after,train_rng.get_state())
        assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+2*step
        log.write(json.dumps(record,allow_nan=False)+'\n');update_history.append(record)
        if step%10==0:print('UPDATE',step,'sampled loss',loss_before,'logvar delta',updates['logvar']['delta_l2'],flush=True)
training_seconds=time.monotonic()-started
mu_pass=all(next(r for r in evals if r['step']==s)['mu']['four_way_perfect'] for s in [50,100,200])
noise_pass=all(noise_checks[s]['simultaneous_perfect']==50 for s in [50,100,200])
updates_pass=all(r['actual_updates'][name]['delta_l2']>0 for r in update_history for name in groups)
perturbation_pass=all(p['actual_noise_nonzero_elements']>0 for r in update_history for p in r['posterior_before_update'])
budget_pass=mu_pass and noise_pass and updates_pass and perturbation_pass
final_unseen=noise_evaluation(200,final_unseen=True)
assert b.SEEN==set(b.ATTENTION_NAMES.values())
assert b.AUDIT['dtype_routes']['forward']==b.AUDIT['dtype_routes']['recompute']
assert model.autoencoder.diagnostic_train_draws==DRAW_OFFSET+400 and len(TRAIN_NOISE_HASHES)==400
c.write(ROOT/'precision_audit.json',dict(**b.AUDIT,attention_modules=sorted(b.SEEN)))
c.write(ROOT/'complete.json',dict(updates=200,initial=evals[0],final=evals[-1],noise_evaluations=noise_checks,acceptance=dict(mu_pass=mu_pass,noise_pass=noise_pass,nonzero_updates=updates_pass,nonzero_perturbation=perturbation_pass,budget_pass=budget_pass),final_unseen=final_unseen,final_unseen_skipped_reason=None,training_rng_draws=400,training_rng_draws_cumulative=DRAW_OFFSET+400,unique_training_epsilon_hashes=len(TRAIN_NOISE_HASHES),cumulative_parameter_change={name:movement(group,initial) for name,group in groups.items()},seconds_through_monitoring=training_seconds,seconds_including_final_unseen=time.monotonic()-started))
print('COMPLETE',json.dumps(dict(budget_pass=budget_pass,final_unseen=final_unseen)),flush=True)
