"""math00_mu_step200 -> 200 fixed-epsilon, KL-zero sampling updates."""
import backend00 as b
import fcntl
import json
import time
from pathlib import Path

torch=b.torch; np=b.np; c=b.c; ROOT=b.ROOT
PREVIOUS=ROOT.parent/'math00_mu_continuation_20260912'
START=PREVIOUS/'checkpoint-update0200.pt'
CHECKS={0,1,10,20,50,100,200}
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
baseline=dict(name='math00_mu_step200',checkpoint=str(START),sha256=start_sha,adam_steps=200,backend_sha=cp['diagnostic_manifest']['backend_sha'],verification=str(PREVIOUS/'complete.json'))
c.write(ROOT/'math00_mu_step200.json',baseline)
c.m.install_clamp(model,-20.)
b.install_deterministic(model)
for section,blocks in [('encoder',model.autoencoder.encoder_blocks),('decoder',model.autoencoder.decoder_blocks)]:
    for i,block in enumerate(blocks):
        for name,module in block.named_modules():
            if isinstance(module,torch.nn.MultiheadAttention):b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
assert len(b.ATTENTION_NAMES)==28
model.requires_grad_(True)
named=dict(model.autoencoder.named_parameters())
groups={name:{n:named[n] for n in names} for name,names in cp['diagnostic_manifest']['groups'].items()}
assert list(groups)==['encoder_mu','decoder_body','edge_head','face_head']
optimizer=torch.optim.Adam([dict(params=list(group.values()),lr=cp['diagnostic_manifest']['lr'][name],name=name) for name,group in groups.items()],weight_decay=0.)
optimizer.load_state_dict(cp['optimizer'])
for group,saved_group in zip(optimizer.param_groups,cp['optimizer']['param_groups']):
    assert {k:v for k,v in group.items() if k!='params'}=={k:v for k,v in saved_group.items() if k!='params'}
    assert group['lr']==(1e-8 if group['name']=='encoder_mu' else 1e-7)
    for p,index in zip(group['params'],saved_group['params']):
        for k,value in cp['optimizer']['state'][index].items():assert torch.equal(optimizer.state[p][k].cpu(),value)
        assert float(optimizer.state[p]['step'])==200
groups['logvar']={n:p for n,p in named.items() if n.startswith('log_variance.')}
assert len(groups['logvar'])==2
assert all(p not in optimizer.state for p in groups['logvar'].values())
optimizer.add_param_group(dict(params=list(groups['logvar'].values()),lr=1e-4,name='logvar'))
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
manifest=dict(baseline=baseline,backend='unchanged math00; deterministic Graph forward/backward, FP32 MATH on all 28 attention modules; TF32/autocast off',groups={n:list(g) for n,g in groups.items()},lr={g['name']:g['lr'] for g in optimizer.param_groups},optimizer='restore math00 mu step200 Adam for E/D/heads; fresh added logvar group only',existing_adam_exact=True,logvar_adam_initial='absent; no old BF16 state imported',betas=optimizer.defaults['betas'],eps=optimizer.defaults['eps'],weight_decay=0.,clip=1.,kl=0.,mode='fixed epsilon sampling',training_seeds=c.SEEDS,evaluation_seeds=SEEDS['evaluation'],soft4='fully-diff Edge+Face; equal mean across meshes; tau1 epsilon1e-8; original candidate/reduction',logvar_clamp=[-20.,10.],logvar_reinitialized=False,mu_loss_weight=0.,updates=200,checks=sorted(CHECKS),threshold=0.,source_sha256=cp['diagnostic_manifest']['source_sha256'],pool_sha256=pool_hashes,script_sha=c.m.probe.digest(Path(__file__)),backend_sha=c.m.probe.digest(ROOT/'backend00.py'),gpu=0,device=torch.cuda.get_device_name(),torch_version=torch.__version__)
c.write(ROOT/'manifest.json',manifest)

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

def evaluate(rows,stats,step,mode,suffix=''):
    detached=tuple(tuple(x.detach() for x in row) for row in rows)
    value,parts,_=b.full_objective(detached,data,scales)
    out=ROOT/f'{mode}_step{step:04d}{suffix}'
    rec,tables=c.capture(detached,data,scales,out)
    result=dict(step=step,mode=mode,loss=float(value),parts=parts,posterior=stats,rec=compact(rec),four_way_perfect=perfect(rec))
    c.write(out.with_suffix('.json'),result)
    return result,tables

def detached_forward(mode,seeds=None):
    # Preserve the already verified grad-mode dispatch; release the graph before scoring.
    rows,stats=forward(mode,seeds)
    return tuple(tuple(x.detach() for x in row) for row in rows),stats

def noise_evaluation(step):
    values=[]
    with (ROOT/f'independent_step{step:04d}.jsonl').open('w',buffering=1) as out:
        for j,seeds in enumerate(SEEDS['evaluation']):
            rows,stats=detached_forward('sample',seeds)
            result,_=evaluate(rows,stats,step,'independent',f'_{j:02d}')
            result.update(pair=j,seeds=seeds);values.append(result);out.write(json.dumps(result,allow_nan=False)+'\n')
            if (j+1)%10==0:print('NOISE',step,j+1,'simultaneous perfect',sum(v['four_way_perfect'] for v in values),flush=True)
            del rows
    result=dict(step=step,total=50,simultaneous_perfect=sum(v['four_way_perfect'] for v in values),seeds=SEEDS['evaluation'])
    c.write(ROOT/f'independent_summary_step{step:04d}.json',result)
    return result

def movement(group,before):
    dsq=torch.zeros((),device='cuda',dtype=torch.float64);tsq=dsq.clone();changed=torch.zeros((),device='cuda',dtype=torch.int64)
    for n,p in group.items():
        delta=p.detach().double()-before[n].double()
        dsq+=delta.square().sum();tsq+=before[n].double().square().sum();changed+=(delta!=0).sum()
    dn=float(dsq.sqrt());tn=float(tsq.sqrt())
    return dict(delta_l2=dn,theta_l2=tn,relative_l2=dn/(tn+1e-30),changed_elements=int(changed))

def save_checkpoint(step):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=step,parent_checkpoint_sha=start_sha,diagnostic_manifest=manifest,args=cp['args']),ROOT/f'checkpoint-update{step:04d}.pt')

rows,stats=detached_forward('mu')
mu0,tables=evaluate(rows,stats,0,'mu')
old=json.loads((PREVIOUS/'mu_step0200.json').read_text())
assert mu0['four_way_perfect'] and mu0['loss']==old['loss'] and mu0['parts']==old['parts']
for uid,table in zip(c.UIDS,tables):
    oldtable=np.load(PREVIOUS/'mu_step0200'/f'{uid}.npz')
    assert all(np.array_equal(table[k],oldtable[k]) for k in oldtable.files)
del rows,tables
rows,stats=detached_forward('sample',c.SEEDS)
sample0,_=evaluate(rows,stats,0,'sample');del rows
print('STEP0',json.dumps(dict(mu=mu0,sample=sample0)),flush=True)
noise0=noise_evaluation(0)
assert all(torch.equal(p,initial[n]) for n,p in named.items())
c.write(ROOT/'step0_acceptance.json',dict(mu=mu0,fixed_epsilon=sample0,independent=noise0,mu_matches_previous_exact=True,no_updates_yet=True))
save_checkpoint(0)
print('BEGIN_TRAINING',flush=True)
started=time.monotonic();evals=[dict(step=0,mu=mu0,sample=sample0)]
rows,stats=forward('sample',c.SEEDS)
with (ROOT/'updates.jsonl').open('w',buffering=1) as log:
    for step in range(1,201):
        optimizer.zero_grad(set_to_none=True)
        value,parts,_=b.full_objective(rows,data,scales)
        value.backward()
        assert all(p.grad is not None for p in active)
        grad={name:c.m.gradnorm(group.values()) for name,group in groups.items()}
        preclip=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
        before={n:p.detach().clone() for n,p in named.items()}
        previous_mu=[x.detach().clone() for x in rows[0]]
        loss_before=float(value.detach());stats_before=stats
        del value,rows
        optimizer.step()
        updates={name:movement(group,before) for name,group in groups.items()};del before
        rows,stats=forward('sample',c.SEEDS)
        record=dict(update=step,loss_before_update=loss_before,parts_before_update=parts,posterior_before_update=stats_before,posterior_after_update=stats,gradient_norms_preclip=grad,global_norm_preclip=float(preclip),clip_coefficient=min(1.,1./(float(preclip)+1e-6)),actual_updates=updates,mu_relative_change=[b.norm([now.detach()-prev])/(b.norm([prev])+1e-30) for now,prev in zip(rows[0],previous_mu)],seconds=time.monotonic()-started)
        if step in CHECKS:
            sr,_=evaluate(rows,stats,step,'sample')
            mr,ms=detached_forward('mu');mu,_=evaluate(mr,ms,step,'mu');del mr
            check=dict(step=step,mu=mu,sample=sr);evals.append(check);record['evaluation_after_update']=check
            print('CHECK',step,'mu/sample perfect',mu['four_way_perfect'],sr['four_way_perfect'],'sample loss',sr['loss'],'sigma median',[v['std']['median'] for v in stats],flush=True)
            if step==200:save_checkpoint(step)
        log.write(json.dumps(record,allow_nan=False)+'\n')
        if step%10==0:print('UPDATE',step,'loss',loss_before,'logvar update',updates['logvar'],flush=True)
        del previous_mu
del rows
training_seconds=time.monotonic()-started
noise200=noise_evaluation(200)
assert b.SEEN==set(b.ATTENTION_NAMES.values())
assert b.AUDIT['dtype_routes']['forward']==b.AUDIT['dtype_routes']['recompute']
c.write(ROOT/'precision_audit.json',dict(**b.AUDIT,attention_modules=sorted(b.SEEN)))
c.write(ROOT/'complete.json',dict(updates=200,initial=evals[0],final=evals[-1],perfect_mu_check_steps=[r['step'] for r in evals if r['mu']['four_way_perfect']],perfect_sample_check_steps=[r['step'] for r in evals if r['sample']['four_way_perfect']],independent_before=noise0,independent_after=noise200,cumulative_parameter_change={name:movement(group,initial) for name,group in groups.items()},training_seconds=training_seconds,seconds_with_final_evaluation=time.monotonic()-started))
print('COMPLETE',flush=True)
