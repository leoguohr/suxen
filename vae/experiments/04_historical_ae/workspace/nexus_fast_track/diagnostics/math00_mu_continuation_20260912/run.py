"""B16100 -> verified math00, step0 acceptance then 200 mu reconstruction updates."""
import backend00 as b
import fcntl
import json
import time
from pathlib import Path

torch=b.torch; np=b.np; c=b.c; ROOT=b.ROOT
CHECKS={0,1,10,20,50,100,200}
lock=(ROOT/'run.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'updates.jsonl').exists()
cp,model,batch,enc,data=c.setup(0)
b.install_deterministic(model)
for section,blocks in [('encoder',model.autoencoder.encoder_blocks),('decoder',model.autoencoder.decoder_blocks)]:
    for i,block in enumerate(blocks):
        for name,module in block.named_modules():
            if isinstance(module,torch.nn.MultiheadAttention):
                b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
assert len(b.ATTENTION_NAMES)==28
scales=model.scoring_contract()
named=dict(model.autoencoder.named_parameters())
encnames=set(enc)
groups={name:{} for name in ['encoder_mu','decoder_body','edge_head','face_head']}
for n,p in named.items():
    p.requires_grad_(not n.startswith('log_variance.'))
    if not p.requires_grad:continue
    group='encoder_mu' if n in encnames else 'edge_head' if n.startswith('edge_embedding.') else 'face_head' if n.startswith('face_embedding.') else 'decoder_body'
    groups[group][n]=p
active=[p for group in groups.values() for p in group.values()]
assert len(active)==len({id(p) for p in active})
assert {id(p) for p in active}=={id(p) for p in model.parameters() if p.requires_grad}
assert all(p.dtype==torch.float32 for p in model.parameters())
frozen={n:p.detach().clone() for n,p in named.items() if not p.requires_grad}
initial={n:p.detach().clone() for n,p in named.items() if p.requires_grad}

def forward(mode):
    return c.m.forward(model,batch,mode,c.SEEDS if mode=='sample' else None)[0]

def compact(rec):
    return [{k:r[k] for k in ['uid','edge','face','gt_faces_missing_from_edge_candidates','margins']} for r in rec]

def perfect(rec):
    return all(r[k]['fp']==r[k]['fn']==0 for r in rec for k in ['edge','face'])

def evaluate(rows,step,mode):
    detached=tuple(tuple(v.detach() for v in row) for row in rows)
    value,parts,_=b.full_objective(detached,data,scales)
    rec,tables=c.capture(detached,data,scales,ROOT/f'{mode}_step{step:04d}')
    result=dict(step=step,mode=mode,loss=float(value),parts=parts,rec=compact(rec),four_way_perfect=perfect(rec))
    c.write(ROOT/f'{mode}_step{step:04d}.json',result)
    return result,tables

# The old fixed-epsilon baseline is reused as an exact anchor after unfreezing.
sample=forward('sample')
sample0,tables=evaluate(sample,0,'sample')
oldroot=ROOT.parent/'cast_2x2_20260912/00'
old=json.loads((oldroot/'baseline.json').read_text())
assert sample0['loss']==old['loss'] and sample0['parts']==old['parts']
for uid,table in zip(c.UIDS,tables):
    oldtable=np.load(oldroot/'baseline'/f'{uid}.npz')
    assert all(np.array_equal(table[k],oldtable[k]) for k in oldtable.files)
del sample,tables
rows=forward('mu')
mu0,tables=evaluate(rows,0,'mu')
repeat=forward('mu')
with torch.no_grad():diff=b.repeat_check(rows,repeat,tables,scales)
assert all(v==0 for v in diff.values())
del repeat,tables
stage='mu_maintenance' if mu0['four_way_perfect'] else 'mu_recovery'
c.write(ROOT/'step0_acceptance.json',dict(mu=mu0,fixed_epsilon=sample0,mu_repeat_differences=diff,sample_anchor_exact=True,selected_stage=stage,prior_00_mu_training_found=False))
print('STEP0',json.dumps(dict(stage=stage,mu=mu0,sample=sample0)),flush=True)

optimizer=torch.optim.Adam([dict(params=list(group.values()),lr=1e-8 if name=='encoder_mu' else 1e-7,name=name) for name,group in groups.items()],weight_decay=0.)
assert len(optimizer.state)==0
manifest=dict(start_checkpoint=str(c.m.SOURCE/'B/checkpoint-16100.pt'),checkpoint_sha=c.m.START_SHA,source_sha256=cp['diagnostic_run']['source_sha256'],script_sha=c.m.probe.digest(Path(__file__)),backend_sha=c.m.probe.digest(ROOT/'backend00.py'),backend_template_sha=c.m.probe.digest(ROOT.parent/'cast_2x2_20260912/run.py'),backend='00; deterministic Graph forward/backward; FP32 MATH encoder and decoder; TF32/autocast disabled',groups={name:list(group) for name,group in groups.items()},lr={g['name']:g['lr'] for g in optimizer.param_groups},optimizer='fresh Adam',betas=optimizer.defaults['betas'],eps=optimizer.defaults['eps'],weight_decay=0.,clip=1.,kl=0.,soft4='fully differentiable membership; tau1 epsilon1e-8; same reduction and candidate pool',mode='mu',updates=200,checks=sorted(CHECKS),threshold=0.,actual_face='enumerate each evaluation predicted edge graph',logvar='frozen; no reconstruction gradient',selected_stage=stage,device=torch.cuda.get_device_name(),torch_version=torch.__version__)
c.write(ROOT/'manifest.json',manifest)

def movement(group,before):
    dsq=torch.zeros((),device='cuda',dtype=torch.float64);tsq=dsq.clone();changed=torch.zeros((),device='cuda',dtype=torch.int64)
    for n,p in group.items():
        delta=p.detach().double()-before[n].double()
        dsq+=delta.square().sum();tsq+=before[n].double().square().sum();changed+=(delta!=0).sum()
    dn=float(dsq.sqrt());tn=float(tsq.sqrt())
    return dict(delta_l2=dn,theta_l2=tn,relative_l2=dn/(tn+1e-30),changed_elements=int(changed))

def save_checkpoint(step):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=step,parent_checkpoint_sha=c.m.START_SHA,diagnostic_manifest=manifest,args=cp['args']),ROOT/f'checkpoint-update{step:04d}.pt')

started=time.monotonic();evals=[mu0]
save_checkpoint(0)
with (ROOT/'updates.jsonl').open('w',buffering=1) as log:
    for step in range(1,201):
        optimizer.zero_grad(set_to_none=True)
        value,parts,_=b.full_objective(rows,data,scales)
        value.backward()
        assert all(named[n].grad is None for n in frozen)
        grad={name:c.m.gradnorm(group.values()) for name,group in groups.items()}
        preclip=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True)
        coefficient=min(1.,1./(float(preclip)+1e-6))
        before={n:p.detach().clone() for group in groups.values() for n,p in group.items()}
        previous_mu=[x.detach().clone() for x in rows[0]]
        loss_before=float(value.detach())
        del value,rows
        optimizer.step()
        updates={name:movement(group,before) for name,group in groups.items()}
        del before
        rows=forward('mu')
        mu_delta=[b.norm([now.detach()-prev])/(b.norm([prev])+1e-30) for now,prev in zip(rows[0],previous_mu)]
        record=dict(update=step,loss_before_update=loss_before,parts_before_update=parts,gradient_norms_preclip=grad,global_norm_preclip=float(preclip),clip_coefficient=coefficient,actual_updates=updates,mu_relative_change=mu_delta,seconds=time.monotonic()-started)
        if step in CHECKS:
            result,_=evaluate(rows,step,'mu');evals.append(result);record['evaluation_after_update']=result
            print('CHECK',step,json.dumps(dict(loss=result['loss'],rec=result['rec'],updates=updates,seconds=record['seconds'])),flush=True)
            if step==200:save_checkpoint(step)
        log.write(json.dumps(record,allow_nan=False)+'\n')
        if step%10==0:print('UPDATE',step,loss_before,updates['encoder_mu'],flush=True)
        del previous_mu
for n,p in frozen.items():assert torch.equal(named[n],p)
assert b.SEEN==set(b.ATTENTION_NAMES.values())
assert b.AUDIT['dtype_routes']['forward']==b.AUDIT['dtype_routes']['recompute']
c.write(ROOT/'precision_audit.json',dict(**b.AUDIT,attention_modules=sorted(b.SEEN)))
c.write(ROOT/'complete.json',dict(updates=200,stage=stage,initial=mu0,final=evals[-1],perfect_evaluation_steps=[r['step'] for r in evals if r['four_way_perfect']],cumulative_parameter_change={name:movement(group,initial) for name,group in groups.items()},logvar_unchanged=True,seconds=time.monotonic()-started))
print('COMPLETE',time.monotonic()-started,flush=True)
