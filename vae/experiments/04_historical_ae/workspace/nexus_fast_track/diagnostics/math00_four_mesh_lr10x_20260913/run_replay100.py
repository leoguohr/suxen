"""Exact 100-update replay used only to fill the missing step100 noise evaluation."""
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--branch',required=True,choices=['control','high10x'])
args=parser.parse_args()

import backend00 as b
import fcntl
import json
import time
import inspect
import hashlib
from pathlib import Path

torch=b.torch;np=b.np;c=b.c
BASE=Path(__file__).resolve().parent
SOURCE=BASE.parent/'math00_four_mesh_20260913'
ROOT=BASE/'replay100'/args.branch;ROOT.mkdir(parents=True,exist_ok=True);b.ROOT=ROOT;b.backend.ROOT=ROOT
(ROOT/'C_graph_only').mkdir(exist_ok=True)
SELECTION=json.loads((BASE/'selection.json').read_text())
c.UIDS=SELECTION['uids'];c.m.UIDS=c.UIDS;c.m.probe.UIDS=c.UIDS
c.SEEDS=[970000,970001,970002,970003]
M=len(c.UIDS);assert M==4
STEPS=100;PARENT_UPDATES=1000
START=SOURCE/'checkpoint-update1000.pt'
CHECKS={0,1,10,20,50,100}
NOISE_CHECKS={0,100}
SEEDS=json.loads((BASE/'evaluation_seeds.json').read_text())
assert SEEDS['training']==c.SEEDS and len(SEEDS['evaluation'])==len(SEEDS['final_unseen'])==50
lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (ROOT/'updates.jsonl').exists()
torch.set_num_threads(1);torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
cp,model,batch=c.m.probe.setup_model(START)
assert cp['completed_updates']==1000
assert c.m.probe.digest(BASE/'backend00.py')==cp['diagnostic_manifest']['backend_sha']
for p,sha in cp['diagnostic_manifest']['source_sha256'].items():assert c.m.probe.digest(p)==sha
start_sha=c.m.probe.digest(START)

runtime=c.m.install_clamp(model,-20.)
assert runtime.count('noise = torch.randn_like(sample_mu)')==1
runtime=runtime.replace('noise = torch.randn_like(sample_mu)','noise = torch.randn(sample_mu.shape, dtype=sample_mu.dtype, device=sample_mu.device, generator=autoencoder.diagnostic_train_rng)\n                autoencoder.diagnostic_train_draws += 1')
exec(compile(runtime,str(ROOT/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__)
(ROOT/'sampling_forward.py').write_text(runtime)
train_rng=torch.Generator(device='cuda');train_rng.set_state(cp['training_rng_state'])
DRAW_OFFSET=cp['training_noise_draws'];assert DRAW_OFFSET==5200
model.autoencoder.diagnostic_train_rng=train_rng;model.autoencoder.diagnostic_train_draws=DRAW_OFFSET

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

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
optimizer=torch.optim.Adam([dict(params=list(group.values()),lr=next(x['lr'] for x in cp['optimizer']['param_groups'] if x['name']==name),name=name) for name,group in groups.items()],weight_decay=0.)
optimizer.load_state_dict(cp['optimizer'])
saved_lrs={g['name']:g['lr'] for g in optimizer.param_groups}
assert saved_lrs=={'encoder_mu':1e-8,'decoder_body':1e-7,'edge_head':1e-7,'face_head':1e-7,'logvar':1e-4}
for group,saved_group in zip(optimizer.param_groups,cp['optimizer']['param_groups']):
    for p,index in zip(group['params'],saved_group['params']):
        for k,value in cp['optimizer']['state'][index].items():assert torch.equal(optimizer.state[p][k].cpu(),value)
        assert float(optimizer.state[p]['step'])==(1600 if group['name']=='logvar' else 1800)
if args.branch=='high10x':
    for group in optimizer.param_groups:
        if group['name']!='logvar':group['lr']*=10
effective_lrs={g['name']:g['lr'] for g in optimizer.param_groups}
assert effective_lrs['logvar']==1e-4
active=[p for group in groups.values() for p in group.values()]
assert len(active)==len({id(p) for p in active})==len(list(model.parameters()))
assert all(p.requires_grad and p.dtype==torch.float32 for p in active)
assert all(torch.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
initial={n:p.detach().clone() for n,p in named.items()}
scales=model.scoring_contract();c.PAIR_CHUNK=cp['args']['pair_chunk_size']
data=[];pool_hashes={}
for i,uid in enumerate(c.UIDS):
    path=BASE/(uid+'_pool.npz');pool_hashes[uid]=c.m.probe.digest(path)
    d=np.load(path);data.append(d)
    assert np.array_equal(d['vertices'],batch.vertices[i,:len(d['vertices'])].cpu().numpy())
    assert np.array_equal(d['positive'],batch.face_set[i].cpu().numpy())
assert pool_hashes==SELECTION['pool_sha256']

manifest=dict(branch=args.branch,parent_checkpoint=str(START),parent_sha256=start_sha,parent_completed_updates=PARENT_UPDATES,selected_uids=c.UIDS,mesh_count=M,backend='unchanged math00',lr=effective_lrs,parent_lr=saved_lrs,lr_change='none' if args.branch=='control' else '10x encoder_mu, decoder_body, edge_head, face_head; logvar unchanged',optimizer='all five Adam states restored exactly from four-mesh step1000',beta=1e-4,clip=1.,weight_decay=0.,updates=STEPS,checks=sorted(CHECKS),noise_checks=sorted(NOISE_CHECKS),training_draw_offset=DRAW_OFFSET,training_rng_initial_sha256=tensor_hash(train_rng.get_state()),fixed_diagnostic_seeds=c.SEEDS,evaluation_seeds=SEEDS['evaluation'],final_unseen_seeds=SEEDS['final_unseen'],pool_sha256=pool_hashes,script_sha=c.m.probe.digest(Path(__file__)),backend_sha=c.m.probe.digest(BASE/'backend00.py'),comparison_rule='Both branches independently restore identical parent RNG, so update-wise epsilon hashes must match.',loss='fully-diff Edge+Face Soft4 plus beta times equal-mesh KL',candidate_and_actual_face='unchanged; actual Face re-enumerated from predicted Edge graph')
c.write(ROOT/'manifest.json',manifest)

def kl_parts(rows):
    per=[dict(mu=.5*mu.float().square().mean(),sigma=.5*(lv.float().exp()-1-lv.float()).mean(),total=.5*(mu.float().square()+lv.float().exp()-1-lv.float()).mean()) for mu,lv in zip(rows[0],rows[1])]
    kl=torch.stack([x['total'] for x in per]).mean()
    return kl,dict(total=float(kl.detach()),mu=float(torch.stack([x['mu'] for x in per]).mean().detach()),sigma=float(torch.stack([x['sigma'] for x in per]).mean().detach()),per_mesh=[{k:float(v.detach()) for k,v in x.items()} for x in per])

def forward(mode,seeds=None):
    rows,stats,zs=c.m.forward(model,batch,mode,seeds)
    for mu,z,stat in zip(rows[0],zs,stats):
        perturb=z.detach()-mu.detach();stat['actual_noise_rms']=float(perturb.double().square().mean().sqrt());stat['actual_noise_max_abs']=float(perturb.abs().max());stat['actual_noise_nonzero_elements']=int((perturb!=0).sum())
        if mode=='sample':assert stat['actual_noise_nonzero_elements']>0
    return rows,stats

def compact(rec):return [{k:r[k] for k in ['uid','edge','face','gt_faces_missing_from_edge_candidates','margins']} for r in rec]
def perfect(rec):return all(r[k]['fp']==r[k]['fn']==0 for r in rec for k in ['edge','face'])
def evaluate(rows,stats,step,mode,suffix='',save_npz=True):
    detached=tuple(tuple(x.detach() for x in row) for row in rows)
    value,parts,_=b.full_objective(detached,data,scales);_,ki=kl_parts(detached)
    out=ROOT/f'{mode}_step{step:04d}{suffix}';rec,_=capture(detached,data,scales,out,save_npz=save_npz)
    result=dict(step=step,mode=mode,loss=float(value),kl=ki,parts=parts,posterior=stats,rec=compact(rec),all_meshes_perfect=perfect(rec),old_meshes_perfect=perfect(rec[:2]),new_meshes_perfect=perfect(rec[2:]))
    c.write(out.with_suffix('.json'),result);return result
def detached_forward(mode,seeds=None):
    before=train_rng.get_state().clone();rows,stats=forward(mode,seeds);assert torch.equal(before,train_rng.get_state());return tuple(tuple(x.detach() for x in row) for row in rows),stats
def noise_evaluation(step,final=False):
    tag='final_unseen' if final else 'independent';seed_sets=SEEDS['final_unseen'] if final else SEEDS['evaluation'];values=[];before=train_rng.get_state().clone()
    with (ROOT/f'{tag}_step{step:04d}.jsonl').open('w',buffering=1) as out:
        for j,seeds in enumerate(seed_sets):
            rows,stats=detached_forward('sample',seeds);result=evaluate(rows,stats,step,tag,f'_{j:02d}',False);result.update(pair=j,seeds=seeds,epsilon_sha256=[tensor_hash(x) for x in model.autoencoder._diagnostic_eps]);values.append(result);out.write(json.dumps(result,allow_nan=False)+'\n')
            if (j+1)%10==0:print('NOISE',tag,step,j+1,'old/all',sum(x['old_meshes_perfect'] for x in values),sum(x['all_meshes_perfect'] for x in values),flush=True)
    assert torch.equal(before,train_rng.get_state())
    summary=dict(step=step,total=50,old_perfect=sum(x['old_meshes_perfect'] for x in values),new_perfect=sum(x['new_meshes_perfect'] for x in values),all_perfect=sum(x['all_meshes_perfect'] for x in values),mean_soft4=float(np.mean([x['loss'] for x in values])),mean_parts=[{k:float(np.mean([x['parts'][i][k] for x in values])) for k in ['edge','face']} for i in range(M)],mean_kl={k:float(np.mean([x['kl'][k] for x in values])) for k in ['total','mu','sigma']},rng_unchanged=True)
    c.write(ROOT/f'{tag}_summary_step{step:04d}.json',summary);return summary
def movement(group,before):
    dsq=torch.zeros((),device='cuda',dtype=torch.float64);tsq=dsq.clone();changed=0
    for n,p in group.items():
        delta=p.detach().double()-before[n].double();dsq+=delta.square().sum();tsq+=before[n].double().square().sum();changed+=int((delta!=0).sum())
    dn=float(dsq.sqrt());tn=float(tsq.sqrt());return dict(delta_l2=dn,theta_l2=tn,relative_l2=dn/(tn+1e-30),changed_elements=changed)
def save_checkpoint(step):
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=PARENT_UPDATES+step,additional_updates=step,parent_checkpoint_sha=start_sha,diagnostic_manifest=manifest,args=cp['args'],training_rng_state=train_rng.get_state(),training_noise_draws=model.autoencoder.diagnostic_train_draws),ROOT/f'checkpoint-update{step:04d}.pt')

rows,stats=detached_forward('mu');mu0=evaluate(rows,stats,0,'mu');del rows
rows,stats=detached_forward('sample',c.SEEDS);sample0=evaluate(rows,stats,0,'sample');del rows
noise={0:noise_evaluation(0)};save_checkpoint(0)
assert all(torch.equal(p,initial[n]) for n,p in named.items()) and model.autoencoder.diagnostic_train_draws==DRAW_OFFSET
print('BEGIN',args.branch,effective_lrs,flush=True)
started=time.monotonic();history=[];noise_hashes=set()
with (ROOT/'updates.jsonl').open('w',buffering=1) as log:
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True);rng_before=train_rng.get_state().clone();rows,stats=forward('sample',None);rng_after=train_rng.get_state().clone();eps_hashes=[tensor_hash(x) for x in model.autoencoder._diagnostic_eps]
        assert len(set(eps_hashes))==M and not set(eps_hashes)&noise_hashes;noise_hashes.update(eps_hashes)
        rec,parts,_=b.full_objective(rows,data,scales);kl,ki=kl_parts(rows);assert abs(float(rec.detach())-sum(x['edge']+x['face'] for x in parts)/M)<1e-5
        value=rec+1e-4*kl;loss=float(value.detach());reco=float(rec.detach());value.backward();assert torch.equal(rng_after,train_rng.get_state())
        grads={name:c.m.gradnorm(group.values()) for name,group in groups.items()};preclip=torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True);before={n:p.detach().clone() for n,p in named.items()};del value,rec,kl,rows;optimizer.step();updates={name:movement(group,before) for name,group in groups.items()};del before
        record=dict(update=step,beta=1e-4,reconstruction_before_update=reco,kl_before_update=ki,loss_before_update=loss,parts_before_update=parts,posterior_before_update=stats,gradient_norms_preclip=grads,global_norm_preclip=float(preclip),clip_coefficient=min(1.,1./(float(preclip)+1e-6)),actual_updates=updates,epsilon_sha256=eps_hashes,rng_before_sha256=tensor_hash(rng_before),rng_after_sha256=tensor_hash(rng_after),seconds=time.monotonic()-started)
        if step in CHECKS:
            sr,ss=detached_forward('sample',c.SEEDS);sample=evaluate(sr,ss,step,'sample');del sr;mr,ms=detached_forward('mu');mu=evaluate(mr,ms,step,'mu');del mr;record['evaluation_after_update']=dict(mu=mu,sample=sample);print('CHECK',step,'old/new/all',mu['old_meshes_perfect'],mu['new_meshes_perfect'],mu['all_meshes_perfect'],flush=True)
        if step in NOISE_CHECKS:
            save_checkpoint(step);noise[step]=noise_evaluation(step);record['noise_evaluation_after_update']=noise[step]
        log.write(json.dumps(record,allow_nan=False)+'\n');history.append(record)
        if step%10==0:print('UPDATE',step,'loss',loss,'clip',record['clip_coefficient'],flush=True)
final_unseen=None
final_mu=json.loads((ROOT/f'mu_step{STEPS:04d}.json').read_text())
complete=dict(branch=args.branch,updates=STEPS,parent_updates=PARENT_UPDATES,final_cumulative_updates=PARENT_UPDATES+STEPS,initial=dict(mu=mu0,sample=sample0,noise=noise[0]),final=dict(mu=final_mu,noise=noise[STEPS],final_unseen=final_unseen),all_groups_updated_every_step=all(x['actual_updates'][name]['delta_l2']>0 for x in history for name in groups),nonzero_sampling_every_step=all(p['actual_noise_nonzero_elements']>0 for x in history for p in x['posterior_before_update']),training_epsilon_hashes=len(noise_hashes),training_rng_final_sha256=tensor_hash(train_rng.get_state()),cumulative_parameter_change={name:movement(group,initial) for name,group in groups.items()},seconds=time.monotonic()-started)
c.write(ROOT/'complete.json',complete)
print('COMPLETE',args.branch,final_mu['old_meshes_perfect'],final_mu['new_meshes_perfect'],final_mu['all_meshes_perfect'],flush=True)
