"""Continue cumulative4500, unchanged mu objective and Adam; 500 new updates plus error identities."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import sys
import json
import time
import random
import hashlib
import shutil
import traceback
import fcntl
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/guohaoran/nexus_fast_track')
REF = BASE/'diagnostics/math00_four_mesh_lr10x_20260913'
OLD = REF/'only804_mu'
UID = 'nexus_2k_001333'
SEED = 20260914
CHECKS = [0, 100, 200, 300, 400, 500]
PARENT = ROOT.parent/'math00_latent512_804_lr03_continue1000_20260914'
PARENT_CHECKPOINT = PARENT/'checkpoint-update1000.pt'
PARENT_SHA = 'd620530cb3654d08f98204f538e5b9bd754e260ce20a6f79a43776ac0cb3ad1b'
BRANCH = 'B_lr03_continue500'
lock = (ROOT/'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not (ROOT/'checkpoint-update0000.pt').exists(), 'Existing experiment: refuse overwrite'
sys.path.insert(0, str(REF))
# Imports in the historical diagnostic tree write their generated source. Redirect those writes.
old_write = Path.write_text
def redirected(path, *args, **kwargs):
    if path == REF/'loss_changes.txt':
        path = ROOT/'loss_changes_exact_runtime.txt'
    return old_write(path, *args, **kwargs)
Path.write_text = redirected
try:
    import backend00 as b
finally:
    Path.write_text = old_write
T, np, c = b.torch, b.np, b.c
T.set_num_threads(1)
T.cuda.set_device(0)
T.backends.cuda.matmul.allow_tf32 = False
T.backends.cudnn.allow_tf32 = False
T.set_float32_matmul_precision('highest')
random.seed(SEED); np.random.seed(SEED); T.manual_seed(SEED); T.cuda.manual_seed_all(SEED)
meta = json.loads((OLD/'manifest.json').read_text())
for path, sha in meta['source_sha256'].items():
    assert c.m.probe.digest(path) == sha, path
assert c.m.probe.digest(REF/'backend00.py') == meta['backend_sha']
assert c.m.probe.digest(PARENT_CHECKPOINT) == PARENT_SHA
cp = T.load(PARENT_CHECKPOINT, map_location='cpu', mmap=True)
assert cp['completed_updates'] == 4500
for path,sha in cp['diagnostic']['source_sha256'].items():
    assert c.m.probe.digest(path) == sha, path
args = dict(cp['args'])
assert args['latent_dim'] == 512 and args['spacetime_dim'] == 32
args.update(seed=SEED,steps=500,expected_samples=1,max_packed_meshes=1,output=str(ROOT),
            calibrate_logit_scales=False,diagnostic_mean_latent=True)
from mini_nexus.flash_varlen_topology import FlashVarlenNexus2KTopologyAESystem
# Construct the same architecture, then strictly restore every parameter before any forward.
model = FlashVarlenNexus2KTopologyAESystem.from_saved_args(args).cuda().float().eval()
model.load_state_dict(cp['model'], strict=True)
assert all(T.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
a = model.autoencoder
initialization_rng = cp['initialization_rng'].clone()
ds = c.m.probe.Nexus2KManifestDataset(Path(args['manifest']), 'train')
batch = c.m.probe.collate_packed_topology([ds[ds.index_for_uid(UID)]]).to('cuda')
c.UIDS = c.m.UIDS = c.m.probe.UIDS = [UID]
c.PAIR_CHUNK = args['pair_chunk_size']
pool_path = PARENT/(UID+'_pool.npz')
assert c.m.probe.digest(pool_path) == meta['pool_sha256'][UID]
shutil.copy2(pool_path, ROOT/pool_path.name)
pool = np.load(ROOT/pool_path.name)
assert np.array_equal(pool['vertices'], batch.vertices[0,:804].cpu().numpy())
assert np.array_equal(pool['positive'], batch.face_set[0].cpu().numpy())
assert len(pool['positive']) == 1604 and len(pool['mixed']) == 2408
b.ROOT = b.backend.ROOT = ROOT
(ROOT/'C_graph_only').mkdir(exist_ok=True)
c.m.install_clamp(model, -20.)
runtime = (PARENT/'sampling_forward.py').read_text()
(ROOT/'sampling_forward.py').write_text(runtime)
exec(compile(runtime, str(ROOT/'sampling_forward.py'), 'exec'), b.backend.b.flash.__dict__)
b.install_deterministic(model)
for section, blocks in [('encoder',a.encoder_blocks), ('decoder',a.decoder_blocks)]:
    for i, block in enumerate(blocks):
        for name, module in block.named_modules():
            if isinstance(module,T.nn.MultiheadAttention):
                b.ATTENTION_NAMES[id(module)] = f'{section}_{i:02d}/{name}'
assert len(b.ATTENTION_NAMES) == 28
assert a.mu.weight.shape == (512,512) and a.log_variance.weight.shape == (512,512)
assert a.latent_input.weight.shape == (1024,512)
assert a.edge_embedding.weight.shape == a.face_embedding.weight.shape == (32,1024)
assert len(a.decoder_blocks) == 16 and all(not hasattr(x,'linear1') for x in a.decoder_blocks)
model.requires_grad_(True); a.log_variance.requires_grad_(False)
frozen = {n:p.detach().clone() for n,p in a.log_variance.named_parameters()}
named = dict(a.named_parameters())
groups = {g:{n:named[n] for n in names} for g,names in meta['groups'].items() if g != 'logvar'}
assert {id(p) for g in groups.values() for p in g.values()} == {id(p) for p in model.parameters() if p.requires_grad}
optimizer = T.optim.Adam([dict(params=list(g.values()),name=name,lr=1e-6 if name=='encoder_mu' else 1e-5)
                          for name,g in groups.items()], betas=(.9,.999),eps=1e-8,weight_decay=0)
assert {g:list(ps) for g,ps in groups.items()} == cp['diagnostic']['groups']
optimizer.load_state_dict(cp['optimizer'])
restored = optimizer.state_dict()
assert restored['param_groups'] == cp['optimizer']['param_groups']
assert set(restored['state']) == set(cp['optimizer']['state'])
for key,state in restored['state'].items():
    for field,value in state.items():
        saved = cp['optimizer']['state'][key][field]
        assert T.equal(value.detach().cpu(),saved.cpu()) if isinstance(value,T.Tensor) else value==saved
    assert int(state['step']) == 4500
PARENT_LR={pg['name']:pg['lr'] for pg in optimizer.param_groups}
assert all(abs(lr-(3e-6 if name=='encoder_mu' else 3e-5))<1e-18 for name,lr in PARENT_LR.items())
# Retain the loaded optimizer groups verbatim, including their current LR.
random.setstate(cp['python_rng']); np.random.set_state(cp['numpy_rng'])
T.set_rng_state(cp['torch_rng']); T.cuda.set_rng_state_all(cp['cuda_rng'])
scales = model.scoring_contract()
pairs = T.triu_indices(804,804,1,device='cuda').T
pairs_cpu = pairs.cpu().numpy()
edges = pool['edges'].T if pool['edges'].shape[0]==2 else pool['edges']
edge_keys = np.unique(np.sort(edges,axis=1) @ np.array([804,1]))
ey = np.isin(pairs_cpu @ np.array([804,1]),edge_keys)
assert len(edge_keys) == 2406 and len(pairs) == 322806
gt = np.sort(pool['positive'],axis=1)
gt_keys = c.m.probe.keys(gt,804)
fy = np.r_[np.ones(1604,dtype=bool),np.zeros(2408,dtype=bool)]

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def scalar_scale(x):
    x = x.detach().float()
    assert T.isfinite(x).all(), 'nonfinite representation'
    return dict(rms=float(x.square().mean().sqrt()),maximum=float(x.abs().max()))
def norm(xs):
    return float(sum((x.detach().double().square().sum() for x in xs),T.zeros((),device='cuda',dtype=T.float64)).sqrt())
hidden = []
mu_actual = []
a.decoder_output_norm.register_forward_hook(lambda module,inputs,out:hidden.append(out.detach()))
def capture_mu(module,inputs,out):
    if out.requires_grad:out.retain_grad()
    mu_actual.append(out)
a.mu.register_forward_hook(capture_mu)
def forward():
    hidden.clear()
    mu_actual.clear()
    rows,stats,zs = c.m.forward(model,batch,'mu')
    assert rows[0][0].shape == (804,512) and zs[0].shape == (804,512)
    assert T.equal(zs[0],rows[0][0]) and hidden[-1].shape == (804,1024)
    return rows
def objective(rows):
    return b.full_objective(rows,[pool],scales)

def save(step,name=None):
    for n,p in a.log_variance.named_parameters():
        assert p.grad is None and T.equal(p,frozen[n]), 'logvar changed'
    path = ROOT/(name or f'checkpoint-update{step:04d}.pt')
    tmp = path.with_suffix('.tmp')
    T.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),args=args,
                completed_updates=4500+step,additional_updates=step,seed=SEED,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=T.get_rng_state(),cuda_rng=T.cuda.get_rng_state_all(),
                initialization_rng=initialization_rng,diagnostic=manifest),tmp)
    tmp.replace(path)
    return c.m.probe.digest(path)

@T.no_grad()
def actual_metrics(rows, step):
    el = c.edge_logits(rows[2][0],pairs,scales).cpu().numpy()
    assert np.isfinite(el).all()
    edge = c.metrics(ey,el)
    adj = np.zeros((804,804),dtype=bool)
    pp = pairs_cpu[el>0]; adj[pp[:,0],pp[:,1]] = True
    covered = adj[gt[:,0],gt[:,1]] & adj[gt[:,0],gt[:,2]] & adj[gt[:,1],gt[:,2]]
    # Stream ALL triangles of the current predicted graph, with no candidate cap.
    tp=fp=tn=nc=0
    fmin_pos=fmin_neg=None
    actual_triples=[]; actual_logits=[]
    def score(tri):
        nonlocal tp,fp,tn,nc,fmin_pos,fmin_neg
        tri = np.asarray(tri,dtype=np.int64).reshape(-1,3)
        fl = c.face_logits(rows[3][0],T.as_tensor(tri,device='cuda'),scales).cpu().numpy()
        assert np.isfinite(fl).all()
        actual_triples.append(tri.copy()); actual_logits.append(fl.copy())
        y = np.isin(c.m.probe.keys(tri,804),gt_keys)
        tp += int(((fl>0)&y).sum()); fp += int(((fl>0)&~y).sum())
        tn += int(((fl<=0)&~y).sum()); nc += len(tri)
        for positive in [True,False]:
            vals = fl[y] if positive else -fl[~y]
            if len(vals):
                v=float(vals.min())
                if positive:fmin_pos=v if fmin_pos is None else min(fmin_pos,v)
                else:fmin_neg=v if fmin_neg is None else min(fmin_neg,v)
    pending=[]
    for i in range(804):
        for j in np.flatnonzero(adj[i]):
            for k in np.flatnonzero(adj[i]&adj[j]):
                pending.append((i,int(j),int(k)))
                if len(pending)==32768:score(pending);pending=[]
    if pending:score(pending)
    # Count independently using adjacency; no omitted/chunk-duplicated candidates.
    expected=sum(int((adj[i][None,:]&adj[np.flatnonzero(adj[i])]).sum()) for i in range(804))
    assert expected == nc
    gt_logits=c.face_logits(rows[3][0],T.as_tensor(gt,device='cuda'),scales).cpu().numpy()
    assert tp == int(((gt_logits>0)&covered).sum())
    face=c.m.probe.counts(tp,fp,1604-tp);face['tn']=tn
    # Read-only identity diagnostics from the exact logits used above.
    train_face_keys=c.m.probe.keys(np.sort(np.concatenate([pool['positive'],pool['mixed']]),axis=1),804)
    at=np.concatenate(actual_triples) if actual_triples else np.empty((0,3),dtype=np.int64)
    al=np.concatenate(actual_logits) if actual_logits else np.empty(0,dtype=np.float32)
    ak=c.m.probe.keys(at,804); ay=np.isin(ak,gt_keys); in_pool=np.isin(ak,train_face_keys)
    errors=[]
    for idx in np.flatnonzero((el>0)!=ey):
        y=bool(ey[idx]); logit=float(el[idx])
        errors.append(dict(kind='edge',error='FN' if y else 'FP',id=pairs_cpu[idx].tolist(),
                           label=int(y),logit=logit,signed_margin=(1 if y else -1)*logit,
                           in_training_pool=True,in_actual_candidates=True))
    for idx in np.flatnonzero((al>0)&~ay):
        errors.append(dict(kind='face',error='FP',id=at[idx].tolist(),label=0,
                           logit=float(al[idx]),signed_margin=-float(al[idx]),
                           in_training_pool=bool(in_pool[idx]),in_actual_candidates=True))
    for idx in np.flatnonzero(~covered | (gt_logits<=0)):
        errors.append(dict(kind='face',error='FN',id=gt[idx].tolist(),label=1,
                           logit=float(gt_logits[idx]),signed_margin=float(gt_logits[idx]),
                           in_training_pool=bool(np.isin(gt_keys[idx],train_face_keys)),
                           in_actual_candidates=bool(covered[idx]),
                           cause='classifier_negative' if covered[idx] else 'missing_edge_candidate'))
    for kind,counts in [('edge',edge),('face',face)]:
        for error in ['FP','FN']:
            assert sum(x['kind']==kind and x['error']==error for x in errors)==counts[error.lower()]
    write(ROOT/f'errors-update{step:04d}.json',dict(update=step,cumulative_update=4500+step,uid=UID,
          threshold=0,vertex_ids='local zero-based indices of the saved pool vertices',
          face_training_pool='positive + mixed; unchanged effective objective pool',errors=errors))
    np.savez_compressed(ROOT/f'candidate-snapshot-update{step:04d}.npz',
        edge_pairs=pairs_cpu,edge_labels=ey,edge_logits=el,
        face_actual_triples=at,face_actual_labels=ay,face_actual_logits=al,face_actual_in_training_pool=in_pool,
        face_gt_triples=gt,face_gt_logits=gt_logits,face_gt_in_actual_candidates=covered,
        face_train_keys=train_face_keys)
    return dict(edge=edge,face=face,actual_face_candidates=nc,gt_face_candidates=int(covered.sum()),
                missing_gt_face_candidates=int((~covered).sum()),
                min_margin=dict(edge_gt=float(el[ey].min()),edge_non_gt=float((-el[~ey]).min()),
                                face_gt_including_missing=float(gt_logits.min()),face_gt_actual=fmin_pos,face_non_gt_actual=fmin_neg),
                perfect=edge['fp']==edge['fn']==face['fp']==face['fn']==0)

source_files={}
for module in list(sys.modules.values()):
    path=getattr(module,'__file__',None)
    if path and path.endswith('.py') and str(BASE) in path and Path(path).is_file():
        src=Path(path);dest=ROOT/'source_archive'/src.relative_to(BASE)
        if ROOT not in src.parents:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            source_files[str(src)]=c.m.probe.digest(src)
manifest=dict(experiment='latent512_804_lr03_continue500',branch=BRANCH,
              parent_checkpoint=str(PARENT_CHECKPOINT),parent_sha256=PARENT_SHA,parent_completed_updates=4500,
              lr={pg['name']:pg['lr'] for pg in optimizer.param_groups},seed=SEED,initialization='exact parent weights and four Adam groups restored', 
              checkpoint_migration=False,uids=[UID],vertex_count=804,edge_count=2406,face_count=1604,
              latent_dim=512,decoder_hidden_dim=1024,edge_face_dim=32,parameter_count=sum(p.numel() for p in model.parameters()),
              args=args,backend='math00; deterministic Graph forward/backward; FP32 attention E+D',
              input='archived vertex XYZ and face centroid XYZ with incidence graph',
              loss='fully-diff Edge Soft4 + Face Soft4; no outer /4; inner /4; eps1e-8; tau1',
              sampling=False,kl_in_objective=False,logvar='frozen; excluded from optimizer',
              optimizer='all four parent Adam states and LR restored unchanged',betas=[.9,.999],eps=1e-8,weight_decay=0,clip=1,
              warmup=None,
              budget=500,checks=CHECKS,scoring=scales,pair_chunk=c.PAIR_CHUNK,
              pool_sha256=c.m.probe.digest(pool_path),source_sha256=source_files,
              groups={g:list(ps) for g,ps in groups.items()},
              failure_policy='Stop on nonfinite loss/logits/representations/gradients/updates. Retain checkpoints and failure state. Occasional finite loss or error increases do not trigger changes.')
write(ROOT/'manifest.json',manifest)
write(ROOT/'status.json',dict(state='preflight',completed_updates=4500,additional_updates=0))
evaluations=[]
def evaluate(step,checkpoint_sha):
    rows=forward()
    loss,parts,_=objective(rows)
    detached=tuple(tuple(x.detach() for x in xs) for xs in rows)
    del rows
    metrics=actual_metrics(detached, step)
    rec=dict(update=step,cumulative_update=4500+step,loss=float(loss.detach()),parts=parts,**metrics,checkpoint_sha256=checkpoint_sha,
             scales=dict(mu=scalar_scale(detached[0][0]),hidden=scalar_scale(hidden[-1]),
                         edge=scalar_scale(detached[2][0]),face=scalar_scale(detached[3][0])))
    del loss,detached
    if step == 0:
        parent_eval=json.loads((PARENT/'eval-update1000.json').read_text())
        for key in ['loss','parts','edge','face','actual_face_candidates','gt_face_candidates',
                    'missing_gt_face_candidates','min_margin','scales']:
            assert rec[key] == parent_eval[key], ('parent baseline mismatch',key)
        assert T.equal(T.get_rng_state(),cp['torch_rng'])
        assert all(T.equal(x,y) for x,y in zip(T.cuda.get_rng_state_all(),cp['cuda_rng']))
        write(ROOT/'parent_baseline_verified.json',dict(weights_exact=True,adam_moments_steps_exact=True,
              baseline_loss_counts_margins_scales_exact=True,rng_restored_and_unchanged_by_evaluation=True,
              parent_sha256=PARENT_SHA,lr={pg['name']:pg['lr'] for pg in optimizer.param_groups}))
    write(ROOT/f'eval-update{step:04d}.json',rec);evaluations.append(rec)
    print(json.dumps(dict(event='evaluation',**rec)),flush=True)

completed=0
try:
    rows=forward(); l0,parts0,s0=objective(rows)
    baseline=tuple(tuple(x.detach().clone() for x in xs) for xs in rows)
    del rows,l0
    rows=forward(); l0,parts1,s1=objective(rows)
    assert all(T.equal(x,y) for xs,ys in zip(rows,baseline) for x,y in zip(xs,ys))
    assert parts0 == parts1 and all(np.array_equal(s0[0][k],s1[0][k]) for k in s0[0])
    del rows,l0,baseline,s0,s1
    write(ROOT/'preflight.json',dict(repeated_forward_bitwise=True,adam_states=len(optimizer.state),parent_adam_step=4500,
            mu_shape=[804,512],latent_input_shape=[1024,512],head_shapes=[32,1024],parent_weights_loaded=True))
    sha=save(0);evaluate(0,sha)
    with (ROOT/'updates.jsonl').open('x',buffering=1) as log:
        for step in range(1,501):
            start=time.monotonic()
            for pg in optimizer.param_groups:
                assert pg['lr']==PARENT_LR[pg['name']]
            optimizer.zero_grad(set_to_none=True)
            rows=forward()
            loss,parts,saved=objective(rows)
            assert T.isfinite(loss), 'nonfinite loss'
            rep={k:scalar_scale(v) for k,v in [('mu',rows[0][0]),('hidden',hidden[-1]),('edge',rows[2][0]),('face',rows[3][0])]}
            assert all(np.isfinite(v).all() for v in saved[0].values()),'nonfinite logits'
            loss.backward()
            grads={g:norm(p.grad for p in ps.values() if p.grad is not None) for g,ps in groups.items()}
            assert all(np.isfinite(v) for v in grads.values()), 'nonfinite gradient'
            channel=mu_actual[0].grad.detach().double().square().sum(0).sqrt()
            interface=dict(mu_gradient_nonzero_channels=int((channel>0).sum()),
                           mu_gradient_tail64_511_l2=float(channel[64:].norm()),
                           mu_weight_tail_gradient=norm([a.mu.weight.grad[64:]]),
                           decoder_input_tail_gradient=norm([a.latent_input.weight.grad[:,64:]]))
            assert interface['mu_gradient_nonzero_channels']==512
            total=float(T.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
            old={g:[p.detach().clone() for p in ps.values()] for g,ps in groups.items()}
            optimizer.step();completed=step
            updates={}
            for g,ps in groups.items():
                base_norm=norm(old[g]);delta=norm(p.detach()-v for p,v in zip(ps.values(),old[g]))
                assert np.isfinite(delta), 'nonfinite update'
                updates[g]=dict(parameter_l2=base_norm,delta_l2=delta,relative_l2=delta/(base_norm+1e-30))
            del old
            record=dict(update=step,cumulative_update=4500+step,edge_soft4=parts[0]['edge'],face_soft4=parts[0]['face'],loss=float(loss.detach()),
                        edge=c.metrics(ey,saved[0]['edge_train_logits']),face_training_pool=c.metrics(fy,saved[0]['face_train_logits']),
                        gradient_norms=grads,global_grad_norm=total,clip_coefficient=min(1.,1/(total+1e-6)),
                        lr={pg['name']:pg['lr'] for pg in optimizer.param_groups},actual_updates=updates,
                        scales=rep,interface=interface,head_weight_norms=dict(edge=norm([a.edge_embedding.weight]),face=norm([a.face_embedding.weight])),
                        seconds=time.monotonic()-start,metrics_parameter_point='before this update')
            log.write(json.dumps(record,allow_nan=False)+'\n')
            del rows,loss,saved,channel
            write(ROOT/'status.json',dict(state='training',completed_updates=4500+completed,additional_updates=completed,last_update_seconds=record['seconds']))
            if step<=3 or step%25==0:print(json.dumps(dict(event='update',update=step,edge=parts[0]['edge'],face=parts[0]['face'],seconds=record['seconds'])),flush=True)
            if step in CHECKS:evaluate(step,save(step))
    write(ROOT/'backend_audit.json',b.AUDIT)
    result=dict(state='completed',completed_updates=4500+completed,additional_updates=completed,logvar_unchanged=True,
                perfect_checks=[r['update'] for r in evaluations if r['perfect']],
                last_three_checks_perfect=all(next(r for r in evaluations if r['update']==s)['perfect'] for s in [300,400,500]),
                final=evaluations[-1])
    write(ROOT/'completion.json',result);write(ROOT/'status.json',result)
    print('COMPLETE',json.dumps(result),flush=True)
except BaseException as exc:
    failure=dict(state='failed',completed_updates=4500+completed,additional_updates=completed,error=str(exc),traceback=traceback.format_exc())
    write(ROOT/'failure.json',failure);write(ROOT/'status.json',failure)
    try:save(completed,'failure-state.pt')
    except BaseException:pass
    raise
