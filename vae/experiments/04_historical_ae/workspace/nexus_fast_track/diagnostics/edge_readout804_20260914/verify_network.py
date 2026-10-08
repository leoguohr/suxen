"""Check the offline readout in an in-memory original model, then restore it."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import hashlib
import json
import sys
from pathlib import Path

BASE=Path('/guohaoran/nexus_fast_track')
REF=BASE/'diagnostics/math00_four_mesh_lr10x_20260913'
BRANCH=REF/'only804_mu'
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'network_verification'
OUT.mkdir(exist_ok=True)
SNAP=BASE/'diagnostics/only804_mu_review_20260914/package/C_snapshot804_step1000'
sys.path.insert(0,str(REF))
old_write=Path.write_text
def redirect(path,*args,**kwargs):
    if path==REF/'loss_changes.txt':path=OUT/'loss_changes_exact_runtime.txt'
    return old_write(path,*args,**kwargs)
Path.write_text=redirect
try:import backend00 as b
finally:Path.write_text=old_write
T,np,c=b.torch,b.np,b.c
T.set_num_threads(1);T.cuda.set_device(0)
T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False
T.set_float32_matmul_precision('highest')
checkpoint=BRANCH/'checkpoint-update1000.pt'
meta=json.loads((BRANCH/'manifest.json').read_text())
snap_summary=json.loads((SNAP/'summary.json').read_text())
for path,expected in meta['source_sha256'].items():assert c.m.probe.digest(path)==expected,path
assert c.m.probe.digest(REF/'backend00.py')==meta['backend_sha']
checkpoint_sha=c.m.probe.digest(checkpoint)
assert checkpoint_sha==snap_summary['checkpoint_sha256']
c.UIDS=meta['selected_uids'];c.m.UIDS=c.UIDS;c.m.probe.UIDS=c.UIDS
cp,model,batch=c.m.probe.setup_model(checkpoint)
assert cp['additional_updates']==1000 and cp['completed_updates']==4400
b.ROOT=OUT;b.backend.ROOT=OUT;(OUT/'C_graph_only').mkdir(exist_ok=True)
c.m.install_clamp(model,-20.)
runtime=(BRANCH/'sampling_forward.py').read_text()
exec(compile(runtime,str(OUT/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__)
(OUT/'sampling_forward.py').write_text(runtime)
rng=T.Generator(device='cuda');rng.set_state(cp['training_rng_state'])
model.autoencoder.diagnostic_train_rng=rng
model.autoencoder.diagnostic_train_draws=cp['training_noise_draws']
b.install_deterministic(model)
for section,blocks in [('encoder',model.autoencoder.encoder_blocks),('decoder',model.autoencoder.decoder_blocks)]:
    for i,block in enumerate(blocks):
        for name,module in block.named_modules():
            if isinstance(module,T.nn.MultiheadAttention):b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
assert len(b.ATTENTION_NAMES)==28
# Preserve the archived forward's grad-mode dispatch, but never call backward or an optimizer.
model.requires_grad_(True);model.autoencoder.log_variance.requires_grad_(False)
a=model.autoencoder
saved=np.load(SNAP/'representations_and_gradients.npz')
edge_data=np.load(SNAP/'edge_all_pairs.npz')
fit=np.load(ROOT/'offline/readout.npz')
readout=T.load(ROOT/'offline/edge_head_candidate.pt',map_location='cpu')
old_weight=a.edge_embedding.weight.detach().clone()
old_bias=a.edge_embedding.bias.detach().clone()
def thash(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
hashes={n:thash(p) for n,p in model.named_parameters()}
rng_before=rng.get_state().clone()
cuda_before=T.cuda.get_rng_state().clone()
counts=[int(x) for x in batch.vertex_mask.sum(1).tolist()]
index=c.UIDS.index('nexus_2k_001333');offset=sum(counts[:index]);n=counts[index]
pairs=T.from_numpy(edge_data['pairs']).cuda();y=T.from_numpy(edge_data['labels']).cuda()
scales=model.scoring_contract()

def forward():
    capture=[]
    hook=a.decoder_output_norm.register_forward_hook(lambda mod,args,out:capture.append(out.detach().clone()))
    try:rows,stats,zs=c.m.forward(model,batch,'mu')
    finally:hook.remove()
    h=capture[0][offset:offset+n].detach()
    e=rows[2][index].detach().clone();mu=rows[0][index].detach().clone();face=rows[3][index].detach().clone()
    del rows,zs,capture
    with T.no_grad():
        logits=c.edge_logits(e,pairs,scales)
        num,mass=b.full_sums(logits,y);loss=(num/(mass+1e-8)).mean()
        pred=logits>0;tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
        metric=dict(tp=tp,fp=fp,fn=fn,tn=int((~pred&~y).sum()),f1=2*tp/(2*tp+fp+fn),
                    edge_soft4=float(loss),min_margin_gt=float(logits[y].min()),
                    min_margin_non_gt=float((-logits[~y]).min()),perfect=fp==0 and fn==0)
    return dict(h=h,e=e,mu=mu,face=face,logits=logits,metrics=metric)

baseline=forward()
assert np.array_equal(baseline['h'].cpu().numpy(),saved['decoder_hidden_after_final_ln'])
assert np.array_equal(baseline['e'].cpu().numpy(),saved['edge_embedding_scoring'])
assert np.array_equal(baseline['logits'].cpu().numpy(),edge_data['logits'])
print('BASELINE_EXACT',json.dumps(baseline['metrics']),flush=True)
try:
    with T.no_grad():
        a.edge_embedding.weight.copy_(readout['weight'])
    assert T.equal(a.edge_embedding.bias,old_bias)
    fitted=forward();repeated=forward()
    assert all(T.equal(fitted[k],repeated[k]) for k in ['h','e','mu','face','logits'])
    assert all(T.equal(fitted[k],baseline[k]) for k in ['h','mu','face'])
    assert fitted['metrics']['perfect']
    changed=[n for n,p in model.named_parameters() if thash(p)!=hashes[n]]
    assert len(changed)==1 and changed[0].endswith('edge_embedding.weight'),changed
    np.savez_compressed(OUT/'actual_network_readout.npz',hidden=fitted['h'].cpu().numpy(),
                        embedding=fitted['e'].cpu().numpy(),logits=fitted['logits'].cpu().numpy(),
                        pairs=edge_data['pairs'],labels=edge_data['labels'])
    actual=fitted['e'].cpu().double();target=T.from_numpy(fit['target']).double()
    offline=T.from_numpy(fit['fit_fp32']).double()
    result=dict(baseline=baseline['metrics'],fitted=fitted['metrics'],
                baseline_hidden_embedding_logits_match_export_exactly=True,
                hidden_mu_face_embedding_unchanged=True,changed_parameter_names=changed,
                repeated_forward_bitwise_equal=True,original_bias_preserved=True,
                packed_uids=c.UIDS,packed_vertex_counts=counts,
                target_relative_frobenius=float((actual-target).norm()/target.norm()),
                offline_embedding_max_abs_difference=float((actual-offline).abs().max()),
                offline_logits_max_abs_difference=float(np.max(np.abs(fitted['logits'].cpu().numpy()-fit['logits_fp32']))),
                optimizer_created=False,optimizer_updates=0,
                other_meshes_and_actual_faces='not evaluated for success; no joint reconstruction claim')
finally:
    with T.no_grad():a.edge_embedding.weight.copy_(old_weight)
assert all(thash(p)==hashes[n] for n,p in model.named_parameters())
restored=forward()
assert all(T.equal(restored[k],baseline[k]) for k in ['h','e','mu','face','logits'])
assert T.equal(rng.get_state(),rng_before) and T.equal(T.cuda.get_rng_state(),cuda_before)
assert c.m.probe.digest(checkpoint)==checkpoint_sha
result.update(all_parameters_restored=True,restored_forward_exact=True,rng_unchanged=True,
              original_checkpoint_sha256=checkpoint_sha,original_checkpoint_unchanged=True)
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPLETE',json.dumps(result),flush=True)
