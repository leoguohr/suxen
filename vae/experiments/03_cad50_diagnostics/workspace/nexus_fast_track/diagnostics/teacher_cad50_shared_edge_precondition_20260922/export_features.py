"""One real B2500 forward per mesh; cache detached readout inputs, zero updates."""
import os,sys,json,subprocess,fcntl
from pathlib import Path
N=Path(__file__).resolve().parent
GPU='GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
lock=(N/'export.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (N/'repro_outputs/FEATURE_MANIFEST.json').exists(),'Already exported; do not overwrite'
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
assert GPU not in processes,'Assigned GPU occupied; no preemption'
sys.path.insert(0,str(N/'export_runtime'))
from runtime import ROOT,BASE,T,np,c,setup,sha,tensor_hash
from helpers import SOURCE,SOURCE_SHA,PARENT,rng,restore_rng,equal
from evaluate import evaluate_mesh
assert sha(SOURCE)==SOURCE_SHA
model,opt,groups,uids,pools,forward,objective,capture,args,batches=setup(False)
cp=T.load(SOURCE,map_location='cpu',mmap=True,weights_only=False)
assert cp['completed_updates']==2500
model.load_state_dict(cp['model'],strict=True);opt.load_state_dict(cp['optimizer']);restore_rng(cp['rng'])
assert equal(model.state_dict(),cp['model']) and equal(opt.state_dict(),cp['optimizer']) and equal(rng(),cp['rng'])
assert sha(ROOT/'data/manifest.json')==cp['config']['parent']['data_manifest_sha256']
assert sha(ROOT/'pool_manifest.json')==cp['config']['parent']['pool_manifest_sha256']
parent=json.loads((PARENT/'B_lr03/eval-new0500.json').read_text())
assert [x['uid'] for x in parent['meshes']]==uids
assert sum(len(p['vertices']) for p in pools.values())==2872
scales=model.scoring_contract();before=tensor_hash(model.named_parameters());records=[]
for uid,reference in zip(uids,parent['meshes']):
    rows=forward(uid);hidden=capture['hidden'];n=len(pools[uid]['vertices'])
    assert hidden.shape==(n,1024) and hidden.dtype==T.float32
    projected=model.autoencoder.edge_embedding(hidden)
    centered=projected-projected.mean(dim=0,keepdim=True)
    assert T.equal(centered,rows[2][0]),uid
    pair=T.triu_indices(n,n,1,device='cuda').T
    with T.no_grad():logits=c.edge_logits(rows[2][0].detach(),pair,scales)
    keys=T.as_tensor(np.unique(pools[uid]['edges'].T@np.array([n,1])),device='cuda')
    labels=T.isin(pair[:,0]*n+pair[:,1],keys)
    positive=logits>0
    counts=dict(tp=int((positive&labels).sum()),fp=int((positive&~labels).sum()),fn=int((~positive&labels).sum()))
    assert all(counts[k]==reference['edge'][k] for k in counts)
    feature=N/'features'/f'{uid}.npz'
    np.savez_compressed(feature,decoder_hidden=hidden.detach().cpu().numpy(),
        original_edge_embedding=rows[2][0].detach().cpu().numpy(),
        original_face_embedding=rows[3][0].detach().cpu().numpy(),
        vertices=pools[uid]['vertices'],local_vertex_indices=np.arange(n,dtype=np.int32),
        edge_pair_ids=pair.cpu().numpy().astype(np.int32),edge_pair_logits=logits.cpu().numpy(),
        edge_pair_gt=labels.cpu().numpy(),gt_edges=np.unique(np.sort(pools[uid]['edges'].T,axis=1),axis=0).astype(np.int32),
        gt_faces=np.sort(pools[uid]['positive'],axis=1).astype(np.int32))
    detached=tuple(tuple(x.detach() for x in group) for group in rows)
    prediction=N/'baseline_predictions'/f'{uid}.npz'
    actual=evaluate_mesh(detached,pools[uid],scales,prediction)
    with np.load(prediction) as now,np.load(PARENT/reference['prediction_path']) as old:
        assert now.files==old.files and all(np.array_equal(now[k],old[k]) for k in now.files),uid
    assert actual['joint_perfect']==reference['joint_perfect']
    records.append(dict(uid=uid,vertices=n,hidden_shape=[n,1024],path=str(feature.relative_to(N)),
        bytes=feature.stat().st_size,sha256=sha(feature),edge=counts,
        real_network_prediction_path=str(prediction.relative_to(N)),prediction_sha256=sha(prediction),
        projection_bitwise_equal=True,parent_prediction_arrays_bitwise_equal=True))
    del rows,detached,hidden,projected,centered,pair,logits,labels;capture.clear()
head=model.autoencoder.edge_embedding
head_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
adam={name:{k:v.detach().cpu().clone() if T.is_tensor(v) else v for k,v in opt.state[param].items()}
      for name,param in head.named_parameters()}
head_file=N/'B2500_edge_head_and_original_adam_rng.pt'
T.save(dict(head=head_state,original_edge_adam_state=adam,
    original_edge_parameter_group=next(g for g in cp['optimizer']['param_groups'] if g['name']=='edge_head'),
    rng=cp['rng'],source=str(SOURCE),source_sha256=SOURCE_SHA,scales=scales),head_file)
np.savez_compressed(N/'B2500_edge_head.npz',weight=head_state['weight'].numpy(),bias=head_state['bias'].numpy())
assert tensor_hash(model.named_parameters())==before and equal(opt.state_dict(),cp['optimizer'])
assert equal(rng(),cp['rng']) and not model.training and not model.autoencoder.training
assert sha(SOURCE)==SOURCE_SHA
assert len(records)==50 and sum(x['vertices'] for x in records)==2872
manifest=dict(source=str(SOURCE),source_sha256=SOURCE_SHA,completed_updates=2500,
    optimizer_updates_in_export=0,model_optimizer_rng_unchanged=True,
    meshes=50,vertices=2872,decoder_hidden_raw_bytes=2872*1024*4,
    feature_files_bytes=sum(x['bytes'] for x in records),records=records,
    head_state=dict(path=str(head_file),bytes=head_file.stat().st_size,sha256=sha(head_file)),
    scales=scales,all50_head_projection_bitwise_equal=True,all50_parent_prediction_arrays_bitwise_equal=True,
    feature_location='decoder_output_norm output; before shared Edge/Face linear heads',
    parameterization='original edge = linear(hidden,W,b), then original per-mesh centering; FP32',
    export_mode='original eval/grad-enabled numeric path, then detach; no backward/optimizer update',
    global_preconditioner_computed=False,head_training_started=False,
    protocol_pending=['prior three-mesh preconditioner formula including regularization and feature weighting','LR and Adam initialization/inheritance rule'],
    environment=dict(torch=T.__version__,cuda=T.version.cuda,cudnn=T.backends.cudnn.version(),gpu_uuid=GPU),
    runtime_code={p.name:sha(p) for p in ROOT.glob('*.py')})
(N/'repro_outputs/FEATURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('EXPORT_PASS',json.dumps({k:v for k,v in manifest.items() if k not in ['records','runtime_code']}),flush=True)
