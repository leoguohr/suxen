"""Current512 full AE, same verified math00; independent float-mesh input."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,random,hashlib,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=Path('/guohaoran/nexus_fast_track')
REF=BASE/'diagnostics/math00_four_mesh_lr10x_20260913'
sys.path.insert(0,str(REF))
old_write=Path.write_text
def redirect(path,*a,**kw):
    return old_write(ROOT/'effective_loss_changes.txt' if path==REF/'loss_changes.txt' else path,*a,**kw)
Path.write_text=redirect
try:import backend00 as b
finally:Path.write_text=old_write
T,np,c=b.torch,b.np,b.c
from loader import sha,load_all,collate
import effective_loss_and_scoring as scoring

def write(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(path)
def tensor_hash(items):
    h=hashlib.sha256()
    for n,t in items:h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def norm(items):
    return float(sum((x.detach().double().square().sum() for x in items),T.zeros((),device='cuda',dtype=T.float64)).sqrt())

def setup():
    T.set_num_threads(1);T.cuda.set_device(0)
    T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False;T.set_float32_matmul_precision('highest')
    random.seed(0);np.random.seed(0);T.manual_seed(0);T.cuda.manual_seed_all(0)
    args=json.loads((ROOT/'construction_args.json').read_text())
    args.update(seed=0,output=str(ROOT),steps=2000,expected_samples=50,max_packed_meshes=1,
        calibrate_logit_scales=False,diagnostic_mean_latent=True,manifest=str(ROOT/'data/manifest.json'),negative_candidate_root=None)
    from mini_nexus.flash_varlen_topology import FlashVarlenNexus2KTopologyAESystem
    model=FlashVarlenNexus2KTopologyAESystem.from_saved_args(args).cuda().float().eval()
    a=model.autoencoder;model.requires_grad_(True);a.log_variance.requires_grad_(False)
    assert a.mu.weight.shape==(512,512) and a.latent_input.weight.shape==(1024,512)
    assert len(a.decoder_blocks)==16 and a.edge_embedding.weight.shape==a.face_embedding.weight.shape==(32,1024)
    assert not a.normalize_spacetime_embeddings
    groups={k:{} for k in ['encoder_mu','decoder','edge_head','face_head']}
    for n,p in a.named_parameters():
        if n.startswith('log_variance.'):assert not p.requires_grad;continue
        if n.startswith(('vertex_input.','face_input.','encoder_blocks.','encoder_output_norm.','mu.')):key='encoder_mu'
        elif n.startswith(('latent_input.','decoder_blocks.','decoder_output_norm.')):key='decoder'
        elif n.startswith('edge_embedding.'):key='edge_head'
        elif n.startswith('face_embedding.'):key='face_head'
        else:raise RuntimeError('Unassigned parameter: '+n)
        groups[key][n]=p
    assert {id(p) for g in groups.values() for p in g.values()}=={id(p) for p in model.parameters() if p.requires_grad}
    opt=T.optim.Adam([dict(params=list(g.values()),name=k,lr=1e-7 if k=='encoder_mu' else 1e-6) for k,g in groups.items()],
        betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert not opt.state
    b.ROOT=b.backend.ROOT=ROOT;(ROOT/'C_graph_only').mkdir(exist_ok=True)
    c.m.install_clamp(model,-20.)
    source=(BASE/'diagnostics/math00_latent512_804_fresh_20260914/sampling_forward.py').read_text()
    (ROOT/'sampling_forward.py').write_text(source)
    exec(compile(source,str(ROOT/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__)
    b.install_deterministic(model)
    T.use_deterministic_algorithms(True)
    for section,blocks in [('encoder',a.encoder_blocks),('decoder',a.decoder_blocks)]:
        for i,block in enumerate(blocks):
            for name,module in block.named_modules():
                if isinstance(module,T.nn.MultiheadAttention):b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
    assert len(b.ATTENTION_NAMES)==28
    c.PAIR_CHUNK=args['pair_chunk_size']
    manifest,samples=load_all(ROOT/'data');uids=manifest['uids'];pools={};batches={};labels={}
    pool_manifest=json.loads((ROOT/'pool_manifest.json').read_text())
    for uid,row in zip(uids,pool_manifest['records']):
        assert row['uid']==uid and sha(ROOT/row['path'])==row['sha256']
        with np.load(ROOT/row['path']) as q:pools[uid]={k:q[k].copy() for k in q.files}
        assert np.array_equal(pools[uid]['vertices'],samples[uid]['vertices'])
        batches[uid]=collate([samples[uid]]).to('cuda')
        n=len(samples[uid]['vertices']);p=pools[uid]
        labels[uid]=dict(edge_keys=T.as_tensor(p['edges'].T@np.array([n,1]),device='cuda'),
            face_ids=T.as_tensor(np.concatenate([p['positive'],p['mixed']]),device='cuda'),
            face_labels=T.as_tensor(np.r_[np.ones(len(p['positive'])),np.zeros(len(p['mixed']))],dtype=T.float32,device='cuda'))
    capture={}
    def capture_mu(module,inputs,output):
        if output.requires_grad:output.retain_grad()
        capture['mu']=output
    a.mu.register_forward_hook(capture_mu)
    a.latent_input.register_forward_pre_hook(lambda m,x:capture.update(z=x[0]))
    a.decoder_output_norm.register_forward_hook(lambda m,x,y:capture.update(hidden=y))
    def forward(uid):
        capture.clear();model.eval()
        rows=model.topology_embedding_rows(batches[uid],sample_seeds=None)
        assert not a.training and T.equal(rows[0][0],capture['z'])
        assert all(x.dtype==T.float32 for group in rows for x in group)
        return rows
    scales=model.scoring_contract()
    def objective(uid):
        rows=forward(uid);d=labels[uid]
        le,_=scoring.soft4_loss(rows[2][0],d['edge_keys'],args['pair_chunk_size'],scales['edge_logit_scale'])
        fl=c.face_logits(rows[3][0],d['face_ids'],scales)
        ns,ms=scoring.soft4_sums(fl,d['face_labels']);lf=(ns/(ms+1e-8)).mean()
        loss=le+lf;assert T.isfinite(loss)
        return rows,loss,dict(edge=float(le.detach()),face=float(lf.detach()))
    return model,opt,groups,uids,pools,forward,objective,capture,args,batches
