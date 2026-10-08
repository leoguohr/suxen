"""Frozen512 architecture/math00/Soft4; fresh initialization only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import sys,json,random,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=Path('/guohaoran/nexus_fast_track')
REF=BASE/'diagnostics/math00_four_mesh_lr10x_20260913'
OLD=BASE/'diagnostics/math00_latent512_804_fresh_20260914'
sys.path.insert(0,str(REF))
old_write=Path.write_text
def redirected(path,*a,**kw):
    if path==REF/'loss_changes.txt':path=ROOT/'loss_changes_exact_runtime.txt'
    return old_write(path,*a,**kw)
Path.write_text=redirected
try:import backend00 as b
finally:Path.write_text=old_write
T,np,c=b.torch,b.np,b.c
SEED=20260915
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(p)
def sha(p):return c.m.probe.digest(p)
def norm(tensors):
    return float(sum((t.detach().double().square().sum() for t in tensors),T.zeros((),device='cuda',dtype=T.float64)).sqrt())
def tensor_hash(items):
    h=hashlib.sha256()
    for name,t in items:
        h.update(name.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def setup():
    T.set_num_threads(1);T.cuda.set_device(0)
    T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False;T.set_float32_matmul_precision('highest')
    random.seed(SEED);np.random.seed(SEED);T.manual_seed(SEED);T.cuda.manual_seed_all(SEED)
    meta=json.loads((OLD/'manifest.json').read_text())
    for path,digest in meta['source_sha256'].items():assert sha(path)==digest,path
    ready=json.loads((ROOT/'READY.json').read_text())
    assert ready['all100_valid']
    for name,key in [('overfit100_manifest.csv','manifest_sha256'),('data_manifest.csv','data_manifest_sha256'),('selection.json','selection_sha256'),('pool_provenance.json','pool_provenance_sha256')]:assert sha(ROOT/name)==ready[key],name
    args=json.loads((ROOT/'construction_args.json').read_text())
    args.update(manifest=str(ROOT/'data_manifest.csv'),seed=SEED,steps=5000,expected_samples=100,max_packed_meshes=1,
                output=str(ROOT),calibrate_logit_scales=False,diagnostic_mean_latent=True)
    from mini_nexus.flash_varlen_topology import FlashVarlenNexus2KTopologyAESystem
    model=FlashVarlenNexus2KTopologyAESystem.from_saved_args(args).cuda().float().eval()
    a=model.autoencoder
    assert a.mu.weight.shape==(512,512) and a.log_variance.weight.shape==(512,512)
    assert a.latent_input.weight.shape==(1024,512) and a.edge_embedding.weight.shape==a.face_embedding.weight.shape==(32,1024)
    model.requires_grad_(True);a.log_variance.requires_grad_(False)
    named=dict(a.named_parameters())
    groups={g:{n:named[n] for n in names} for g,names in meta['groups'].items() if g!='logvar'}
    assert {id(p) for g in groups.values() for p in g.values()}=={id(p) for p in model.parameters() if p.requires_grad}
    opt=T.optim.Adam([dict(params=list(g.values()),name=n,lr=1e-6 if n=='encoder_mu' else 1e-5) for n,g in groups.items()],betas=(.9,.999),eps=1e-8,weight_decay=0)
    assert not opt.state
    b.ROOT=b.backend.ROOT=ROOT;(ROOT/'C_graph_only').mkdir(exist_ok=True)
    c.m.install_clamp(model,-20.)
    source=(OLD/'sampling_forward.py').read_text();(ROOT/'sampling_forward.py').write_text(source)
    exec(compile(source,str(ROOT/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__)
    b.install_deterministic(model)
    for section,blocks in [('encoder',a.encoder_blocks),('decoder',a.decoder_blocks)]:
        for i,block in enumerate(blocks):
            for name,module in block.named_modules():
                if isinstance(module,T.nn.MultiheadAttention):b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
    assert len(b.ATTENTION_NAMES)==28
    c.PAIR_CHUNK=args['pair_chunk_size']
    uids=json.loads((ROOT/'selection.json').read_text())['uids'];assert len(set(uids))==100
    ds=c.m.probe.Nexus2KManifestDataset(ROOT/'data_manifest.csv','train');samples={};pools={}
    records=json.loads((ROOT/'pool_provenance.json').read_text())['records']
    for u,record in zip(uids,records):
        assert u==record['uid'];path=ROOT/'pools'/f'{u}_pool.npz';assert sha(path)==record['pool_sha256']
        d=np.load(path);pools[u]={k:d[k] for k in ['vertices','edges','positive','mixed']};samples[u]=ds[ds.index_for_uid(u)]
        assert np.array_equal(samples[u].vertices.numpy(),pools[u]['vertices'])
    capture={}
    def mu_hook(module,inputs,out):
        if out.requires_grad:out.retain_grad()
        capture['mu']=out
    a.mu.register_forward_hook(mu_hook)
    a.decoder_output_norm.register_forward_hook(lambda module,inputs,out:capture.update(hidden=out.detach()))
    def forward(uid):
        capture.clear();batch=c.m.probe.collate_packed_topology([samples[uid]]).to('cuda')
        rows,stats,zs=c.m.forward(model,batch,'mu')
        assert rows[0][0].shape==(len(pools[uid]['vertices']),512) and T.equal(rows[0][0],zs[0])
        return rows
    return model,opt,groups,uids,pools,forward,capture,args

