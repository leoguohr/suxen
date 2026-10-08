"""Lock stratified100 and reproduce the existing frozen Face pool preparation rule."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from pathlib import Path
import csv,json,hashlib,sys,importlib.util,time
import inspect
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=Path('/guohaoran/nexus_fast_track')
NEG=BASE/'data_2k_v1/runs/independent_reimplementation_v2_2k/derived/topology_negative_candidates_v2'
SEED=20260915
BANDS=[(4,255,17),(256,511,17),(512,1023,17),(1024,1535,17),(1536,2047,16),(2048,2600,16)]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvwrite(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

assert not (ROOT/'READY.json').exists(),'Already prepared; do not alter the frozen selection/pools'
summary=json.loads((NEG/'summary.json').read_text());source=Path(summary['source_manifest'])
assert sha(source)==summary['source_manifest_sha256']
allrows=list(csv.DictReader(source.open()));source_by={x['uid']:x for x in allrows}
negative_rows=list(csv.DictReader((NEG/'candidate_manifest.csv').open()));neg_by={x['uid']:x for x in negative_rows}
eligible=[x for x in negative_rows if x['split']=='train' and 4<=int(x['vertex_count'])<=2600]
if not (ROOT/'selection.json').exists():
    rng=np.random.default_rng(SEED);selected=[];strata=[]
    for lo,hi,count in BANDS:
        candidates=sorted([x['uid'] for x in eligible if lo<=int(x['vertex_count'])<=hi])
        selected.extend(rng.choice(candidates,count,replace=False).tolist())
        strata.append(dict(min_vertices=lo,max_vertices=hi,available=len(candidates),selected=count))
    assert len(set(selected))==100
    selected=sorted(selected,key=lambda u:(int(neg_by[u]['vertex_count']),u))
    write(ROOT/'selection.json',dict(seed=SEED,source_manifest=str(source),source_sha256=sha(source),
          uids=selected,strata=strata,rule='fixed size stratification, before any model evaluation',
          existing20_overlap=sorted(set(selected)&{x['uid'] for x in csv.DictReader((BASE/'mini_nexus/experiments/topology_ae_overfit20_manifest.csv').open())})))
    csvwrite(ROOT/'excluded.csv',[dict(uid=x['uid'],vertices=x['vertex_count'],reason='validation_split' if x['split']!='train' else 'outside_predeclared_4_to_2600_vertices') for x in negative_rows if x not in eligible])
sel=json.loads((ROOT/'selection.json').read_text());uids=sel['uids']
csvwrite(ROOT/'data_manifest.csv',[source_by[u] for u in uids])
spec=importlib.util.spec_from_file_location('pool_reference',BASE/'diagnostics/aggressive_topology_probe_20260907/experiment.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
# Resource guard only: preserve every arithmetic and enumeration operation of the archived miner.
# 100M keys/scores/triples fit this host's available RAM; do not truncate or change sample selection.
graph_source=inspect.getsource(p.graph)
assert graph_source.count('total > 5_000_000')==1
graph_source=graph_source.replace('total > 5_000_000','total > 100_000_000')
exec(compile(graph_source,'<same-miner-extended-resource-guard>','exec'),p.__dict__)
p.torch.set_num_threads(1);p.torch.cuda.set_device(0)
p.torch.backends.cuda.matmul.allow_tf32=False;p.torch.backends.cudnn.allow_tf32=False
ds=p.Nexus2KManifestDataset(ROOT/'data_manifest.csv','train')
samples={};rows=[]
for u in uids:
    sample=ds[ds.index_for_uid(u)];samples[u]=sample
    nr=neg_by[u];v=sample.vertices.numpy();f=sample.face_set.numpy();e=sample.edge_index.numpy();n=len(v)
    assert n==int(nr['vertex_count']) and len(f)==int(nr['positive_face_count']) and e.shape[1]==int(nr['positive_edge_count'])
    assert np.isfinite(v).all() and (f>=0).all() and (f<n).all()
    assert len(np.unique(p.keys(f,n)))==len(f)
    expected=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[0,2]],f[:,[1,2]]]),axis=1),axis=0)
    assert np.array_equal(expected,e.T)
    data_sha={k:sha(source_by[u][k]) for k in ['condition_point','mesh_quantized_training','octree','topology']}
    assert data_sha['mesh_quantized_training']==nr['source_mesh_sha256'] and data_sha['topology']==nr['source_topology_sha256']
    assert sha(BASE/nr['candidate_path'])==nr['candidate_sha256']
    rows.append(dict(uid=u,mesh_path=source_by[u]['mesh_quantized_training'],mesh_sha256=data_sha['mesh_quantized_training'],
        topology_path=source_by[u]['topology'],topology_sha256=data_sha['topology'],vertices=n,gt_edges=len(expected),gt_faces=len(f),
        edge_pairs=n*(n-1)//2,data_hashes_json=json.dumps(data_sha,sort_keys=True),data_validation='passed'))
    print('DATA_VALID',u,n,len(f),flush=True)
write(ROOT/'data_validation.json',dict(all100_passed=True,rows=rows))

# This old checkpoint is ONLY a frozen offline negative miner, never the fresh training initializer.
cp=p.torch.load(p.CHECKPOINT,map_location='cpu',mmap=True,weights_only=False)
miner=p.load_topology_system_from_checkpoint(cp,device='cuda');sc=miner.scoring_contract()
from mini_nexus.negative_candidates import TopologyNegativeCandidateStore
miner.negative_candidate_store=TopologyNegativeCandidateStore(NEG,'mixed_medium')
miner.fixed_overfit_face_negatives=True
(ROOT/'pools').mkdir(exist_ok=True)
pool_records=[]
for row in rows:
    u=row['uid'];s=samples[u];n=row['vertices'];path=ROOT/'pools'/f'{u}_pool.npz'
    old=BASE/'diagnostics/math00_twenty_mesh_training_preparation/pools'/path.name
    reused=old.exists()
    if not path.exists():
        if reused:
            import shutil
            shutil.copy2(old,path)
        else:
            neg=miner._select_face_negatives(uid=u,vertex_count=n,positive_edges=s.edge_index,positive_faces=s.face_set,
                negative_seed=cp['args']['seed'],fixed_negatives=True,sample_seed_offset=0)
            orig=neg.faces.cpu().numpy();pos=s.face_set.numpy();cycle=neg.face_source_counts['cycle']
            batch=p.collate_packed_topology([s]).to('cuda');p.UIDS=[u]
            preds=[];hard=[]
            for mode in ['mu','sample0']:
                with p.torch.no_grad():
                    outputs=p.get_rows(miner,batch,mode,1000)
                    k,_,total=p.graph(outputs[2][0],s.edge_index.numpy(),sc)
                    if k is None:raise RuntimeError(f'Frozen miner pool exceeds 100M resource bound for {u}/{mode}: {total}; stop without reselection')
                    false=k[~np.isin(k,p.keys(pos,n))]
                    scores=p.score_faces(outputs[3][0],p.triples(false,n),sc)
                    preds.append(false);hard.append(false[scores>0])
                del outputs
            union=np.unique(np.concatenate(preds));hard=np.unique(np.concatenate(hard));hard=hard[~np.isin(hard,p.keys(orig,n))]
            rng=np.random.default_rng(p.stable_uid_seed(20260907,1000,u));hard=rng.permutation(hard)
            heldout=hard[:len(hard)//5];eligible_hard=hard[len(hard)//5:];rest=len(orig)-cycle
            replace=min(rest//2,len(eligible_hard));chosen=eligible_hard[:replace]
            keep=rng.permutation(np.arange(cycle,len(orig)))[:rest-replace]
            mixed=np.concatenate([orig[:cycle],orig[keep],p.triples(chosen,n)])
            np.savez_compressed(path,vertices=s.vertices.numpy(),positive=pos,edges=s.edge_index.numpy(),original=orig,
                                mixed=mixed,pool=p.triples(union,n),heldout=p.triples(heldout,n))
            del batch
    d=np.load(path);pos=d['positive'];mixed=d['mixed']
    assert np.array_equal(d['vertices'],s.vertices.numpy()) and np.array_equal(pos,s.face_set.numpy())
    assert np.array_equal(d['edges'],s.edge_index.numpy())
    assert mixed.ndim==2 and mixed.shape[1]==3 and ((mixed>=0)&(mixed<n)).all()
    assert (np.diff(np.sort(mixed,axis=1),axis=1)>0).all()
    assert len(np.unique(p.keys(mixed,n)))==len(mixed) and not np.intersect1d(p.keys(pos,n),p.keys(mixed,n)).size
    row.update(face_pool_size=len(pos)+len(mixed),face_negatives=len(mixed),pool_path=str(path),pool_sha256=sha(path),candidate_validation='passed',reused_existing_pool=reused)
    pool_records.append(dict(uid=u,pool_sha256=sha(path),negative_count=len(mixed),reused_existing_pool=reused))
    print('POOL_READY',u,n,len(mixed),flush=True)
csvwrite(ROOT/'overfit100_manifest.csv',rows)
write(ROOT/'pool_provenance.json',dict(rule='unchanged frozen aggressive-probe preparation: all false GT cycles + fixed wedge/uniform, replace up to half noncycle slots by fixed-miner hard negatives; no online mining',
    fixed_miner_checkpoint=str(p.CHECKPOINT),fixed_miner_sha256=sha(p.CHECKPOINT),mining_code_sha256=sha(p.__file__),records=pool_records,
    resource_guard_candidates=100000000,prepare_script_sha256=sha(__file__),
    caveat='Face training pool is not the actual predicted-edge candidate set. Miner weights are never used as training initialization.'))
write(ROOT/'READY.json',dict(all100_valid=True,manifest_sha256=sha(ROOT/'overfit100_manifest.csv'),data_manifest_sha256=sha(ROOT/'data_manifest.csv'),
    selection_sha256=sha(ROOT/'selection.json'),pool_provenance_sha256=sha(ROOT/'pool_provenance.json'),
    vertices=dict(min=min(x['vertices'] for x in rows),max=max(x['vertices'] for x in rows),median=float(np.median([x['vertices'] for x in rows]))),
    total_edge_pairs=sum(x['edge_pairs'] for x in rows),total_gt_faces=sum(x['gt_faces'] for x in rows),training_started=False))
print('PREPARATION_COMPLETE',flush=True)
