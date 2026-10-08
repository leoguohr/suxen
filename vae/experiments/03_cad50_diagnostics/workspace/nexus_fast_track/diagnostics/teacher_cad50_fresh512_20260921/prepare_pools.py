"""Existing GT-based fixed-overfit rule, regenerated for CAD50; no learned miner."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import importlib.util,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
from loader import load_all,sha
R=Path(__file__).resolve().parent;BASE=Path('/guohaoran/nexus_fast_track')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
builder_path=BASE/'data_2k_v1/scripts/build_topology_negative_candidates.py'
selector_path=BASE/'diagnostics/layernorm_no_rms_20260907_0824/variant_project/mini_nexus/negative_candidates.py'
builder=module('cad_candidate_builder',builder_path);selector=module('cad_candidate_selector',selector_path)
manifest,samples=load_all(R/'data');seed=0
(R/'pools').mkdir(exist_ok=True);(R/'candidate_sources').mkdir(exist_ok=True)
assert not (R/'pool_manifest.json').exists(),'Refuse to change a fixed pool'
records=[]
for uid,s in samples.items():
    n=len(s['vertices']);nf=len(s['face_set']);item_seed=builder.stable_seed(uid,seed)
    arrays,counts=builder.build_candidate_arrays(s['vertices'],s['faces'],s['edge_index'],s['face_set'],seed=item_seed,knn_k=16)
    source=R/'candidate_sources'/f'{uid}.npz';np.savez_compressed(source,**arrays)
    # Reuse the actual fixed-overfit selector. Only its read-only sidecar is supplied in memory.
    store=selector.TopologyNegativeCandidateStore.__new__(selector.TopologyNegativeCandidateStore)
    store._load=lambda requested,expected=uid,a=arrays:a if requested==expected else None
    selected=store.sample_fixed_overfit_faces(uid,positive_edges=torch.from_numpy(s['edge_index']),
        positive_faces=torch.from_numpy(s['face_set']),vertex_count=n,seed=seed,wedge_ratio=1.,uniform_ratio=.5)
    neg=selected.faces.numpy();pos=s['face_set'];negset={tuple(x) for x in neg.tolist()};posset={tuple(x) for x in pos.tolist()}
    assert len(negset)==len(neg) and not negset.intersection(posset)
    assert all(len(set(x))==3 and 0<=x[0]<x[1]<x[2]<n for x in negset)
    path=R/'pools'/f'{uid}.npz';np.savez_compressed(path,vertices=s['vertices'],edges=s['edge_index'],positive=pos,mixed=neg)
    records.append(dict(uid=uid,path=str(path.relative_to(R)),sha256=sha(path),gt_faces=nf,negatives=len(neg),
        face_pool_size=nf+len(neg),source_counts=selected.face_source_counts,
        requested_wedge=nf,requested_uniform=int(np.ceil(nf*.5)),unique_legal_negatives=n*(n-1)*(n-2)//6-nf,
        source_candidate_counts=counts,source_candidates_sha256=sha(source),seed=item_seed))
    print('POOL',uid,nf,selected.face_source_counts,flush=True)
closure=R/'pool_code';closure.mkdir(exist_ok=True)
for p in [builder_path,selector_path]:shutil.copy2(p,closure/p.name)
result=dict(rule='all non-GT GT-graph 3-cycles + fixed up-to-1F wedge and up-to-0.5F uniform; no repeat padding',
    model_mining=False,old100_pools_used=False,teacher_predictions_used=False,seed=seed,knn_k=16,
    rule_scope='existing GT-based sample_fixed_overfit_faces; excludes old learned mining and subsequent hard-negative rounds by user clarification',
    builder=dict(path=str(builder_path),sha256=sha(builder_path)),selector=dict(path=str(selector_path),sha256=sha(selector_path)),
    data_manifest_sha256=sha(R/'data/manifest.json'),records=records,total_candidates=sum(x['face_pool_size'] for x in records),
    insufficient_sources=[x['uid'] for x in records if x['source_counts']['wedge']<x['requested_wedge'] or x['source_counts']['uniform']<x['requested_uniform']])
(R/'pool_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print('TOTAL',result['total_candidates'],flush=True)
