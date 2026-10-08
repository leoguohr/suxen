"""Lossless export of the teacher's actual FP32 training targets, not predictions."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import json
from pathlib import Path
import numpy as np
import torch
from loader import sha,collate
R=Path(__file__).resolve().parent
source=R/'teacher_training_cache.pt'
file_manifest=json.loads((R/'source_FILE_MANIFEST.json').read_text())
entry=next(x for x in file_manifest if x['path']=='data/point50/training.pt')
assert sha(source)==entry['sha256']
cache=torch.load(source,map_location='cpu',weights_only=True)
assert len(cache['samples'])==50
out=R/'data';(out/'meshes').mkdir(parents=True,exist_ok=True)
assert not (out/'manifest.json').exists(),'Do not overwrite a locked data export'
records=[];samples=[]
for i,s in enumerate(cache['samples']):
    assert int(s['sample_id'])==i
    v=s['vertices'].numpy();faces=s['faces'].numpy();mapping=s['original_vertex_indices'].numpy()
    assert v.dtype==np.float32 and faces.dtype==np.int64 and np.isfinite(v).all()
    n=len(v);assert np.array_equal(np.sort(mapping),np.arange(n))
    assert faces.ndim==2 and faces.shape[1]==3 and faces.min()>=0 and faces.max()<n
    face_set=np.unique(np.sort(faces,axis=1),axis=0)
    assert len(face_set)==len(faces) and (np.diff(face_set,axis=1)>0).all()
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]]]),axis=1),axis=0).T
    vertex_nodes=faces.reshape(-1);face_nodes=np.repeat(n+np.arange(len(faces)),3)
    incidence=np.stack([np.r_[vertex_nodes,face_nodes],np.r_[face_nodes,vertex_nodes]])
    sample=dict(vertices=v,faces=faces,face_set=face_set,edge_index=edges,
        incidence_index=incidence,original_vertex_indices=mapping)
    uid=f'teacher_cad50_{i:02d}';p=out/'meshes'/f'{uid}.npz'
    np.savez_compressed(p,**sample)
    with np.load(p) as q:
        for key,value in sample.items():assert np.array_equal(q[key],value)
    batch=collate([sample]);assert batch.vertex_mask.all() and torch.equal(batch.vertices[0],s['vertices'])
    assert torch.equal(batch.faces[0],s['faces']) and incidence.shape==(2,6*len(faces))
    records.append(dict(uid=uid,sample_id=i,path=str(p.relative_to(out)),sha256=sha(p),vertices=n,
        gt_edges=edges.shape[1],gt_faces=len(faces),edge_pairs=n*(n-1)//2,unique_coordinate_rows=len(np.unique(v,axis=0))))
    samples.append(sample)
totals={k:sum(x[k] for x in records) for k in ['vertices','gt_edges','gt_faces','edge_pairs']}
assert totals==dict(vertices=2872,gt_edges=8364,gt_faces=5576,edge_pairs=235741)
assert records[0]['vertices']==68 and records[13]['vertices']==16 and max(x['vertices'] for x in records)==274
# Two unequal meshes exercise padding; incidence uses each mesh's local N+face index.
batch=collate([samples[0],samples[13]])
assert batch.vertex_mask.sum(1).tolist()==[68,16] and not batch.vertex_mask[1,16:].any()
assert not batch.vertices[1,16:].count_nonzero() and batch.incidence_index[1].max()==39
manifest=dict(uids=[x['uid'] for x in records],meshes=records,totals=totals,max_vertices=274,
    source_cache_sha256=sha(source),source_archive='nexus_overfit_data_results_no_code.zip',
    supplied_adapter_archive=False,export='new lossless export from actual teacher cache; no teacher model loaded',
    numbering='exact cache numbering; no reorder; original_vertex_indices retained for original-numbering comparisons only',
    transformations=['derive canonical undirected edge/face label sets','derive bidirectional vertex-face incidence'],
    forbidden_transformations_applied=False,interface_tests_passed=True,optimizer_updates=0)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'README.md').write_text('Teacher CAD50 lossless float-mesh targets\n\n'
    'These files are exported from data/point50/training.pt in the supplied original results ZIP. '
    'The separately named teacher50_for_current_ae.zip was not supplied.\n'
    'vertices are the exact FP32 normalized coordinates in that training cache. faces retain original cache order/winding. '
    'No quantization, normalization, welding, simplification, or vertex reorder was applied.\n'
    'face_set and edge_index are derived undirected supervision; incidence_index has face nodes offset by the local vertex count. '
    'No octree or condition point cloud exists or is fabricated.\n'
    'The teacher text features, weights, and predictions are not model inputs or supervision.\n')
print(json.dumps(manifest['totals']),flush=True)
