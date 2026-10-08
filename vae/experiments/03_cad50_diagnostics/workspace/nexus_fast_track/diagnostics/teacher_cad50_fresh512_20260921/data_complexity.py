"""Read-only dataset scale and exact unlabelled simplicial-topology comparisons."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import collections,csv,json,time
from pathlib import Path
import networkx as nx
import numpy as np
from loader import load_all
R=Path(__file__).resolve().parent
manifest,samples=load_all(R/'data');records=manifest['meshes'];classes=[]
for uid,s in samples.items():
    n=len(s['vertices']);f=s['faces'];edges=s['edge_index'].T
    # Colored incidence graphs encode every face, not just edge-graph isomorphism.
    graph=nx.Graph();graph.add_nodes_from((i,{'kind':0}) for i in range(n))
    graph.add_nodes_from((n+i,{'kind':1}) for i in range(len(f)))
    graph.add_edges_from((int(v),n+i) for i,face in enumerate(f) for v in face)
    key=(n,len(edges),len(f),tuple(sorted(d for v,d in graph.degree() if v<n)))
    matched=None
    for k,group in enumerate(classes):
        if group['key']==key and nx.is_isomorphic(graph,group['graph'],node_match=nx.algorithms.isomorphism.categorical_node_match('kind',-1)):
            matched=k;break
    if matched is None:classes.append(dict(key=key,graph=graph,uids=[uid]))
    else:classes[matched]['uids'].append(uid)
    print('TOPOLOGY',uid,len(classes),flush=True)
old=Path('/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914/overfit100_manifest.csv')
previous=list(csv.DictReader(old.open()))
def stats(rows):
    n=np.array([int(r['vertices']) for r in rows]);e=sum(int(r['gt_edges']) for r in rows);f=sum(int(r['gt_faces']) for r in rows)
    return dict(meshes=len(rows),vertices_total=int(n.sum()),vertices_min=int(n.min()),vertices_median=float(np.median(n)),vertices_max=int(n.max()),
        gt_edges=e,gt_faces=f,all_edge_pairs=int(sum(x*(x-1)//2 for x in n)),
        vertex_count_histogram=dict(sorted(collections.Counter(map(int,n)).items())),
        edge_positive_fraction=e/sum(x*(x-1)//2 for x in n))
teacher=stats(records);original=stats(previous)
result=dict(teacher50=teacher,original100=original,
    original100_over_teacher50_pair_ratio=original['all_edge_pairs']/teacher['all_edge_pairs'],
    teacher8vertex12face_uids=[r['uid'] for r in records if r['vertices']==8 and r['gt_faces']==12],
    teacher_exact_incidence_isomorphism_classes=[dict(uids=g['uids'],vertices=g['key'][0],edges=g['key'][1],faces=g['key'][2]) for g in classes],
    topology_definition='exact bipartite vertex-face incidence graph isomorphism, preserving node kinds; ignores geometry and face orientation only for this read-only analysis',
    limitations='Scale/topology repetition describes dataset demands, not a proof of trainability, parameter capacity or teacher runtime causation. Training inputs unchanged.')
(R/'data_complexity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='teacher_exact_incidence_isomorphism_classes'}),flush=True)
