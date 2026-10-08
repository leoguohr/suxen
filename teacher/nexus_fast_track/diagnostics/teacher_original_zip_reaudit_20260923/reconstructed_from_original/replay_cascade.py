"""Full 50 CPU noise-to-mesh replay of this recovered implementation.

Seeds and sample order match yesterday's first CPU cascade. Uses no optimizer.
References/ground truth are read only after inference for evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from recovered_networks import load_network, generate_points, generate_topology_latent
from topology_scores import decode_mesh, canonical_set, metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output must be absent or empty')
    args.out.mkdir(parents=True,exist_ok=True)
    here=Path(__file__).resolve().parent
    spec=json.loads((here/'verification_input.json').read_text())
    for name,expected in spec['asset_sha256'].items():
        assert hashlib.sha256((args.teacher_root/name).read_bytes()).hexdigest()==expected,name
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    cp=args.teacher_root/'results/minkowski_target_099'
    ae=load_network(cp/'vae.pt','ae')
    flow=load_network(cp/'best_flow.pt','topology')
    points=load_network(args.teacher_root/'results/point_diffusion/latest.pt','points')
    text=torch.load(args.teacher_root/'results/point_diffusion/text_conditions.pt',map_location='cpu',weights_only=True)['features']
    normalization=torch.load(cp/'latent_normalization.pt',map_location='cpu',weights_only=True)
    point_rng=torch.Generator().manual_seed(34567)
    topology_rng=torch.Generator().manual_seed(12345)
    records=[]
    start=time.time()
    with torch.inference_mode():
        # Only cached text features plus Gaussian noise enter point generation.
        # Only those generated points plus independent noise enter topology.
        for i in range(len(text)):
            vertex=generate_points(points,text[i:i+1],point_rng,100)
            latent=generate_topology_latent(flow,vertex,normalization['mean'],normalization['std'],topology_rng,50)
            ze,zf=ae.decode(latent)
            pred=decode_mesh(ze,zf)
            np.savez_compressed(args.out/f'{i:02d}_mesh.npz',vertices=vertex.numpy(),
                                pred_edges=pred['edges'].numpy(),pred_faces=pred['faces'].numpy())
            print(json.dumps({'generated':i,'vertices':len(vertex),'faces':len(pred['faces']),
                              'elapsed_seconds':time.time()-start}),flush=True)
    generation_seconds=time.time()-start
    # Evaluation begins only after all predictions have been saved.
    cache=torch.load(args.teacher_root/'data/point50/training.pt',map_location='cpu',weights_only=True)
    reference=json.loads((here/'yesterday_cascade_reference.json').read_text())
    for i,sample in enumerate(cache['samples']):
        saved=np.load(args.out/f'{i:02d}_mesh.npz')
        vertex=saved['vertices'].astype('float64')
        target=sample['vertices'].double().numpy()
        assert len(vertex)==len(target),(i,'Count mismatch')
        distance=((vertex[:,None]-target[None])**2).sum(-1)
        row,col=linear_sum_assignment(distance)
        predicted_to_true=np.empty(len(vertex),dtype=np.int64)
        predicted_to_true[row]=col
        true_to_predicted=np.argsort(predicted_to_true)
        truth_faces=torch.from_numpy(true_to_predicted[sample['faces'].numpy()])
        edges=torch.from_numpy(saved['pred_edges'])
        faces=torch.from_numpy(saved['pred_faces'])
        # Reconstruct candidates from the saved predicted edge graph for metrics.
        neighbors=[set() for _ in range(len(vertex))]
        for a,b in edges.tolist():neighbors[a].add(b);neighbors[b].add(a)
        triples=[(a,b,c) for a in range(len(vertex)) for b in sorted(neighbors[a]) if a<b
                 for c in sorted(neighbors[a]&neighbors[b]) if b<c]
        prediction={'edges':edges,'faces':faces,'candidates':torch.tensor(triples,dtype=torch.long).reshape(-1,3)}
        error=vertex-target[predicted_to_true]
        spacing=((target[:,None]-target[None])**2).sum(-1)
        np.fill_diagonal(spacing,np.inf)
        record=metrics(prediction,truth_faces)
        old=reference['samples'][str(i)]
        new_faces=canonical_set(faces)
        old_faces={tuple(sorted(f)) for f in old['pred_faces']}
        record.update(sample_id=i,point_rmse=float(np.sqrt((error**2).mean())),
            point_sum_squared_error=float((error**2).sum()),point_coordinate_count=error.size,
            max_vertex_error=float(np.sqrt((error**2).sum(-1)).max()),
            max_spacing_ratio=float(np.sqrt((error**2).sum(-1)).max()/np.sqrt(spacing.min())),
            symmetric_squared_chamfer=float(distance.min(0).mean()+distance.min(1).mean()),
            yesterday_record=old['metrics'],
            face_symmetric_difference_vs_yesterday=len(new_faces^old_faces),
            point_max_abs_vs_yesterday=float(np.max(np.abs(vertex-np.asarray(old['vertices'])))))
        records.append(record)
    total={}
    for kind in ['edge','face']:
        total[kind]={key:sum(r[kind][key] for r in records) for key in ['tp','fp','fn']}
        d=total[kind]
        d['f1']=2*d['tp']/max(2*d['tp']+d['fp']+d['fn'],1)
    total['joint_strict']=sum(all(r[k]['fp']==r[k]['fn']==0 for k in ['edge','face']) for r in records)
    total['exact_face_sets_vs_yesterday']=sum(r['face_symmetric_difference_vs_yesterday']==0 for r in records)
    result={'author_model':'gpt-6-astra','reasoning_effort':'xhigh','device':'cpu',
        'torch':torch.__version__,'threads':1,'mha_fastpath':False,'optimizer_updates':0,
        'point_seed':34567,'topology_seed':12345,'point_steps':100,'topology_steps':50,
        'rng_context':'Two CPU generators initialized once before the 00..49 loop; not reset per sample.',
        'ground_truth_access':'Only after all 50 new predictions are saved; vertex matching used for evaluation only.',
        'original_zip_sha256':spec['archive_sha256'],'asset_sha256':spec['asset_sha256'],
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob('*.py')},
        'generation_seconds':generation_seconds,'total_seconds':time.time()-start,
        'total':total,'point_total':{'count_accuracy':1.,
            'coordinate_rmse':float(np.sqrt(sum(r['point_sum_squared_error'] for r in records)/sum(r['point_coordinate_count'] for r in records))),
            'max_spacing_ratio':max(r['max_spacing_ratio'] for r in records)},
        'yesterday_environment':reference['environment'],'yesterday_total':reference['total'],
        'reference_sha256':hashlib.sha256((here/'yesterday_cascade_reference.json').read_bytes()).hexdigest(),
        'comparison_limit':'Historical baseline used torch2.10.0+cpu; current runtime differs. Small same-runtime paired tests are reported separately.',
        'per_mesh':records}
    (args.out/'CASCADE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_mesh'},indent=2),flush=True)


if __name__=='__main__':main()
