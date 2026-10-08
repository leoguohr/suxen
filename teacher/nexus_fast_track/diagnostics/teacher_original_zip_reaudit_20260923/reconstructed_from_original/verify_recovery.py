"""Verify this recovery against original assets and yesterday's candidate.

All tensors stay on CPU. Actual AE replay plus bounded diffusion sampling.
Ground truth enters AE reconstruction/evaluation only. Point generation gets
text features and noise; topology generation gets vertices and noise.
No optimizer is constructed or updated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from recovered_networks import load_network, generate_points, generate_topology_latent
from recovered_objectives import grouped_bce, standard_normal_kl_elements
from topology_scores import decode_mesh, canonical_set, metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_error(a, b):
    return float((a - b).abs().max())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path)
    parser.add_argument('--sample-ids', default='2,0')
    args=parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'VERIFICATION_RESULT.json').exists():
        raise ValueError('Refusing to overwrite an existing verification result')
    here=Path(__file__).resolve().parent
    specification=json.loads((here/'verification_input.json').read_text())
    torch.set_num_threads(1)
    torch.manual_seed(20260923)
    # Prevent native fused-encoder selection from obscuring operator comparison.
    torch.backends.mha.set_fastpath_enabled(False)
    start=time.time()
    result={'author_model':'gpt-6-astra', 'reasoning_effort':'xhigh',
        'torch':torch.__version__, 'device':'cpu', 'threads':1, 'mha_fastpath':False,
        'optimizer_updates':0, 'original_zip_sha256':specification['archive_sha256'],
        'code_sha256':{p.name:digest(p) for p in sorted(here.glob('*.py'))}}
    verified={}
    for name, expected in specification['asset_sha256'].items():
        actual=digest(args.teacher_root/name)
        if actual!=expected: raise ValueError(f'Original asset hash mismatch: {name}')
        verified[name]=actual
    result['asset_sha256']=verified
    roots={'ae':'results/minkowski_target_099/vae.pt',
           'topology':'results/minkowski_target_099/best_flow.pt',
           'points':'results/point_diffusion/latest.pt'}
    models={k:load_network(args.teacher_root/v,k) for k,v in roots.items()}
    result['strict_loads']={}
    for name,model in models.items():
        expected=specification['state_shapes'][name]
        actual={k:list(v.shape) for k,v in model.state_dict().items()}
        assert expected==actual, (name,'keys/shapes differ')
        assert all(p.device.type=='cpu' for p in model.parameters())
        result['strict_loads'][name]={'passed':True, 'states':len(actual),
            'parameter_count':sum(p.numel() for p in model.parameters()),
            'missing_keys':[], 'unexpected_keys':[], 'key_mapping':'identity',
            'coverage_fraction':1.0}
    cache=torch.load(args.teacher_root/'data/point50/training.pt',map_location='cpu',weights_only=True)
    features=torch.load(args.teacher_root/'results/point_diffusion/text_conditions.pt',map_location='cpu',weights_only=True)['features']
    norm=torch.load(args.teacher_root/'results/minkowski_target_099/latent_normalization.pt',map_location='cpu',weights_only=True)
    ae,flow,point=models['ae'],models['topology'],models['points']
    baseline=None
    if args.baseline_root:
        for name, expected in specification['baseline_sha256'].items():
            assert digest(args.baseline_root/name)==expected, name
        sys.path.insert(0,str(args.baseline_root))
        import models as yesterday
        import objectives as yesterday_losses
        baseline={'ae':yesterday.load_ae(args.teacher_root/roots['ae'])[0],
                  'topology':yesterday.load_topology(args.teacher_root/roots['topology']),
                  'points':yesterday.load_points(args.teacher_root/roots['points'])}
        result['baseline_sha256']=specification['baseline_sha256']
    records=[]
    paired=[]
    with torch.inference_mode():
        for i,sample in enumerate(cache['samples']):
            order=sample['original_vertex_indices']
            vertices=sample['vertices'][order.argsort()]
            faces=order[sample['faces']]
            ze,zf=ae(vertices,faces)
            prediction=decode_mesh(ze,zf)
            row=metrics(prediction,faces)
            reference={tuple(f) for f in specification['saved_ae_faces'][str(i)]}
            row.update(sample_id=i,reference_face_symmetric_difference=len(canonical_set(prediction['faces'])^reference))
            if baseline:
                old_e,old_f=baseline['ae'](vertices,faces)
                old_prediction=decode_mesh(old_e,old_f)
                row['vs_yesterday']={'edge_embedding_max_abs':max_error(ze,old_e),
                    'face_embedding_max_abs':max_error(zf,old_f),
                    'face_symmetric_difference':len(canonical_set(prediction['faces'])^canonical_set(old_prediction['faces']))}
            records.append(row)
        expected_count=torch.tensor([len(s['vertices']) for s in cache['samples']])
        result['point_count_accuracy_50']=float((point.count_logits(features).argmax(-1)==expected_count).float().mean())
        for i in map(int,args.sample_ids.split(',')):
            sample=cache['samples'][i]
            order=sample['original_vertex_indices']
            vertices=sample['vertices'][order.argsort()]
            faces=order[sample['faces']]
            text=features[i:i+1]
            pts=generate_points(point,text,torch.Generator().manual_seed(34567))
            latent=generate_topology_latent(flow,vertices,norm['mean'],norm['std'],torch.Generator().manual_seed(12345))
            te,tf=ae.decode(latent)
            row={'sample_id':i,'point_seed':34567,'topology_seed':12345,
                'point_steps':100,'topology_steps':50,
                'point_rmse_ordered':float((pts.double()-sample['vertices'].double()).square().mean().sqrt()),
                'point_all_finite':bool(torch.isfinite(pts).all()),
                'latent_all_finite':bool(torch.isfinite(latent).all()),
                'topology_metrics':metrics(decode_mesh(te,tf),faces)}
            if baseline:
                shared=torch.Generator().manual_seed(987)
                x=torch.randn(1,len(vertices),3,generator=shared)
                z=torch.randn(1,len(vertices),64,generator=shared)
                time_query=torch.tensor([.37])
                old_pts=yesterday.sample_points(baseline['points'],text,torch.Generator().manual_seed(34567))
                old_latent=yesterday.sample_topology(baseline['topology'],vertices,norm['mean'],norm['std'],torch.Generator().manual_seed(12345))
                old_te,old_tf=baseline['ae'].decode(old_latent)
                row['vs_yesterday_shared_inputs']={
                    'point_denoiser_max_abs':max_error(point.denoise(x,time_query,text),baseline['points'].denoise(x,time_query,text)),
                    'point_final_forward_max_abs':max_error(point(x,time_query,text),baseline['points'](x,time_query,text)),
                    'topology_velocity_max_abs':max_error(flow(z,time_query,vertices[None]),baseline['topology'](z,time_query,vertices[None])),
                    'sampled_points_max_abs':max_error(pts,old_pts),
                    'sampled_topology_latent_max_abs':max_error(latent,old_latent),
                    'sampled_topology_faces_symmetric_difference':len(canonical_set(decode_mesh(te,tf)['faces'])^canonical_set(decode_mesh(old_te,old_tf)['faces']))}
            paired.append(row)
    total={}
    for kind in ['edge','face']:
        counts={k:sum(r[kind][k] for r in records) for k in ['tp','fp','fn']}
        counts['f1']=2*counts['tp']/max(2*counts['tp']+counts['fp']+counts['fn'],1)
        total[kind]=counts
    total['joint_strict']=sum(all(r[k]['fp']==r[k]['fn']==0 for k in ['edge','face']) for r in records)
    total['exact_face_sets_vs_original']=sum(r['reference_face_symmetric_difference']==0 for r in records)
    if baseline:
        total['exact_face_sets_vs_yesterday']=sum(r['vs_yesterday']['face_symmetric_difference']==0 for r in records)
        total['max_edge_embedding_difference_vs_yesterday']=max(r['vs_yesterday']['edge_embedding_max_abs'] for r in records)
        total['max_face_embedding_difference_vs_yesterday']=max(r['vs_yesterday']['face_embedding_max_abs'] for r in records)
    scores=torch.tensor([2.,-3.,-1.,4.],requires_grad=True)
    labels=torch.tensor([1,0,1,0])
    loss=grouped_bce(scores,labels)
    loss.backward()
    result['loss_component_probe']={'grouped_bce':float(loss.detach()),
        'gradient_finite':bool(torch.isfinite(scores.grad).all()),
        'zero_standard_normal_kl':float(standard_normal_kl_elements(torch.zeros(2,3),torch.zeros(2,3)).sum())}
    if baseline:
        result['loss_component_probe']['vs_yesterday_bce_abs']=float((loss.detach()-yesterday_losses.hard4_bce(scores.detach(),labels)).abs())
    result.update(ae_50_summary=total,ae_50_records=records,bounded_diffusion_samples=paired,
                  elapsed_seconds=time.time()-start)
    (args.out/'VERIFICATION_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    compact={k:v for k,v in result.items() if k!='ae_50_records'}
    print(json.dumps(compact,indent=2))


if __name__=='__main__':main()
