"""Bounded read-only CPU probe of reviewed reconstructed candidates.

No training, gradient, optimizer step, CUDA calls or full-dataset sampling.
Author: gpt-6-astra/xhigh, 2026-09-23.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'candidate_code'))
import torch
from models import load_ae, load_topology, load_points, sample_points, sample_topology
from indicators import predict_mesh, mesh_metrics

ROOT = Path('/guohaoran/nexus_fast_track/diagnostics/teacher_reverse_audit_20260922/teacher_assets')

def main():
    torch.set_num_threads(1)
    spec = json.loads((HERE/'cpu_probe_input.json').read_text())
    verified = {}
    for name, expected in spec['asset_sha256'].items():
        path = ROOT/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f'Original ZIP asset mismatch: {name}')
        verified[name] = actual
    for name, expected in spec['candidate_sha256'].items():
        actual = hashlib.sha256((HERE/'candidate_code'/name).read_bytes()).hexdigest()
        if actual != expected: raise ValueError(f'Candidate code mismatch: {name}')
    start = time.time()
    cp=ROOT/'results/minkowski_target_099'
    ae,_=load_ae(cp/'vae.pt')
    flow=load_topology(cp/'best_flow.pt')
    point=load_points(ROOT/'results/point_diffusion/latest.pt')
    cache=torch.load(ROOT/'data/point50/training.pt',map_location='cpu',weights_only=True)
    text=torch.load(ROOT/'results/point_diffusion/text_conditions.pt',map_location='cpu',weights_only=True)['features']
    norm=torch.load(cp/'latent_normalization.pt',map_location='cpu',weights_only=True)
    sample_id=2
    sample=cache['samples'][sample_id]
    old=sample['original_vertex_indices']
    vertices=sample['vertices'][old.argsort()]
    faces=old[sample['faces']]
    result={'author_model':'gpt-6-astra','reasoning_effort':'xhigh',
        'torch':torch.__version__, 'device':'cpu', 'threads':1,
        'original_zip_sha256':spec['archive_sha256'], 'verified_asset_sha256':verified,
        'candidate_sha256':spec['candidate_sha256'],
        'sample_id':sample_id, 'vertices':len(vertices), 'faces':len(faces),
        'strict_loads':{}, 'qualification':'One-sample CPU replay; no new full-50 evaluation or training.'}
    for name, model in [('ae',ae),('topology_flow',flow),('point_flow',point)]:
        result['strict_loads'][name]={'passed':True,'state_dict_entries':len(model.state_dict()),
            'parameters':sum(p.numel() for p in model.parameters()),
            'all_parameters_cpu':all(p.device.type=='cpu' for p in model.parameters())}
    with torch.inference_mode():
        ze,zf=ae(vertices,faces)
        ae_pred=predict_mesh(ze,zf)
        result['ae']=mesh_metrics(ae_pred,faces)
        ref=set(tuple(sorted(f)) for f in spec['original_ae_sample_02_faces'])
        result['ae']['face_symmetric_difference_vs_original_saved']=len(ref ^ set(map(tuple,ae_pred['pred_faces'].tolist())))
        counts=point.count_logits(text).argmax(-1)
        result['point_count_accuracy_50']=float((counts==torch.tensor([len(s['vertices']) for s in cache['samples']])).float().mean())
        p=sample_points(point,text[sample_id:sample_id+1],torch.Generator(device='cpu').manual_seed(34567),steps=100)
        result['point']={'sample_steps':100,'seed':34567,'output_shape':list(p.shape),
            'all_finite':bool(torch.isfinite(p).all()),
            'ordered_coordinate_rmse':float((p.double()-sample['vertices'].double()).square().mean().sqrt()),
            'residual_scale':float(point.residual_scale)}
        latent=sample_topology(flow,vertices,norm['mean'],norm['std'],torch.Generator(device='cpu').manual_seed(12345),steps=50)
        te,tf=ae.decode(latent)
        result['topology']=mesh_metrics(predict_mesh(te,tf),faces)
        result['topology'].update(sample_steps=50,seed=12345,latent_all_finite=bool(torch.isfinite(latent).all()))
    result['elapsed_seconds']=time.time()-start
    (HERE/'CPU_PROBE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
