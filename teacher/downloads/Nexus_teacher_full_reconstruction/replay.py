"""Read-only reconstruction CLI. Does not train or modify any teacher checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from models import load_ae, load_topology, load_points, sample_points, sample_topology, FORWARD_CONTRACT
from indicators import predict_mesh, mesh_metrics, aggregate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open('w') as f:
        for p in vertices: f.write('v '+' '.join(f'{float(x):.9g}' for x in p)+'\n')
        for p in faces: f.write('f '+' '.join(str(int(x)+1) for x in p)+'\n')


def point_metrics(pred: torch.Tensor, gt: torch.Tensor) -> tuple[dict, np.ndarray | None]:
    p, g = pred.detach().cpu().double().numpy(), gt.detach().cpu().double().numpy()
    if len(p) != len(g):
        return dict(count_correct=False,predicted_count=len(p),target_count=len(g)), None
    dist = ((p[:,None]-g[None,:])**2).sum(-1)
    row, col = linear_sum_assignment(dist)
    pred_to_gt = np.empty(len(p),np.int64); pred_to_gt[row] = col
    delta = p-g[pred_to_gt]
    gd = ((g[:,None]-g[None,:])**2).sum(-1); np.fill_diagonal(gd,np.inf)
    spacing = float(np.sqrt(gd.min()))
    max_error = float(np.sqrt((delta**2).sum(-1)).max())
    return dict(count_correct=True,predicted_count=len(p),target_count=len(g),
        coordinate_rmse=float(np.sqrt(np.mean(delta**2))),sum_squared_error=float((delta**2).sum()),
        coordinates=delta.size,max_error=max_error,minimum_spacing=spacing,
        spacing_ratio=max_error/spacing if spacing>0 else None,
        symmetric_squared_chamfer=float(dist.min(0).mean()+dist.min(1).mean())), pred_to_gt


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--teacher-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--stage',choices=['points','topology','cascade'],required=True)
    ap.add_argument('--phase',choices=['early','final'],default='final')
    ap.add_argument('--method',choices=['cosine','euclidean','spacetime'],default='spacetime')
    ap.add_argument('--point-seed',type=int,default=34567)
    ap.add_argument('--topology-seed',type=int,default=12345)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--limit',type=int,default=50)
    ap.add_argument('--point-dir',type=Path,help='Optional previously generated point outputs; provenance is recorded. Cascade only.')
    args=ap.parse_args(); root=args.teacher_root
    if not 1<=args.limit<=50: raise ValueError('limit must be 1..50')
    if args.out.exists() and any(args.out.iterdir()): raise ValueError('Output must be absent or empty')
    if args.stage!='cascade' and args.point_dir is not None: raise ValueError('--point-dir requires cascade')
    if args.phase=='final' and args.method!='spacetime': raise ValueError('Final high-F1 weights exist only for spacetime')
    args.out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(1)
    dev=torch.device(args.device)
    if dev.type=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data_path=root/'data/point50/training.pt'
    cache=torch.load(data_path,map_location='cpu',weights_only=True)
    samples=cache['samples']
    textpath=root/'results/point_diffusion/text_conditions.pt'
    text=torch.load(textpath,map_location='cpu',weights_only=True)['features'].to(dev)
    paths={'data':data_path,'text_conditions':textpath}
    pm=None;ae=None;flow=None
    if args.stage=='points' or (args.stage=='cascade' and args.point_dir is None):
        paths['point_checkpoint']=root/'results/point_diffusion/latest.pt'
        pm=load_points(paths['point_checkpoint']).to(dev)
    if args.stage in ('topology','cascade'):
        cp=root/'results/minkowski_target_099' if args.phase=='final' else root/f'results/overfit/{args.method}'
        paths.update(ae_checkpoint=cp/'vae.pt',topology_checkpoint=cp/('best_flow.pt' if args.phase=='final' else 'flow.pt'),normalization=cp/'latent_normalization.pt')
        ae,_=load_ae(paths['ae_checkpoint']);ae=ae.to(dev)
        flow=load_topology(paths['topology_checkpoint']).to(dev)
        norm={k:v.to(dev) for k,v in torch.load(paths['normalization'],map_location='cpu',weights_only=True).items()}
    gp=torch.Generator(device=dev).manual_seed(args.point_seed)
    gz=torch.Generator(device=dev).manual_seed(args.topology_seed)
    recs=[];point_recs=[];start=time.time()
    with torch.inference_mode():
        for j,s in enumerate(samples[:args.limit]):
            if args.stage in ('points','cascade'):
                # The inference function receives only text_feature and random generator.
                if args.point_dir is None:
                    vertices=sample_points(pm,text[j:j+1],gp)
                else:
                    vertices=torch.from_numpy(np.load(args.point_dir/f'{j:02d}.npy')).to(dev)
                np.save(args.out/f'{j:02d}.npy',vertices.cpu().numpy())
                # Matching is evaluation-only; predicted vertices are not reordered.
                pr,mapping=point_metrics(vertices,s['vertices']);pr['id']=j;point_recs.append(pr)
                if mapping is None:
                    if args.stage=='cascade': raise RuntimeError('Count mismatch: refusing fabricated GT-face correspondence')
                else:
                    gt_to_pred=np.argsort(mapping)
                    gf=torch.from_numpy(gt_to_pred[s['faces'].numpy()])
                old=s['original_vertex_indices']
            else:
                old=s['original_vertex_indices']
                vertices=s['vertices'][old.argsort()].to(dev)
                gf=old[s['faces']]
            if args.stage=='points':
                if (root/f'results/point_prior_candidate/verified_points/{args.point_seed}/{j:02d}.npy').exists():
                    ref=np.load(root/f'results/point_prior_candidate/verified_points/{args.point_seed}/{j:02d}.npy')
                    if ref.shape==tuple(vertices.shape):pr['reference_max_abs']=float(np.max(np.abs(vertices.cpu().numpy()-ref)))
            else:
                latent=sample_topology(flow,vertices,norm['mean'],norm['std'],gz)
                ze,zf=ae.decode(latent)
                pred=predict_mesh(ze,zf,args.method,float(ae.indicator.edge_threshold),float(ae.indicator.face_threshold))
                rec=mesh_metrics(pred,gf);rec['id']=j
                if args.stage=='topology':
                    refp=(cp/f'confirmation_predictions/{j:02d}_faces.json' if args.topology_seed==23456 else cp/f'latest_predictions/flow/{j:02d}_faces.json') if args.phase=='final' else cp/f'reconstructions/flow_{j:02d}_faces.json'
                else:refp=root/f'results/cascade_overfit/{j:02d}/faces.json'
                ref=set(tuple(sorted(f)) for f in json.load(open(refp)))
                predset=set(map(tuple,pred['pred_faces'].tolist()))
                rec['reference_face_symmetric_diff']=len(predset^ref)
                recs.append(rec)
                np.savez_compressed(args.out/f'{j:02d}_mesh.npz',vertices=vertices.cpu().numpy(),
                    **{k:v.cpu().numpy() for k,v in pred.items()})
                write_obj(args.out/f'{j:02d}.obj',vertices.cpu().numpy(),pred['pred_faces'].cpu().numpy())
            if j%10==0:print(j,'elapsed',round(time.time()-start,2),flush=True)
    result=dict(scope='Actual read-only candidate network inference; not exact original source recovery or a retraining result.',
        stage=args.stage,phase=args.phase,method=args.method,device=str(dev),torch=torch.__version__,
        forward_contract=FORWARD_CONTRACT,point_seed=args.point_seed,topology_seed=args.topology_seed,
        point_steps=100,topology_steps=50,inputs={k:{'path':str(v),'sha256':sha256(v)} for k,v in paths.items()},
        point_dir=str(args.point_dir) if args.point_dir else None,seconds=time.time()-start)
    if recs:result.update(total=aggregate(recs),per_mesh=recs)
    if point_recs:
        ok=[r for r in point_recs if r['count_correct']]
        result['point_total']=dict(samples=len(point_recs),count_correct=len(ok),
            coordinate_rmse=float(np.sqrt(sum(r['sum_squared_error'] for r in ok)/sum(r['coordinates'] for r in ok))) if ok else None,
            max_spacing_ratio=max((r['spacing_ratio'] for r in ok if r['spacing_ratio'] is not None),default=None))
        result['point_samples']=point_recs
        if args.point_dir:
            result['point_source_files']={f'{j:02d}.npy':sha256(args.point_dir/f'{j:02d}.npy') for j in range(args.limit)}
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({k:result[k] for k in ('total','point_total') if k in result},indent=2))

if __name__=='__main__':main()
