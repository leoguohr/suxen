"""Actual teacher-weight AE forward under all three recovered indicators."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from models import load_ae
from indicators import predict_mesh,mesh_metrics,aggregate
from replay import sha256

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teacher-root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['early','final'],default='early');args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('Output must be new/empty')
    args.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    samples=torch.load(args.teacher_root/'data/point50/training.pt',weights_only=True,map_location='cpu')['samples']
    result={'scope':'Actual CPU AE forward; no training. Metrics computed from predicted edge triangles.', 'methods':{}}
    with torch.inference_mode():
        for method in (['cosine','euclidean','spacetime'] if args.phase=='early' else ['spacetime']):
            start=time.time();cp=args.teacher_root/f'results/overfit/{method}/vae.pt' if args.phase=='early' else args.teacher_root/'results/minkowski_target_099/vae.pt';ae,_=load_ae(cp);records=[]
            dest=args.out/method;dest.mkdir()
            for j,s in enumerate(samples):
                old=s['original_vertex_indices'];vertices=s['vertices'][old.argsort()];faces=old[s['faces']]
                ze,zf=ae(vertices,faces)
                pred=predict_mesh(ze,zf,method,float(ae.indicator.edge_threshold),float(ae.indicator.face_threshold))
                rec=mesh_metrics(pred,faces);rec['id']=j
                refp=args.teacher_root/f'results/overfit/{method}/reconstructions/ae_{j:02d}_faces.json' if args.phase=='early' else args.teacher_root/f'results/minkowski_target_099/latest_predictions/ae/{j:02d}_faces.json'
                ref=set(tuple(sorted(f)) for f in json.load(open(refp)))
                rec['reference_face_symmetric_diff']=len(set(map(tuple,pred['pred_faces'].tolist()))^ref);records.append(rec)
                np.savez_compressed(dest/f'{j:02d}.npz',vertices=vertices.numpy(),**{k:v.numpy() for k,v in pred.items()})
            result['methods'][method]=dict(total=aggregate(records),per_mesh=records,checkpoint_sha256=sha256(cp),seconds=time.time()-start)
            print(method,result['methods'][method]['total'],flush=True)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
