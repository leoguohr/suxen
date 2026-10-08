"""Frozen checkpoint -> Gaussian generation -> Decoder -> complete actual triangles."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from _faces_reference import atomic_json, atomic_npz, file_sha, stream_faces, aggregate
from _fixed100_reference import edge_logits, face_logits
from topology_flow import PointCloudTopologyFlow, TopologyFlowConfig, math_context, configure_math_backend
from latent_data import LatentCache
from vae_codec import load_frozen_vae, decode_latents
from sampling import generate_latents


def seed_for_mesh(seed, uid):
    return int.from_bytes(hashlib.sha256(f'topology-eval/{seed}/{uid}'.encode()).digest()[:8],'little')


def with_metrics(result):
    for task in ('edge','face'):
        x=result[task]
        x['micro_f1']=2*x['tp']/max(2*x['tp']+x['fp']+x['fn'],1)
    result['edge_strict']=result['edge']['fp']==result['edge']['fn']==0
    result['joint_strict']=result['edge_strict'] and result['face']['fp']==result['face']['fn']==0
    return result


def complete_aggregate(results, uids):
    if [r['uid'] for r in results]!=list(uids) or not all(r['complete'] for r in results):
        raise ValueError('Only the complete, ordered selection can be aggregated')
    metrics=aggregate(results)
    metrics['edge_strict_count']=len(metrics['edge_perfect_uids'])
    metrics['joint_strict_count']=metrics['joint_perfect']
    metrics['strict50_under_this_noise']=len(uids)==50 and metrics['joint_perfect']==50
    return metrics


def evaluator_sources():
    names=('evaluate_flow.py','evaluate_vae.py','sampling.py','vae_codec.py','latent_data.py',
           '_faces_reference.py','_fixed100_reference.py','_vae_reference.py',
           '_vertex_reference.py','topology_flow.py')
    return {name:file_sha(Path(__file__).parent/name) for name in names}


@torch.no_grad()
def evaluate_embeddings(edge,face,gt_edges,gt_faces,directory,identity,stop=lambda:False,chunk_size=32768):
    """Threshold zero; all Edge pairs, then only triangles in the predicted graph."""
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    n,device = len(edge),edge.device
    base = directory/'edge-and-face-embedding.npz'
    binding = directory/'binding.json'
    if binding.exists():
        old = json.loads(binding.read_text())
        if old['identity']!=identity: raise ValueError('Evaluation checkpoint/seed changed')
        if file_sha(base)!=old['sha256']: raise ValueError('Evaluation base corrupted')
        with np.load(base,allow_pickle=False) as b:
            pairs,logits,labels,face_np = b['pairs'],b['logits'],b['labels'],b['face_embedding']
        face = torch.from_numpy(face_np).to(device)
    else:
        pairs = np.stack(np.triu_indices(n,k=1),axis=1).astype(np.int64)
        truth = gt_edges.cpu().numpy()
        labels = np.isin(pairs@np.array([n,1]),truth@np.array([n,1]))
        logits = np.empty(len(pairs),dtype=np.float32)
        for start in range(0,len(pairs),chunk_size):
            if stop(): return dict(complete=False)
            logits[start:start+chunk_size] = edge_logits(edge,torch.from_numpy(pairs[start:start+chunk_size]).to(device)).cpu().numpy()
        if not np.isfinite(logits).all(): raise FloatingPointError('Nonfinite Edge scores')
        atomic_npz(base,pairs=pairs,logits=logits,labels=labels,edge_embedding=edge.cpu().numpy(),face_embedding=face.cpu().numpy())
        atomic_json(binding,dict(identity=identity,sha256=file_sha(base)))
    prediction = logits>0
    es = dict(tp=int((prediction&labels).sum()),fp=int((prediction&~labels).sum()),
              fn=int((~prediction&labels).sum()),tn=int((~prediction&~labels).sum()))
    adjacency = np.zeros((n,n),dtype=np.bool_)
    adjacency[pairs[prediction,0],pairs[prediction,1]]=True
    gt = np.unique(np.sort(gt_faces.cpu().numpy(),axis=1),axis=0)
    def score(ids): return face_logits(face,torch.from_numpy(ids.astype(np.int64)).to(device)).cpu().numpy()
    fs = stream_faces(adjacency,gt,score,directory/'face-shards',identity,stop,chunk_size)
    if not fs['complete']: return dict(complete=False,edge=es,face_partial=fs)
    return with_metrics(dict(complete=True,uid=identity['uid'],edge=es,
        face={k:fs[k] for k in ('tp','fp','fn','tn')},actual_face_candidates=fs['candidates'],
        face_fn_missing=fs['fn_missing'],face_fn_present=fs['fn_present']))


def export_visualizations(item, directory, identity, result, stop=lambda:False):
    """Full OBJ geometry, with original vertices and no output repair or truncation."""
    directory=Path(directory); manifest=directory/'visualizations.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        if saved['identity']!=identity or any(file_sha(directory/r['file'])!=r['sha256'] for r in saved['files']):
            raise ValueError('Visualization binding mismatch')
        return True
    if stop(): return False
    with np.load(directory/'edge-and-face-embedding.npz',allow_pickle=False) as a:
        predicted_edges=a['pairs'][a['logits']>0]
    progress=json.loads((directory/'face-shards/progress.json').read_text())
    if not progress['complete']: raise ValueError('Cannot export incomplete predicted faces')
    vertices=item['vertices'].cpu().numpy()
    for name in ('gt','predicted'):
        path=directory/(name+'.obj'); tmp=path.with_suffix('.obj.tmp')
        try:
            with tmp.open('w') as f:
                f.write('# Original GT vertex coordinates and local numbering. No repair.\n')
                f.write('# Predicted faces are unordered topology triples; winding is not inferred.\n')
                for xyz in vertices: f.write('v '+' '.join(format(float(x),'.9g') for x in xyz)+'\n')
                edges=item['edges'].cpu().numpy() if name=='gt' else predicted_edges
                for offset in range(0,len(edges),32768):
                    if stop(): raise InterruptedError()
                    for ids in edges[offset:offset+32768]: f.write('l '+' '.join(str(int(x)+1) for x in ids)+'\n')
                if name=='gt':
                    for ids in item['faces'].cpu().numpy(): f.write('f '+' '.join(str(int(x)+1) for x in ids)+'\n')
                else:
                    for index in range(progress['shards']):
                        if stop(): raise InterruptedError()
                        with np.load(directory/f'face-shards/part-{index:08d}.npz',allow_pickle=False) as shard:
                            for ids in shard['ids'][shard['logits']>0]:
                                f.write('f '+' '.join(str(int(x)+1) for x in ids)+'\n')
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        except InterruptedError:
            tmp.unlink(missing_ok=True)
            return False
    atomic_json(manifest,dict(identity=identity,complete=True,
        format='OBJ; identical original coordinates; l=edges; f=faces; predicted winding unspecified',
        predicted_edges=result['edge']['tp']+result['edge']['fp'],
        predicted_faces=result['face']['tp']+result['face']['fp'],
        files=[dict(file=name+'.obj',sha256=file_sha(directory/(name+'.obj')),
                    bytes=(directory/(name+'.obj')).stat().st_size) for name in ('gt','predicted')]))
    return True


@torch.no_grad()
def evaluate_item(vae, item, directory, identity, make_latent, device, stop):
    """Resume a bound latent/embedding/triangle evaluation without resampling noise."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    metrics=directory/'metrics.json'
    if metrics.exists():
        result=json.loads(metrics.read_text())
        if result['identity']!=identity or not result['complete']: raise ValueError('Invalid committed metrics')
        return result
    if stop(): return None
    with math_context(device.type):
        base=directory/'edge-and-face-embedding.npz'
        if (directory/'binding.json').exists():
            binding=json.loads((directory/'binding.json').read_text())
            if binding['identity']!=identity or binding['sha256']!=file_sha(base):
                raise ValueError('Resume evaluation base mismatch')
            with np.load(base,allow_pickle=False) as a:
                decoded={k:torch.from_numpy(a[k+'_embedding'].copy()).to(device) for k in ('edge','face')}
        else:
            latent_path=directory/'latent.npz'; latent_binding=directory/'latent.json'
            if latent_binding.exists():
                binding=json.loads(latent_binding.read_text())
                if binding['identity']!=identity or binding['sha256']!=file_sha(latent_path):
                    raise ValueError('Resume latent binding mismatch')
                with np.load(latent_path,allow_pickle=False) as a: latent=torch.from_numpy(a['latent'].copy()).to(device)
            else:
                latent=make_latent()
                if not torch.isfinite(latent).all(): raise FloatingPointError('Nonfinite evaluation latent')
                atomic_npz(latent_path,latent=latent.cpu().numpy())
                atomic_json(latent_binding,dict(identity=identity,sha256=file_sha(latent_path)))
            if stop(): return None
            decoded=decode_latents(vae,latent)
        result=evaluate_embeddings(decoded['edge'],decoded['face'],item['edges'],item['faces'],
            directory,identity,stop)
    if not result['complete']: return None
    result['identity']=identity
    atomic_json(metrics,result)
    return result


def load_vae_baseline(directory, cache):
    directory=Path(directory); summary=json.loads((directory/'summary.json').read_text())
    identity=summary['identity']
    if not summary['complete'] or identity['cache_sha256']!=cache.sha256 or identity['vae_sha256']!=cache.manifest['source_checkpoint_sha256']:
        raise ValueError('A complete VAE baseline on the identical cache is required')
    if identity['uids']!=cache.uids or 'mu' not in summary['variants']:
        raise ValueError('VAE baseline selection mismatch')
    variants={}
    for variant,detail in summary['variants'].items():
        rows=[]
        for uid in cache.uids:
            path=directory/variant/uid/'metrics.json'
            if file_sha(path)!=detail['metrics_sha256'][uid]: raise ValueError('VAE baseline metric hash mismatch')
            row=json.loads(path.read_text())
            expected=dict(identity,variant=variant,uid=uid)
            if not row['complete'] or row['identity']!=expected: raise ValueError('VAE baseline row identity mismatch')
            rows.append(row)
        actual=complete_aggregate(rows,cache.uids)
        if actual!=detail['metrics']: raise ValueError('VAE baseline aggregate mismatch')
        variants[variant]=dict(rows={r['uid']:r for r in rows},metrics=actual)
    return variants, file_sha(directory/'summary.json')


def metric_difference(current, baseline):
    difference={task:{key:current[task][key]-baseline[task][key] for key in ('tp','fp','fn','micro_f1')}
                for task in ('edge','face')}
    for key in ('edge_strict','joint_strict','edge_strict_count','joint_strict_count'):
        if key in current and key in baseline: difference[key]=int(current[key])-int(baseline[key])
    return difference


def run(args, budget):
    start=time.monotonic()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def stop():
        return (out/'STOP').exists() or time.monotonic()-start>=args.max_seconds or budget.stop(60)
    def stopped_setup():
        if not stop(): return False
        status=dict(complete=False,reason='Budget/STOP during setup',checkpoint_sha256=args.checkpoint_sha256,
            seconds_this_invocation=time.monotonic()-start,gpu_budget=budget.snapshot())
        atomic_json(out/'setup-status.json',status)
        if not (out/'summary.json').exists(): atomic_json(out/'summary.json',status)
        return True
    if stopped_setup(): return
    if file_sha(args.checkpoint)!=args.checkpoint_sha256: raise ValueError('Flow checkpoint hash mismatch')
    cache=LatentCache(args.cache)
    baseline, baseline_sha=load_vae_baseline(args.vae_baseline,cache)
    if stopped_setup(): return
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if state['cache_sha256']!=cache.sha256 or state['normalization']!=cache.stats:
        raise ValueError('Flow checkpoint/cache normalization binding mismatch')
    if state['source_vae_sha256']!=cache.manifest['source_checkpoint_sha256']:
        raise ValueError('Wrong source VAE')
    completed=state['completed_updates']
    identity=dict(flow_sha256=args.checkpoint_sha256,vae_sha256=cache.manifest['source_checkpoint_sha256'],
        cache_sha256=cache.sha256,uids=cache.uids,seed=args.seed,euler_steps=args.steps,
        completed_updates=completed,vae_baseline_summary_sha256=baseline_sha,
        noise_recipe='CPU torch.Generator; sha256(topology-eval/seed/uid) first 8 bytes little-endian; float32 Normal(0,1) [1,N,512]',
        sampling='local explicit Euler from independent Gaussian t=0 to t=1; no Encoder or GT topology input',
        vertices='original_GT_coordinates; not Vertex-generated',threshold='logit>0',
        evaluation_code_sha256=evaluator_sources())
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:
        raise ValueError('Resume evaluation binding mismatch')
    atomic_json(out/'identity.json',identity)
    results=[]; visualized=[]
    configure_math_backend()
    device=torch.device(args.device)
    if stopped_setup(): return
    state.pop('optimizer',None)
    model=PointCloudTopologyFlow(TopologyFlowConfig(**state['config']['model']))
    model.load_state_dict(state['model'],strict=True)
    del state
    if stopped_setup(): return
    model=model.to(device).eval().requires_grad_(False)
    if stopped_setup(): return
    vae=load_frozen_vae(args.vae_checkpoint,device)
    failure=None
    try:
        for uid in cache.uids:
            if stop(): break
            item=cache.get(uid);directory=out/uid; mesh_identity=dict(identity,uid=uid)
            # Only coordinates + real point condition enter generation; GT is read solely for metrics.
            def make_latent():
                return generate_latents(model,item['vertices'],item['points'],cache.stats,
                    seed_for_mesh(args.seed,uid),args.steps,item['point_mask'],stop)
            result=evaluate_item(vae,item,directory,mesh_identity,make_latent,device,stop)
            if result is None: break
            result['difference_from_vae']={variant:metric_difference(result,detail['rows'][uid])
                for variant,detail in baseline.items()}
            atomic_json(directory/'metrics.json',result);results.append(result)
            if export_visualizations(item,directory,mesh_identity,result,stop): visualized.append(uid)
    except InterruptedError:
        pass
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),uid=uid,
            completed_evaluations=len(results),gpu_budget=budget.snapshot())
        atomic_json(out/'failure.json',failure)
        raise
    finally:
        complete=len(results)==len(cache.uids)
        summary=dict(complete=complete,expected_meshes=len(cache.uids),evaluated_uids=[r['uid'] for r in results],
            identity=identity,seconds_this_invocation=time.monotonic()-start,gpu_budget=budget.snapshot(),
            metrics_sha256={r['uid']:file_sha(out/r['uid']/'metrics.json') for r in results},
            visualization_complete=len(visualized)==len(cache.uids),visualized_uids=visualized)
        if complete:
            summary['metrics']=complete_aggregate(results,cache.uids)
            summary['difference_from_vae']={variant:metric_difference(summary['metrics'],detail['metrics'])
                for variant,detail in baseline.items()}
        else:
            summary['note']='Incomplete; no full-selection score. Resume the identical checkpoint, seed and Euler steps.'
        if failure: summary['failure']=failure
        atomic_json(out/'summary.json',summary)
        print(json.dumps(summary,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--vae-checkpoint',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--vae-baseline',required=True,help='Completed evaluate_vae.py output on this identical cache')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--steps',type=int,default=50,choices=(50,),help='Fixed local explicit Euler baseline')
    parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--budget-file',help='Required for CUDA; shared ledger for every phase')
    args = parser.parse_args()
    if args.max_seconds<1: raise ValueError('Positive evaluation time limit required')
    from gpu_budget import GPUBudget
    try:
        with GPUBudget(args.budget_file,phase='flow_evaluation',device=args.device) as budget:
            run(args,budget)
    except Exception as error:
        atomic_json(Path(args.output)/'invocation-failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':main()
