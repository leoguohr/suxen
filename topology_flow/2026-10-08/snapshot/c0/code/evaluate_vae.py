"""Frozen OwnAE-v2 cache reconstruction baseline: mu and two fixed posterior draws."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from _faces_reference import atomic_json, file_sha
from latent_data import LatentCache
from topology_flow import configure_math_backend
from vae_codec import load_frozen_vae
from evaluate_flow import evaluate_item, export_visualizations, evaluator_sources, complete_aggregate


def posterior_seed_for_mesh(seed,uid):
    return int.from_bytes(hashlib.sha256(f'topology-vae-baseline/{seed}/{uid}'.encode()).digest()[:8],'little')


def baseline_latent(item,seed,uid,device):
    """Private CPU noise; does not consume any global or training random stream."""
    mu=item['mu']
    if seed is None: return mu.to(device)
    generator=torch.Generator().manual_seed(posterior_seed_for_mesh(seed,uid))
    noise=torch.randn(mu.shape,generator=generator,dtype=torch.float32)
    return (mu+(item['logvar']*.5).exp()*noise).to(device)


def run(args,budget):
    start=time.monotonic()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def stop():
        return (out/'STOP').exists() or time.monotonic()-start>=args.max_seconds or budget.stop(60)
    cache=LatentCache(args.cache)
    variants={'mu':None,'posterior_seed0':0,'posterior_seed1':1}
    identity=dict(vae_sha256=cache.manifest['source_checkpoint_sha256'],cache_sha256=cache.sha256,
        uids=cache.uids,variants=variants,
        noise_recipe='CPU torch.Generator; sha256(topology-vae-baseline/seed/uid) first 8 bytes little-endian; float32 Normal(0,1) [N,512]',
        decoding='frozen Decoder of cached mu or mu+exp(0.5*effective_logvar)*fixed_noise; no topology repair',
        vertices='original_GT_coordinates; local vertex numbering unchanged',threshold='logit>0',
        evaluation_code_sha256=evaluator_sources())
    if (out/'identity.json').exists() and json.loads((out/'identity.json').read_text())!=identity:
        raise ValueError('Resume VAE baseline binding mismatch')
    atomic_json(out/'identity.json',identity)
    details={};failure=None;uid=None;variant=None
    try:
        configure_math_backend();device=torch.device(args.device)
        if stop(): return
        vae=load_frozen_vae(args.vae_checkpoint,device)
        for variant,seed in variants.items():
            results=[];visualized=[]
            for uid in cache.uids:
                if stop(): break
                item=cache.get(uid);directory=out/variant/uid;mesh_identity=dict(identity,variant=variant,uid=uid)
                try:
                    result=evaluate_item(vae,item,directory,mesh_identity,
                        lambda:baseline_latent(item,seed,uid,device),device,stop)
                except InterruptedError:
                    break
                if result is None: break
                results.append(result)
                if export_visualizations(item,directory,mesh_identity,result,stop): visualized.append(uid)
            detail=dict(complete=len(results)==len(cache.uids),evaluated_uids=[r['uid'] for r in results],
                metrics_sha256={r['uid']:file_sha(out/variant/r['uid']/'metrics.json') for r in results},
                visualization_complete=len(visualized)==len(cache.uids),visualized_uids=visualized)
            if detail['complete']: detail['metrics']=complete_aggregate(results,cache.uids)
            else: detail['note']='Incomplete; no full-selection score or substitution for the missing UIDs.'
            details[variant]=detail
            atomic_json(out/variant/'summary.json',dict(identity=dict(identity,variant=variant),**detail))
            if not detail['complete']: break
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error),uid=uid,variant=variant)
        atomic_json(out/'failure.json',failure)
        raise
    finally:
        complete=list(details)==list(variants) and all(d['complete'] for d in details.values())
        summary=dict(complete=complete,expected_meshes=len(cache.uids),identity=identity,variants=details,
            seconds_this_invocation=time.monotonic()-start,gpu_budget=budget.snapshot())
        if not complete: summary['note']='Baseline incomplete; committed per-UID metrics and triangle cursors remain resumable.'
        if failure: summary['failure']=failure
        atomic_json(out/'summary.json',summary)
        print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vae-checkpoint',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--max-seconds',type=int,required=True)
    parser.add_argument('--budget-file',help='Required for CUDA; shared ledger for every phase')
    args=parser.parse_args()
    if args.max_seconds<1: raise ValueError('Positive evaluation time limit required')
    from gpu_budget import GPUBudget
    try:
        with GPUBudget(args.budget_file,phase='vae_baseline',device=args.device) as budget:
            run(args,budget)
    except Exception as error:
        atomic_json(Path(args.output)/'invocation-failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':main()
