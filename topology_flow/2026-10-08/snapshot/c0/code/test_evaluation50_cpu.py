"""Targeted synthetic checks for fixed VAE noise, full-selection scoring and artifact resume."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from _faces_reference import atomic_json, file_sha
from _vae_reference import NativeTopologyAE, Config
from evaluate_flow import (seed_for_mesh, evaluate_item, export_visualizations, with_metrics,
    complete_aggregate, metric_difference, load_vae_baseline)
from evaluate_vae import posterior_seed_for_mesh, baseline_latent


def main():
    torch.set_num_threads(1);torch.manual_seed(87)
    tests=[]
    item=dict(mu=torch.randn(4,512),logvar=torch.zeros(4,512))
    rng=torch.get_rng_state().clone()
    draw=baseline_latent(item,0,'synthetic','cpu')
    torch.testing.assert_close(draw,baseline_latent(item,0,'synthetic','cpu'),rtol=0,atol=0)
    torch.testing.assert_close(item['mu'],baseline_latent(item,None,'synthetic','cpu'),rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    assert not torch.equal(draw,baseline_latent(item,1,'synthetic','cpu'))
    assert posterior_seed_for_mesh(0,'synthetic')!=seed_for_mesh(0,'synthetic')
    tests.append('mu unchanged; two fixed posterior draws repeat exactly and leave global RNG unchanged; Flow domain differs')

    uids=[f'synthetic_{i:02d}' for i in range(50)]
    rows=[with_metrics(dict(complete=True,uid=uid,edge=dict(tp=3,fp=0,fn=0,tn=0),
        face=dict(tp=1,fp=0,fn=0,tn=0),face_fn_missing=0,face_fn_present=0)) for uid in uids]
    total=complete_aggregate(rows,uids)
    assert total['strict50_under_this_noise'] and total['edge_strict_count']==50
    for bad in (rows[:49],rows[::-1],rows[:49]+[rows[0]]):
        try: complete_aggregate(bad,uids);raise AssertionError('Invalid selection aggregated')
        except ValueError: pass
    changed=copy.deepcopy(rows);changed[-1]['face']['fn']=1;with_metrics(changed[-1])
    result=complete_aggregate(changed,uids)
    assert not result['strict50_under_this_noise'] and result['joint_strict_count']==49
    assert metric_difference(result,total)['face']['fn']==1
    tests.append('49, duplicated and reordered selections rejected; strict50 requires all four FP/FN zero under one identity')

    with tempfile.TemporaryDirectory(prefix='topology-evaluation50-cpu-') as tmp:
        root=Path(tmp)
        cfg=Config(model_variant='B_v2_teacher_blocks',encoder_width=24,decoder_width=24,heads=3,
            latent_width=512,encoder_composite_blocks=1,decoder_blocks=1,activation_checkpointing=False)
        vae=NativeTopologyAE(cfg).eval().requires_grad_(False)
        def forbidden(*args,**kwargs): raise AssertionError('Encoder entered baseline evaluation')
        vae.vertex_input.forward=forbidden;vae.face_input.forward=forbidden
        item.update(vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
            edges=torch.tensor([[0,1],[0,2],[1,2]]),faces=torch.tensor([[0,1,2]]))
        directory=root/'mesh';identity=dict(uid='synthetic',checkpoint='synthetic')
        result=evaluate_item(vae,item,directory,identity,lambda:baseline_latent(item,0,'synthetic','cpu'),
            torch.device('cpu'),lambda:False)
        assert result['complete']
        again=evaluate_item(vae,item,directory,identity,forbidden,torch.device('cpu'),lambda:False)
        assert result==again
        assert not export_visualizations(item,directory,identity,result,lambda:True)
        assert export_visualizations(item,directory,identity,result)
        vertices=[];edges=[];faces=[]
        for line in (directory/'predicted.obj').read_text().splitlines():
            fields=line.split()
            if fields[0]=='v': vertices.append([float(x) for x in fields[1:]])
            if fields[0]=='l': edges.append(tuple(int(x)-1 for x in fields[1:]))
            if fields[0]=='f': faces.append(tuple(int(x)-1 for x in fields[1:]))
        np.testing.assert_array_equal(vertices,item['vertices'].numpy())
        with np.load(directory/'edge-and-face-embedding.npz') as a:
            assert edges==list(map(tuple,a['pairs'][a['logits']>0].tolist()))
        progress=json.loads((directory/'face-shards/progress.json').read_text());expected=[]
        for index in range(progress['shards']):
            with np.load(directory/f'face-shards/part-{index:08d}.npz') as a:
                expected.extend(map(tuple,a['ids'][a['logits']>0].tolist()))
        assert faces==expected and len(faces)==result['face']['tp']+result['face']['fp']
        assert export_visualizations(item,directory,identity,result)
        tests.append('actual decoder-only baseline resume does not redraw; OBJ exactly preserves vertices and every predicted edge/face')

        cache=SimpleNamespace(sha256='synthetic-cache',manifest={'source_checkpoint_sha256':'synthetic-vae'},uids=uids)
        base=root/'baseline';base_identity=dict(cache_sha256=cache.sha256,vae_sha256='synthetic-vae',uids=uids)
        for row in rows:
            atomic_json(base/'mu'/row['uid']/'metrics.json',dict(row,identity=dict(base_identity,variant='mu',uid=row['uid'])))
        detail=dict(metrics=total,metrics_sha256={uid:file_sha(base/'mu'/uid/'metrics.json') for uid in uids})
        atomic_json(base/'summary.json',dict(complete=True,identity=base_identity,variants={'mu':detail}))
        loaded,_=load_vae_baseline(base,cache)
        assert loaded['mu']['metrics']==total
        atomic_json(base/'summary.json',dict(complete=False,identity=base_identity,variants={'mu':{'metrics':total}}))
        try: load_vae_baseline(base,cache);raise AssertionError('Incomplete baseline accepted')
        except ValueError: pass
        tests.append('Flow comparison requires a complete baseline bound to all 50 ordered UIDs and the same cache/VAE')
    evidence=dict(passed=True,device='cpu',tests=tests,real_meshes_evaluated=0,gpu_used=False,
        real_source_checkpoint_loaded=False,temporary_fixtures_removed=True)
    atomic_json(Path(__file__).resolve().parent.parent/'evidence/evaluation50_cpu_tests.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
