"""Temporary synthetic fixtures only. No production checkpoints or GPU access."""
import copy
import itertools
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from _faces_reference import atomic_json, atomic_npz, file_sha, stream_faces, aggregate
from _fixed100_reference import array_sha, edge_logits, face_logits
from _vae_reference import NativeTopologyAE, Config
from _vertex_reference import farthest_point_sample
from export_cache import export_item
from latent_data import posterior_statistics, transform_latent, LatentCache
from flow_training import build_training_state, train_update, checkpoint_state, atomic_checkpoint, restore_training, batch_cursor, rng_state
from sampling import integrate_euler, generate_latents
from topology_flow import math_context
from vae_codec import load_frozen_vae, decode_latents
from evaluate_flow import evaluate_embeddings, seed_for_mesh


def same(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)


def main():
    torch.set_num_threads(1);torch.manual_seed(31);torch.use_deterministic_algorithms(True)
    tests=[]
    # The small mesh and large mesh get equal mass: mean=5, variance=26 (including q variance=1).
    mu1=torch.zeros(2,512);mu2=torch.full((20,512),10.)
    stats=posterior_statistics([(mu1,torch.zeros_like(mu1)),(mu2,torch.zeros_like(mu2))])
    torch.testing.assert_close(torch.tensor(stats['mean']),torch.full((512,),5.))
    torch.testing.assert_close(torch.tensor(stats['std']).square(),torch.full((512,),26.))
    x=torch.randn(2,7,512);mask=torch.tensor([[True]*7,[True]*3+[False]*4])
    y=transform_latent(x,stats,mask=mask)
    restored=transform_latent(y,stats,True,mask)
    torch.testing.assert_close(restored[mask],x[mask],atol=1e-6,rtol=1e-6)
    assert not restored[~mask].count_nonzero()
    tests.append('equal-mesh posterior moments include variance; reversible channel scaling and padding')
    initial=torch.randn(2,7,512);target=torch.randn_like(initial)
    expected=target.masked_fill(~mask[:,:,None],0)
    sampled=integrate_euler(lambda z,t:target-initial,initial,9,mask)
    torch.testing.assert_close(sampled,expected,atol=1e-6,rtol=1e-6)
    tests.append('Euler noise0 to data1 velocity sign and endpoint with an oracle field')
    with tempfile.TemporaryDirectory(prefix='topology-flow-cpu-') as tmp:
        root=Path(tmp)
        vae_cfg=Config(model_variant='B_v2_teacher_blocks',encoder_width=24,decoder_width=24,heads=3,
            latent_width=512,encoder_composite_blocks=1,decoder_blocks=2,activation_checkpointing=False)
        vae=NativeTopologyAE(vae_cfg).eval().requires_grad_(False)
        path=root/'synthetic-vae.pt'
        torch.save(dict(model=vae.state_dict(),model_config=vae.cfg.to_dict()),path)
        source_sha=file_sha(path)
        loaded=load_frozen_vae(path,'cpu',source_sha)
        same(loaded.state_dict(),vae.state_dict())
        try:load_frozen_vae(path,'cpu','0'*64);raise AssertionError('Wrong source hash accepted')
        except ValueError:pass
        cache_root=root/'cache';cache_root.mkdir()
        records=[];post=[];uids=[f'synthetic_{i}' for i in range(4)]
        for i,uid in enumerate(uids):
            vertices=torch.randn(i+4,3)
            faces=torch.tensor([[0,1,2],[1,2,3]])
            edges=torch.unique(torch.sort(torch.cat((faces[:,[0,1]],faces[:,[0,2]],faces[:,[1,2]])),dim=1).values,dim=0)
            points=np.random.default_rng(i).standard_normal((9,6)).astype(np.float32)
            point_mask=np.array([True]*8+[False])
            arrays,errors=export_item(loaded,dict(vertices=vertices,faces=faces,edges=edges),points,point_mask)
            assert all(e==0 for e in errors.values())
            f=cache_root/(uid+'.npz');atomic_npz(f,**arrays)
            records.append(dict(uid=uid,file=f.name,sha256=file_sha(f),vertices_sha256=array_sha(arrays['vertices'])))
            post.append((torch.from_numpy(arrays['mu']),torch.from_numpy(arrays['logvar'])))
        manifest=dict(complete=True,uids=uids,source_checkpoint_sha256=source_sha,records=records,
            normalization=posterior_statistics(post),synthetic_fixture=True)
        atomic_json(cache_root/'manifest.json',manifest)
        cache=LatentCache(cache_root,expected_meshes=4,source_sha256=source_sha)
        tests.append('hash-bound synthetic frozen VAE load, exact encode/decode and cache validation')
        config=dict(model=dict(hidden_dim=24,num_layers=1,num_heads=3,condition_dim=24,condition_layers=1,
            condition_heads=3,condition_tokens=3,rope_scale=1.,latent_dim=512,recompute=True),
            seed=11,batch_meshes=2,lr=1e-3,weight_decay=.01,warmup_updates=2,clip=1.,
            max_updates=3,max_seconds=120,save_every=1)
        model,opt,rngs=build_training_state(config,torch.device('cpu'))
        fps={u:farthest_point_sample(cache.get(u)['points'][None,:,:3],3,cache.get(u)['point_mask'][None]) for u in uids}
        # Explicit new block-wise condition path equals the unchanged reference.
        item=cache.get(uids[0]);model.eval()
        actual=model.encode_condition(item['points'][None],item['point_mask'][None],fps[uids[0]])
        with math_context('cpu'):
            reference=model.condition_encoder(item['points'][None],item['point_mask'][None],fps_indices=fps[uids[0]])
        same(actual,reference);model.train()
        tests.append('condition-encoder reference equality with explicit recompute context')
        orders=[batch_cursor(uids,k,2,11)['next_uids'] for k in range(2)]
        assert sorted(sum(orders,[]))==sorted(uids)
        first=train_update(model,opt,rngs,cache,config,0,fps=fps)
        assert first['forward_at_completed_updates']==0 and first['completed_updates']==1 and first['adam_steps']==[1]
        assert first['lr']==.0005
        cp=root/'boundary.pt'
        atomic_checkpoint(cp,checkpoint_state(model,opt,rngs,cache,config,1,0.,{'test':'synthetic'}))
        state=torch.load(cp,weights_only=False)
        uninterrupted=[]
        for step in (1,2):uninterrupted.append(train_update(model,opt,rngs,cache,config,step,fps=fps))
        resumed,ropt,rrng=build_training_state(config,torch.device('cpu'))
        restore_training(state,resumed,ropt,rrng,cache,config,{'test':'synthetic'})
        for index,step in enumerate((1,2)):
            same(train_update(resumed,ropt,rrng,cache,config,step,fps=fps),uninterrupted[index])
        same(model.state_dict(),resumed.state_dict());same(opt.state_dict(),ropt.state_dict())
        same(rng_state(rngs,torch.device('cpu')),rng_state(rrng,torch.device('cpu')))
        tests.append('five synthetic AdamW updates: boundary restore exactly matches uninterrupted weights, moments, LR, draws and cursor')
        before=copy.deepcopy(checkpoint_state(resumed,ropt,rrng,cache,config,3,0.,{'test':'synthetic'}))
        calls=0
        def stop_second():
            nonlocal calls
            calls+=1
            return calls>=2
        assert train_update(resumed,ropt,rrng,cache,config,3,stop_second,fps) is None
        after=checkpoint_state(resumed,ropt,rrng,cache,config,3,0.,{'test':'synthetic'})
        same(before,after)
        assert all(p.grad is None for p in resumed.parameters())
        tests.append('STOP during accumulation discards gradients and restores RNG without an optimizer update')
        resumed.eval();snapshot=rng_state(rrng,torch.device('cpu'))
        seed=seed_for_mesh(123,uids[0])
        generated=generate_latents(resumed,item['vertices'],item['points'],cache.stats,seed,3,item['point_mask'])
        again=generate_latents(resumed,item['vertices'],item['points'],cache.stats,seed,3,item['point_mask'])
        same(generated,again);same(snapshot,rng_state(rrng,torch.device('cpu')))
        def forbidden(*a,**kw):raise AssertionError('Encoder entered generation')
        loaded.vertex_input.forward=forbidden;loaded.face_input.forward=forbidden
        out=decode_latents(loaded,generated)
        assert out['edge'].shape==(len(item['vertices']),32)
        assert all(p.grad is None for p in loaded.parameters())
        tests.append('reproducible Gaussian-only generation and decoder-only inference leave training RNG and frozen VAE untouched')

        # All edges except (0,1); GT face (0,1,2) must remain FN regardless of its score.
        edge=torch.zeros(6,32);edge[:,0]=torch.tensor([0.,0.,2.,3.,4.,5.])
        face=torch.zeros(6,32);face[:,:3]=torch.randn(6,3)
        gt_faces=torch.tensor([[0,1,2],[3,4,5]])
        gt_edges=torch.unique(torch.sort(torch.cat((gt_faces[:,[0,1]],gt_faces[:,[0,2]],gt_faces[:,[1,2]])),dim=1).values,dim=0)
        directory=root/'eval';identity=dict(uid='synthetic',checkpoint='fixture')
        def stop_after_shard():return (directory/'face-shards'/'part-00000000.npz').exists()
        partial=evaluate_embeddings(edge,face,gt_edges,gt_faces,directory,identity,stop_after_shard,2)
        assert not partial['complete']
        metrics=evaluate_embeddings(edge,face,gt_edges,gt_faces,directory,identity,chunk_size=2)
        pairs=torch.combinations(torch.arange(6),r=2);ep=edge_logits(edge,pairs)>0
        aset={tuple(x) for x in pairs[ep].tolist()};gset={tuple(x) for x in gt_faces.tolist()}
        candidates=[t for t in itertools.combinations(range(6),3) if all(e in aset for e in itertools.combinations(t,2))]
        pred={tuple(candidates[i]) for i,x in enumerate(face_logits(face,torch.tensor(candidates))) if x>0}
        assert metrics['face']['tp']==len(pred&gset) and metrics['face']['fp']==len(pred-gset)
        assert metrics['face']['fn']==len(gset-pred) and metrics['face_fn_missing']==1
        assert metrics['actual_face_candidates']==len(candidates)
        saved_ids=[]
        for part in sorted((directory/'face-shards').glob('part-*.npz')):
            with np.load(part) as a:saved_ids.extend(map(tuple,a['ids'].tolist()))
        assert len(saved_ids)==len(set(saved_ids))==len(candidates) and set(saved_ids)==set(candidates)
        same(metrics,evaluate_embeddings(edge,face,gt_edges,gt_faces,directory,identity,chunk_size=2))
        tests.append('complete streamed actual triangles equal brute force; resume has no duplicates and missing GT is FN')
        # Simulate a crash after an NPZ write but before progress commits it.
        adj=np.triu(np.ones((6,6),dtype=np.bool_),1);d=root/'orphan'
        def crash(path):raise RuntimeError('injected crash')
        try:stream_faces(adj,gt_faces.numpy(),lambda x:np.ones(len(x)),d,identity,chunk_size=2,after_shard=crash)
        except RuntimeError:pass
        result=stream_faces(adj,gt_faces.numpy(),lambda x:np.ones(len(x)),d,identity,chunk_size=2)
        assert result['complete'] and result['candidates']==20 and result['tp']==2 and result['fp']==18
        tests.append('orphan shard is overwritten on resume, never double-counted')
        a=dict(complete=True,uid='a',edge=dict(tp=1,fp=0,fn=0,tn=0),face=dict(tp=1,fp=0,fn=0,tn=0),face_fn_missing=0,face_fn_present=0)
        b=dict(complete=True,uid='b',edge=dict(tp=1,fp=8,fn=0,tn=0),face=dict(tp=1,fp=8,fn=0,tn=0),face_fn_missing=0,face_fn_present=0)
        total=aggregate([a,b]);assert total['face']['micro_f1']==1/3 and total['joint_perfect']==1
        tests.append('aggregate sums confusion counts before micro-F1 and keeps strict success separate')
    evidence=dict(passed=True,device='cpu',torch_version=torch.__version__,tests=tests,
        synthetic_optimizer_updates=5,real_mesh_optimizer_updates=0,real_source_checkpoint_loaded=False,
        real100_exported=False,gpu_used=False,temporary_fixtures_removed=True)
    path=Path(__file__).resolve().parent.parent/'evidence/pipeline_cpu_tests.json'
    atomic_json(path,evidence);print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
