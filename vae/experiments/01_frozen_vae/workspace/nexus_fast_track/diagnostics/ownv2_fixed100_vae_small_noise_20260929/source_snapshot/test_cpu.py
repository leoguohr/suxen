"""CPU-only migration checks; no real-model GPU or optimizer test is performed."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import copy
import itertools
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from data_objective import (load_dataset, epoch_batches, resume_cursor, negative_faces,
                            hard4_sums, hard4_chunks, edge_logits)
from native_models import Config
from runtime_fixed100 import next_state, build_checkpoint, restore_checkpoint, validated_config
from run_support import tensor_hash
from stream_faces import triangle_chunks, stream_faces, aggregate, atomic_json


def run(data_source=None):
    results = {}
    # Guard against accidentally adding a CUDA test to this suite.
    def forbidden(*args, **kwargs): raise AssertionError('CUDA is prohibited in CPU tests')
    torch.cuda.init = forbidden
    torch.cuda._lazy_init = forbidden
    torch.set_num_threads(1)
    uids = [f'item{i:03d}' for i in range(100)]
    item = dict(vertices=torch.zeros(7,3), gt_faces=torch.tensor([[0,1,2],[1,2,3]]))
    items = {u:dict(item, uid=u) for u in uids}
    for epoch in (0,1,19):
        batches = epoch_batches(uids, epoch)
        flat = [u for b in batches for u in b]
        assert len(batches) == 20 and all(len(b)==5 for b in batches)
        assert len(set(flat)) == 100 and set(flat) == set(uids)
    for completed in (0,1,19,20,21,399,400):
        cursor = next_state(items,uids,completed)
        saved = json.loads(json.dumps(cursor))
        assert saved == next_state(items,uids,completed)
        assert cursor['epoch'] == completed//20 and cursor['next_mesh_position'] == 5*(completed%20)
    results['synthetic_epoch_and_next_negative_resume'] = 'passed'
    # Same original negative sampler, fixed state per epoch/UID.
    a,ha = negative_faces(items[uids[0]],3); b,hb = negative_faces(items[uids[0]],3)
    assert torch.equal(a,b) and ha==hb and len(a)==3
    assert len(set(map(tuple,a.tolist())))==3 and not set(map(tuple,a.tolist())) & {(0,1,2),(1,2,3)}

    with tempfile.TemporaryDirectory() as td:
        d = Path(td); n=7
        adj = np.triu(np.ones((n,n),bool),1); adj[0,1]=False
        gt = np.array([[0,1,2],[0,2,3],[2,3,4]],np.int64)
        exact = [t for t in itertools.combinations(range(n),3) if all(adj[a,b] for a,b in itertools.combinations(t,2))]
        pieces=list(triangle_chunks(adj,chunk_size=4))
        generated=[tuple(x) for ids,_ in pieces for x in ids]
        assert generated==exact and len(set(generated))==len(exact)
        def score(ids): return np.where(np.sum(ids,axis=1)%2==0,1.,-1.).astype(np.float32)
        truth = set(map(tuple,gt)); pred={x for x in exact if sum(x)%2==0}
        calls=[0]
        def limited_score(ids): calls[0]+=1; return score(ids)
        identity=dict(checkpoint_sha256='synthetic-only',uid='synthetic')
        partial=stream_faces(adj,gt,limited_score,d/'resume',identity,stop=lambda:calls[0]>=2,chunk_size=4)
        assert not partial['complete'] and partial['shards']==2
        final=stream_faces(adj,gt,score,d/'resume',identity,chunk_size=4)
        assert final['complete'] and final['tp']==len(pred & truth) and final['fp']==len(pred-truth)
        assert final['fn']==len(truth-pred) and final['fn_missing']==1
        assert final['candidates']==len(exact)
        def crash(_): raise RuntimeError('synthetic crash between shard and commit')
        try: stream_faces(adj,gt,score,d/'crash',identity,chunk_size=4,after_shard=crash)
        except RuntimeError: pass
        else: raise AssertionError('Crash was not injected')
        assert json.loads((d/'crash/progress.json').read_text())['shards']==0
        repaired=stream_faces(adj,gt,score,d/'crash',identity,chunk_size=4)
        assert {k:repaired[k] for k in ('tp','fp','fn','tn','candidates')} == {k:final[k] for k in ('tp','fp','fn','tn','candidates')}
        seen=[]
        for p in sorted((d/'crash').glob('part-*.npz')):
            with np.load(p) as a: seen.extend(map(tuple,a['ids']))
        assert seen==exact
        try: stream_faces(adj,gt,score,d/'crash',dict(identity,checkpoint_sha256='different'),chunk_size=4)
        except AssertionError: pass
        else: raise AssertionError('Mixed checkpoints accepted')
        results['synthetic_streaming_exhaustive_resume_orphan_and_missing_gt'] = 'passed'

        # Test real torch model/AdamW/RNG state restoration on a small CPU model,
        # explicitly not a substitute for native V2 training validation.
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.cfg=Config(model_variant='B_v2_teacher_blocks')
                self.readout=torch.nn.Linear(3,2); self.log_variance=torch.nn.Linear(3,2)
                self.log_variance.requires_grad_(False)
            def forward(self,x): return self.readout(x)
        config=dict(seed=0,lr=1e-4,warmup_updates=100)
        model=Toy(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6,weight_decay=.01)
        participation={u:0 for u in uids}
        def update(m,o,step):
            for g in o.param_groups:g['lr']=1e-4*min(step/100,1)
            o.zero_grad(set_to_none=True)
            loss=m(torch.randn(5,3)).square().mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
            o.step()
        update(model,opt,1)
        for u in epoch_batches(uids,0)[0]:participation[u]+=1
        cp=build_checkpoint(model,opt,1,participation,config,dict(uids=uids),items,tensor_hash(model.log_variance.state_dict()))
        torch.save(cp,d/'toy.pt')
        update(model,opt,2)
        expected=copy.deepcopy(model.state_dict()); expected_opt=copy.deepcopy(opt.state_dict())
        resumed=Toy(); ropt=torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=1e-6,weight_decay=.01)
        loaded=torch.load(d/'toy.pt',weights_only=False,map_location='cpu')
        done,part=restore_checkpoint(loaded,resumed,ropt,items,dict(uids=uids),config)
        assert done==1 and part==participation
        update(resumed,ropt,2)
        assert all(torch.equal(expected[k],v) for k,v in resumed.state_dict().items())
        for i,state in expected_opt['state'].items():
            for k,v in state.items():assert torch.equal(v,ropt.state_dict()['state'][i][k])
        results['toy_cpu_AdamW_model_rng_exact_resume'] = 'passed'
        # Missing authorization fails without entering CUDA or creating a run.
        atomic_json(d/'disabled.json',dict(gpu_authorized=False))
        try: validated_config(d/'disabled.json')
        except AssertionError: pass
        else: raise AssertionError('Missing GPU authorization accepted')
        results['disabled_authorization_gate'] = 'passed'

        # Exercise the actual migrated entrypoint using CPU-only substitutes for
        # hardware and the large AE. This validates wiring, not native AE learning.
        import train_fixed100 as trainer
        class SmallAE(torch.nn.Module):
            def __init__(self,cfg):
                super().__init__();self.cfg=cfg
                self.edge=torch.nn.Linear(3,32);self.face=torch.nn.Linear(3,32)
                self.log_variance=torch.nn.Linear(3,2);self.log_variance.requires_grad_(False)
            def cuda(self):return self
            def forward(self,v,f,**kwargs):
                e=self.edge(v);a=self.face(v)
                return dict(edge=e-e.mean(0),face=a-a.mean(0),mu=v,latent=v)
        synthetic={}
        for u in uids:
            faces=torch.tensor([[0,1,2],[1,2,3]],dtype=torch.long)
            edges=torch.tensor([[0,1],[0,2],[1,2],[1,3],[2,3]])
            synthetic[u]=dict(uid=u,vertices=torch.randn(7,3),faces=faces,gt_faces=faces,edges=edges)
        cc=dict(gpu_authorized=True,continuous_authorized=False,gpu_uuid='GPU-cpu-test',
            max_new_updates=2,gpu_hours=1,evaluation_every=1,checkpoint_every=1,
            evaluate_during_training=False,evaluate_after_training=True,seed=0,
            model_variant='B_v2_teacher_blocks',mesh_per_update=5,warmup_updates=100,
            lr=1e-4,clip=1,weight_decay=.01,betas=[.9,.999],eps=1e-8,
            data_source='synthetic-only',run_directory=str(d/'entry-run'),mainline_lock=str(d/'main.lock'))
        atomic_json(d/'entry-config.json',cc)
        with patch.object(trainer,'verify_gpu',return_value='CPU surrogate; no hardware allocation'), \
             patch.object(trainer,'configure',side_effect=lambda seed:torch.manual_seed(seed)), \
             patch.object(trainer,'NativeTopologyAE',SmallAE), \
             patch.object(trainer,'load_dataset',return_value=(synthetic,dict(uids=uids))), \
             patch.object(trainer,'move_item',side_effect=lambda item,device:item), \
             patch.object(trainer,'evaluate',return_value=dict(complete=True)) as eval_mock, \
             patch.object(torch.cuda,'set_device'),patch.object(torch.cuda,'synchronize'), \
             patch.object(torch.cuda,'reset_peak_memory_stats'), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=0), \
             patch.object(torch.cuda,'max_memory_reserved',return_value=0):
            trainer.main(str(d/'entry-config.json'),'fresh')
            assert eval_mock.call_count==1 and eval_mock.call_args.args[3]['completed_updates']==2
            first=json.loads((d/'entry-run/recovery-latest.json').read_text())
            loaded=torch.load(first['path'],map_location='cpu',weights_only=False)
            assert loaded['completed_updates']==2 and sum(loaded['participation'].values())==10
            before=tensor_hash(loaded['model'])
            initial=torch.load(d/'entry-run/recovery-a.pt',map_location='cpu',weights_only=False)
            probe=SmallAE(Config(model_variant='B_v2_teacher_blocks'))
            probe_opt=torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad],lr=1e-4*.01,weight_decay=.01)
            assert restore_checkpoint(initial,probe,probe_opt,synthetic,dict(uids=uids),cc)[0]==0
            assert not probe_opt.state
            trainer.main(str(d/'entry-config.json'),'resume')
            assert eval_mock.call_count==2 and eval_mock.call_args.args[3]['completed_updates']==2
            second=json.loads((d/'entry-run/recovery-latest.json').read_text())
            loaded=torch.load(second['path'],map_location='cpu',weights_only=False)
            assert loaded['completed_updates']==2 and tensor_hash(loaded['model'])==before
            logs=(d/'entry-run/updates.jsonl').read_text().splitlines()
            assert len(logs)==2 and [json.loads(x)['update'] for x in logs]==[1,2]
        results['migrated_entrypoint_tiny_CPU_fresh_stop_resume_no_replay'] = 'passed'

    emb=torch.randn(8,32,requires_grad=True);pairs=torch.triu_indices(8,8,1).T
    labels=torch.arange(len(pairs))%3==0
    n,c=hard4_sums(edge_logits(emb,pairs),labels);full=(n/c.clamp_min(1)).sum()/4
    full_grad=torch.autograd.grad(full,emb,retain_graph=True)[0]
    chunked,_=hard4_chunks(edge_logits,emb,pairs,labels,5)
    chunk_grad=torch.autograd.grad(chunked,emb)[0]
    assert torch.allclose(full,chunked,rtol=1e-6,atol=1e-6)
    assert torch.allclose(full_grad,chunk_grad,rtol=1e-5,atol=1e-6)
    results['Hard4_chunk_reduction_preserved_CPU'] = 'passed'
    if data_source:
        real,manifest=load_dataset(data_source)
        ids=manifest['uids']; assert len(epoch_batches(ids,0))==20
        cursor=next_state(real,ids,19)
        assert cursor==next_state(real,ids,19)
        assert set(u for batch in epoch_batches(ids,0) for u in batch)==set(ids)
        results['real_fixed100_data'] = dict(passed=True,**manifest)
    return dict(passed=True,tests=results,CUDA_forward_backward=False,native_model_training=False,
                real_data_test_executed=bool(data_source))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-source');p.add_argument('--output',required=True)
    args=p.parse_args();result=run(args.data_source);atomic_json(args.output,result)
    print(json.dumps(dict(passed=result['passed'],tests=list(result['tests']),real_data_test_executed=result['real_data_test_executed'])))
