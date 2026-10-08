"""Necessary CPU checks only; no optimizer steps or historical checkpoints."""
import copy
import importlib.util
import json
import sys
from dataclasses import replace
import numpy as np
import torch
from run_support import ROOT, configure, write, code_hashes
from native_models import Config, NativeTopologyAE, Graph, VARIANTS
from data_objective import hard4_sums, hard4_chunks, edge_logits, face_logits, negative_faces, epoch_batches, load_dataset


def test_loss():
    logits = torch.tensor([-2.,3.,-1.,4.,0.], requires_grad=True)
    y = torch.tensor([0.,1.,1.,0.,1.])
    ns,cs = hard4_sums(logits,y)
    assert cs.tolist() == [1,1,1,2]
    loss = (ns/cs.clamp_min(1)).sum()/4
    expected = sum(torch.nn.functional.binary_cross_entropy_with_logits(logits[m],y[m])
        for m in [torch.tensor([1]),torch.tensor([0]),torch.tensor([3]),torch.tensor([2,4])])/4
    assert torch.equal(loss,expected)
    gradient, = torch.autograd.grad(loss,logits)
    assert torch.all(gradient[y==1]<0) and torch.all(gradient[y==0]>0)
    z = torch.tensor([1.,2.],requires_grad=True); n,c = hard4_sums(z,torch.ones(2))
    value = (n/c.clamp_min(1)).sum()/4
    assert torch.equal(value,torch.nn.functional.binary_cross_entropy_with_logits(z,torch.ones(2))/4)
    e = torch.randn(9,32,requires_grad=True)
    for scorer, ids in [(edge_logits,torch.triu_indices(9,9,1).T),
                        (face_logits,torch.combinations(torch.arange(9),r=3))]:
        labels = torch.arange(len(ids))%3 == 0
        losses, gradients = [],[]
        for chunk in [4,1000]:
            l,_ = hard4_chunks(scorer,e,ids,labels,chunk)
            losses.append(l); gradients.append(torch.autograd.grad(l,e)[0])
        torch.testing.assert_close(losses[0],losses[1],rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(gradients[0],gradients[1],rtol=3e-6,atol=3e-6)


def map_reference(model, ref):
    state = model.state_dict(); target = ref.state_dict(); mapping = {}
    for name in target:
        dest = name
        if name.startswith('encoder_blocks.'):
            dest = dest.replace('.graph.message_norm.','.graph_norm.')
            dest = dest.replace('.transformer.attn_norm.','.transformer.norm1.')
            dest = dest.replace('.transformer.attn.','.transformer.self_attn.')
            dest = dest.replace('.transformer.ffn_norm.','.transformer.norm2.')
            dest = dest.replace('.transformer.ffn.0.','.transformer.linear1.')
            dest = dest.replace('.transformer.ffn.3.','.transformer.linear2.')
        elif name.startswith('decoder_blocks.'):
            dest = dest.replace('.attn_norm.','.norm.').replace('.attn.','.attention.').replace('.ffn.3.','.ffn.2.')
        assert dest in state and target[name].shape == state[dest].shape,(name,dest)
        mapping[name] = state[dest]
    ref.load_state_dict(mapping,strict=True)


def test_models():
    vertices = torch.randn(8,3)
    faces = torch.tensor([[0,1,2],[0,2,3],[0,3,4],[0,4,5],[0,5,6],[0,6,7],[1,6,7]])
    config = Config(encoder_width=16,latent_width=16,decoder_width=32,encoder_composite_blocks=2,
        decoder_blocks=2,heads=4,activation_checkpointing=False)
    results = {}
    for variant in VARIANTS:
        m = NativeTopologyAE(replace(config,model_variant=variant))
        a = m(vertices,faces,sample_latent=False)
        assert a['latent'] is a['mu']
        m.eval(); b = m(vertices,faces,sample_latent=False)
        assert all(torch.equal(a[k],b[k]) for k in a)
        m.train(); clone = NativeTopologyAE(replace(m.cfg,activation_checkpointing=True))
        clone.load_state_dict(m.state_dict(),strict=True)
        c = clone(vertices,faces,sample_latent=False,graph=Graph.from_faces(faces,len(vertices)))
        assert all(torch.equal(a[k],c[k]) for k in a)
        def loss(x):return x['mu'].square().mean()+x['edge'].square().mean()+x['face'].square().mean()
        loss(a).backward();loss(c).backward()
        for (name,p),(_,q) in zip(m.named_parameters(),clone.named_parameters()):
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            else: assert p.grad is None and q.grad is None
        if variant == VARIANTS[1]:
            spec=importlib.util.spec_from_file_location('user_v2_reference',ROOT/'reference/student_v2_reference.py')
            refmod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=refmod;spec.loader.exec_module(refmod)
            cfg={k:v for k,v in m.cfg.to_dict().items() if k not in ['model_variant','activation_checkpointing']}
            ref=refmod.OwnTopologyAEV2(refmod.Config(**cfg));map_reference(m,ref)
            d=ref(vertices,faces,sample_latent=False)
            for k in a:torch.testing.assert_close(a[k],d[k],rtol=1e-5,atol=2e-6)
            assert all(p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith('decoder_blocks.') and '.ffn.' in n)
        perm=torch.tensor([2,0,7,4,5,1,3,6]);inverse=torch.argsort(perm)
        e=m(vertices[perm],inverse[faces],sample_latent=False)
        for k in a:torch.testing.assert_close(a[k][perm],e[k],rtol=3e-5,atol=3e-6)
        assert not any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.modules())
        results[variant]=dict(train_eval_mu_equal=True,checkpoint_gradients_equal=True,native_no_hooks=True)
    for variant,expected in zip(VARIANTS,[112212544,246575680]):
        with torch.device('meta'):large=NativeTopologyAE(Config(model_variant=variant))
        count=sum(p.numel() for p in large.parameters() if p.requires_grad)
        assert count==expected,(variant,count,expected)
        results[variant]['trainable_parameters']=count
    return results


def test_data():
    items,manifest=load_dataset(ROOT/'data',ROOT/'pools')
    uids=manifest['uids']
    for epoch in [0,1,1999]:
        batches=epoch_batches(uids,epoch)
        assert sorted(sum(batches,[]))==uids
        for uid in uids:
            a,h=negative_faces(items[uid],epoch);b,h2=negative_faces(items[uid],epoch)
            assert h==h2 and torch.equal(a,b)
            assert len(a)==int(np.ceil(1.5*len(items[uid]['gt_faces'])))
            assert not set(map(tuple,a.tolist())) & set(map(tuple,items[uid]['gt_faces'].tolist()))
            assert len(set(map(tuple,a.tolist())))==len(a) and torch.all(a[:,0]<a[:,1]) and torch.all(a[:,1]<a[:,2])
    return manifest


if __name__=='__main__':
    configure();test_loss();models=test_models();data=test_data()
    write(ROOT/'repro_outputs/CORE_TESTS.json',dict(passed=True,optimizer_updates=0,models=models,
        data=data,hard4_groups_zero_threshold_empty_chunk_gradients=True,paired_stateless_sampling=True,
        code_sha256=code_hashes()))
    print(json.dumps(dict(passed=True,models=models,totals=data['totals'])))
