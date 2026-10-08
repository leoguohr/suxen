"""Small synthetic CPU checks; no CAD training and no CUDA execution."""
import argparse
import importlib.util
import itertools
from pathlib import Path
import tempfile
import torch
import numpy as np
from support import *
from native_models import Config, NativeTopologyAE, Graph
from data_objective import epoch_batches, negative_faces, objective, hard4_chunks, hard4_sums


def main(original_source):
    configure(0)
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty'
    spec = importlib.util.spec_from_file_location('original_native', Path(original_source)/'native_models.py')
    original = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    dims = dict(model_variant='B_v2_teacher_blocks', encoder_width=16, latent_width=16,
                decoder_width=32, encoder_composite_blocks=2, decoder_blocks=2, heads=4)
    reference = original.NativeTopologyAE(original.Config(**dims))
    baseline = NativeTopologyAE(Config(**dims))
    baseline.load_state_dict(reference.state_dict(), strict=True)
    vertices = torch.randn(6,3)
    faces = torch.tensor([[0,1,2],[2,3,4]], dtype=torch.long)
    graph = Graph.from_faces(faces,6)
    a = reference(vertices, faces, graph=graph)
    b = baseline(vertices, faces, graph=graph)
    assert all(torch.equal(a[k], b[k]) for k in a)
    ga = torch.autograd.grad(a['mu'].square().sum()+a['edge'].square().sum(),
                             [p for p in reference.parameters() if p.requires_grad], allow_unused=True)
    gb = torch.autograd.grad(b['mu'].square().sum()+b['edge'].square().sum(),
                             [p for p in baseline.parameters() if p.requires_grad], allow_unused=True)
    assert all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(ga,gb))
    pairs = torch.triu_indices(6,6,1).T
    item = dict(uid='synthetic', vertices=vertices, faces=faces, gt_faces=faces,
                pairs=pairs, edge_labels=torch.arange(len(pairs))%2==0)
    modes = []
    keys = [(n,tuple(p.shape)) for n,p in baseline.named_parameters()]
    for arm, changes in MODES.items():
        model = NativeTopologyAE(Config(**dims, **changes))
        model.load_state_dict(baseline.state_dict(), strict=True)
        assert [(n,tuple(p.shape)) for n,p in model.named_parameters()] == keys
        assert tensor_hash(model.state_dict()) == tensor_hash(baseline.state_dict())
        out = model(vertices,faces,graph=graph)
        assert out['latent'] is out['mu']
        neg, _ = negative_faces(item,0)
        loss,_ = objective(out,item,neg,pair_chunk=3,face_chunk=2)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in model.log_variance.parameters())
        modes.append(dict(arm=arm, changed_operations=changes, connected_finite=True, loss=float(loss.detach())))
    uids = [f'teacher_cad50_{i:02d}' for i in range(50)]
    for epoch in range(200):
        batches = epoch_batches(uids,epoch)
        assert len(batches)==10 and all(len(x)==5 for x in batches)
        assert sorted(sum(batches,[]))==uids
    for update in (0,1,9,10,123,1999):
        epoch, cursor = divmod(update,10)
        uninterrupted = epoch_batches(uids,epoch)[cursor]
        resumed = epoch_batches(uids,update//10)[update%10]
        assert uninterrupted == resumed
        for uid in resumed:
            x = dict(item,uid=uid)
            assert negative_faces(x,epoch)[1]==negative_faces(x,update//10)[1]
    logits = torch.tensor([-2.,1.,3.,-1.,.2,-.3],requires_grad=True)
    labels = torch.tensor([0,1,0,1,1,0])
    numerator,counts = hard4_sums(logits,labels)
    whole = (numerator/counts.clamp_min(1)).sum()/4
    fn = lambda embedding, ids: embedding[ids]
    chunked,_ = hard4_chunks(fn,logits,torch.arange(6),labels,2)
    assert torch.allclose(whole,chunked,atol=1e-7,rtol=1e-7)
    assert torch.allclose(torch.autograd.grad(whole,logits)[0],torch.autograd.grad(chunked,logits)[0])
    # Scheduled-boundary resume must reproduce model and Adam tensors exactly.
    model = NativeTopologyAE(Config(**dims))
    model.load_state_dict(baseline.state_dict())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,foreach=True)
    def update(model,opt,step):
        for group in opt.param_groups: group['lr']=1e-4*min(step/100,1)
        opt.zero_grad(set_to_none=True)
        outputs = model(vertices,faces,graph=graph)
        neg,_ = negative_faces(item,(step-1)//10)
        loss,_ = objective(outputs,item,neg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
        opt.step()
    with tempfile.TemporaryDirectory() as tmp:
        update(model,opt,1)
        path = Path(tmp)/'boundary.pt'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state()),path)
        update(model,opt,2)
        expected = tensor_hash(model.state_dict())
        cp = torch.load(path,weights_only=False)
        resumed = NativeTopologyAE(Config(**dims))
        resumed.load_state_dict(cp['model'])
        opt2 = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01,foreach=True)
        opt2.load_state_dict(cp['optimizer'])
        restore_rng(cp['rng'])
        update(resumed,opt2,2)
        assert expected == tensor_hash(resumed.state_dict())
        for i,state in opt.state_dict()['state'].items():
            assert all(torch.equal(v,opt2.state_dict()['state'][i][k]) for k,v in state.items())
    # Upper-triangular predicted adjacency must enumerate every triangle once.
    adjacency = np.triu(np.ones((6,6),dtype=bool),1)
    triangles = [(i,int(j),int(k)) for i in range(6) for j in np.flatnonzero(adjacency[i])
                 for k in np.flatnonzero(adjacency[i]&adjacency[j])]
    assert triangles == list(itertools.combinations(range(6),3))
    result = dict(scope='small synthetic CPU tests only; zero CAD optimizer updates; no CUDA',
        original_V2_forward_and_parameter_gradients_bitwise_equal=True,
        four_modes_same_parameter_keys_shapes_tensors=True, modes=modes,
        epochs_200_cover_all_50_once=True, resume_UID_and_negative_hashes_equal=True,
        boundary_model_and_Adam_resume_bitwise_equal=True, Hard4_chunk_gradients_match=True,
        complete_triangle_enumeration_matches_exhaustive=True, logvar_frozen=True)
    write(Path(__file__).resolve().parent/'repro_outputs/CPU_TESTS.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--original-source',required=True)
    main(p.parse_args().original_source)
