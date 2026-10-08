"""Compare unweighted small/large Soft4 gradients on one fixed step6000 forward."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT.parent/'soft4_two_mesh_resume_6000_20260910'
spec = importlib.util.spec_from_file_location('fixed_step6000_training_source', TRAIN/'run.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
probe, teacher = training.probe, training.teacher


def compare(xs, ys):
    small_sq = sum(float(x.double().square().sum()) for x in xs)
    large_sq = sum(float(y.double().square().sum()) for y in ys)
    dot = sum(float((x.double()*y.double()).sum()) for x,y in zip(xs,ys))
    delta_sq = sum(float((x.double()-y.double()).square().sum()) for x,y in zip(xs,ys))
    return dict(small_l2=small_sq**.5, large_l2=large_sq**.5, dot=dot,
                cosine=dot/(small_sq*large_sq)**.5 if small_sq and large_sq else None,
                small_over_large_l2=(small_sq/large_sq)**.5 if large_sq else None,
                difference_over_first_l2=(delta_sq/small_sq)**.5 if small_sq else None)


def main():
    assert not (ROOT/'complete.json').exists()
    started = time.monotonic()
    checkpoint = TRAIN/'continue/checkpoint-6000.pt'
    checksum = probe.digest(checkpoint)
    probe.UIDS = ['nexus_2k_000387','nexus_2k_001849']
    cp, model, batch = probe.setup_model(checkpoint)
    meta = cp['diagnostic_run']
    assert meta['completed_steps'] == 6000 and meta['objective'] == 'soft4'
    assert meta['script_sha256'] == probe.digest(TRAIN/'run.py')
    assert meta['loss_script_sha256'] == probe.digest(training.SOFT_ROOT/'run.py')
    for path,sha in meta['source_sha256'].items():
        assert probe.digest(path) == sha
    model.eval(); model.requires_grad_(True)
    a = model.autoencoder
    a.log_variance.requires_grad_(False); a.face_embedding.requires_grad_(False)
    assert {id(p) for p in model.parameters()} == {id(p) for p in a.parameters()}
    named = [(n,p) for n,p in a.named_parameters() if p.requires_grad]
    params = [p for n,p in named]
    for n,p in model.named_parameters():
        assert torch.equal(p.cpu(),cp['model'][n]),n
    initial = {n:p.detach().clone() for n,p in model.named_parameters()}
    buffers = {n:b.detach().clone() for n,b in model.named_buffers()}
    prefix = ('vertex_input.','face_input.','encoder_blocks.','encoder_output_norm.','mu.')
    groups = dict(encoder=[i for i,(n,p) in enumerate(named) if n.startswith(prefix)],
        decoder_body=[i for i,(n,p) in enumerate(named) if not n.startswith(prefix) and not n.startswith('edge_embedding.')],
        edge_head=[i for i,(n,p) in enumerate(named) if n.startswith('edge_embedding.')])
    assert sorted(sum(groups.values(),[])) == list(range(len(named)))
    assert all(groups.values())
    groups['all'] = list(range(len(named)))
    cpu_rng,gpu_rng = torch.get_rng_state(),torch.cuda.get_rng_state()
    rows = probe.get_rows(model,batch,'mu')
    assert [len(z) for z in rows[2]] == [386,2575]
    losses, metrics = [],[]
    prep = json.loads((training.OLD/'preparation.json').read_text())
    scale = model.scoring_contract()['edge_logit_scale']
    for i,uid in enumerate(probe.UIDS):
        pool = teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')
        assert probe.digest(pool) == prep['previous_configuration']['pool_sha256'][uid]
        data = np.load(pool)
        assert np.array_equal(data['vertices'],batch.vertices[i,:len(data['vertices'])].cpu().numpy())
        assert np.array_equal(data['positive'],batch.face_set[i].cpu().numpy())
        keys = teacher._canonical_positive_edge_keys(batch.edge_index[i],len(rows[2][i]),'cuda')
        loss,soft_groups = training.soft4_loss(rows[2][i],keys,cp['args']['pair_chunk_size'],scale)
        with torch.no_grad():
            _,counts = teacher.paper_edge_loss_all_pairs(rows[2][i],batch.edge_index[i],
                pair_chunk_size=cp['args']['pair_chunk_size'],positive_keys=keys,counts_on_device=True,logit_scale=scale)
        counts = {k:int(v) for k,v in counts.items()}
        metrics.append(dict(uid=uid,soft4_loss=float(loss.detach()),soft4_groups=soft_groups,**counts,
            edge_f1=2*counts['tp']/(2*counts['tp']+counts['fp']+counts['fn'])))
        losses.append(loss)
        np.savez_compressed(ROOT/(uid+'_forward.npz'),mu=rows[0][i].detach().cpu().numpy(),edge=rows[2][i].detach().cpu().numpy())

    def gradient(loss):
        values = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        result = [v.detach() if v is not None else torch.zeros_like(p) for v,p in zip(values,params)]
        assert all(torch.isfinite(v).all() for v in result)
        return result

    # No batch factor1/2, gradient clipping, optimizer, or second forward.
    gs,gl = [gradient(loss) for loss in losses]
    comparisons = {g:compare([gs[i] for i in ix],[gl[i] for i in ix]) for g,ix in groups.items()}
    repeats = {}
    for name,loss,first in [('small',losses[0],gs),('large',losses[1],gl)]:
        repeat = gradient(loss)
        repeats[name] = {g:compare([first[i] for i in ix],[repeat[i] for i in ix]) for g,ix in groups.items()}
    torch.save(dict(parameter_names=[n for n,p in named],
        small={n:gs[i].cpu() for i,(n,p) in enumerate(named)},
        large={n:gl[i].cpu() for i,(n,p) in enumerate(named)},
        group_parameter_names={g:[named[i][0] for i in ix] for g,ix in groups.items()},
        checkpoint_sha256=checksum),ROOT/'per_mesh_parameter_gradients.pt')
    assert all(torch.equal(p,initial[n]) for n,p in model.named_parameters())
    assert all(torch.equal(b,buffers[n]) for n,b in model.named_buffers())
    assert all(p.grad is None for p in params)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
    assert probe.digest(checkpoint) == checksum
    for path,sha in meta['source_sha256'].items():
        assert probe.digest(path) == sha
    result = dict(checkpoint=str(checkpoint),checkpoint_sha256=checksum,metrics=metrics,
        saved_checkpoint_metrics=cp['diagnostic_metrics']['rows'],comparison=comparisons,backward_repeat=repeats,
        parameter_counts={g:sum(params[i].numel() for i in ix) for g,ix in groups.items()},
        provenance=dict(forward_count=1,optimizer_steps=0,parameters_and_buffers_unchanged=True,rng_unchanged=True,
            gradient_definition='unweighted per-mesh Soft4, tau1, detached membership, FP32 group reduction; raw unclipped gradients',
            ratio_definition='norm(g_small)/norm(g_large)',dot_and_norm_reduction='FP64',
            groups='Encoder, Decoder body, edge head are disjoint; all is their union. Frozen logvar/face head have zero edge-loss gradients.',
            backend='unchanged FP32/BF16 Flash',script_sha256=probe.digest(Path(__file__)),source_sha256=meta['source_sha256']),
        seconds=time.monotonic()-started)
    probe.write(ROOT/'complete.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1);torch.manual_seed(20260910);torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    main()
