"""Evidence for the fixed-target, fixed-time B1 diagnostic; no model changes."""
import hashlib

import numpy as np
import torch

from mini_nexus.flow import flow_matching_batch
from mini_nexus.octree import expand_occupied_children
from mini_nexus.vertex_evaluation import binary_metrics, cell_metrics
from scripts.check_vertex_sampler import exact_cells


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def cache_training_input(path, batch, noise, times, uid, seed):
    level = batch.octree_levels[8]
    noisy, velocity, _, _ = flow_matching_batch(level.target, noise=noise, time=times)
    assert bool(level.mask.all()), 'single-mesh B1 must have no padded parents'
    arrays = dict(noise=noise, noisy=noisy, target_velocity=velocity,
                  target_occupancy=level.target, parents=level.positions.long(), time=times)
    np.savez_compressed(path, **{k: v.detach().cpu().numpy() for k, v in arrays.items()})
    return {'uid': uid, 'seed': seed, 'update': 1, 'microbatch': 0,
            'path': path.name, 'tensor_sha256': {k: tensor_sha(v) for k, v in arrays.items()}}


@torch.no_grad()
def cached_probe(model, sample, context, cache, save):
    with np.load(cache) as source:
        arrays = {k: torch.from_numpy(source[k]).to(context.device) for k in source.files}
    x, target_v = arrays['noisy'], arrays['target_velocity']
    target, parents, times = arrays['target_occupancy'], arrays['parents'], arrays['time']
    assert bool((times == .5).all())
    assert torch.equal(parents[0], sample.octree_levels[8].parent_codes.to(context.device))
    assert torch.equal(target[0], sample.octree_levels[8].target.to(context.device))
    prediction = model.flow(x, times, parents, torch.tensor([9], device=context.device), context).float()
    estimate = x + .5 * prediction
    cells = expand_occupied_children(parents[0], estimate[0] >= .5)
    gt = sample.quantized_vertices.to(context.device)
    mse, baseline = float((prediction - target_v).square().mean()), float(target_v.square().mean())
    np.savez_compressed(save, predicted_velocity=prediction.cpu().numpy(), estimate=estimate.cpu().numpy(),
                        predicted_cells=cells.cpu().numpy(), target_cells=gt.cpu().numpy())
    return {'uid': sample.uid, 'time': .5, 'kind': 'actual_training_cache', 'velocity_mse': mse,
            'zero_velocity_mse': baseline, 'relative_mse': mse / max(baseline, 1e-12),
            'exact_occupancy': bool(torch.equal(estimate >= .5, target.bool())),
            'exact_coordinate_set': exact_cells(cells, gt),
            'occupancy': binary_metrics(estimate >= .5, target.bool()), 'cells': cell_metrics(cells, gt),
            'noisy_sha256': tensor_sha(x), 'target_velocity_sha256': tensor_sha(target_v),
            'empty_baseline': empty_baseline(x, target_v, target)}


def empty_baseline(noisy, velocity, target):
    prediction = -2 * noisy
    estimate = noisy + .5 * prediction
    return {'velocity_mse': float((prediction - velocity).square().mean()),
            'theoretical_mse_4p': float(4 * target.float().mean()),
            'occupancy': binary_metrics(estimate >= .5, target.bool()),
            'generated_vertices': int((estimate >= .5).sum()),
            'purpose': 'all-empty control, not shape recovery'}


def projection_snapshot(model):
    return {name: p.detach().clone() for name, p in model.flow.named_parameters()
            if name.startswith(('data_embedding.', 'output.'))}


def projection_gradients(model):
    return {name: None if p.grad is None else float(p.grad.float().norm())
            for name, p in model.flow.named_parameters()
            if name.startswith(('data_embedding.', 'output.'))}


def projection_updates(model, before):
    return {name: {'parameter_norm_before': float(before[name].norm()),
                   'update_norm': float((p.detach() - before[name]).norm()),
                   'relative_update_norm': float((p.detach() - before[name]).norm() / before[name].norm().clamp_min(1e-30))}
            for name, p in model.flow.named_parameters() if name in before}
