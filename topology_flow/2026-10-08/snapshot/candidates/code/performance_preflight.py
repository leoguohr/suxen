"""Bounded, optimizer-free FP32/MATH recomputation calibration on the fixed50 cache."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from _faces_reference import atomic_json
from _vertex_reference import farthest_point_sample
from topology_flow import configure_math_backend, linear_flow_target, equal_mesh_velocity_loss
from flow_training import build_training_state, batch_cursor
from latent_data import LatentCache, transform_latent
from vae_codec import sample_posterior
from gpu_budget import GPUBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cache', 'output', 'budget-file'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    cache = LatentCache(args.cache)
    configure_math_backend()
    report = dict(complete=False, optimizer_updates=0, isolated_probe_model=True,
        variant=config['model']['variant'], cache_sha256=cache.sha256,
        note='Isolated probe timing after one-mesh warmup is a selection heuristic, not a steady-state training speed claim. No optimizer steps occur. Production first-three update profiles are authoritative.',
        measurements=[])
    with GPUBudget(args.budget_file, 'performance_preflight', 'cuda:0') as budget:
        try:
            model, optimizer, rngs = build_training_state(config, torch.device('cuda:0'))
            # Exercise all branches; zero-initialized output/gates would make gradient comparisons vacuous.
            with torch.no_grad():
                model.flow.output.weight.normal_(std=.002)
                model.flow.position_embedding.weight.normal_(std=.002)
                for block in model.flow.blocks:
                    block.modulation[-1].weight.normal_(std=.001)
                    block.cross_attention.output.weight.normal_(std=.002)
                for p in model.parameters():
                    optimizer.state[p] = dict(step=torch.tensor(0.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
            group = batch_cursor(cache.uids, 0, 5, config['seed'])['next_uids']
            largest = max(cache.uids, key=lambda u: cache.records[u]['vertices'])
            report.update(probe_update_uids=group, largest_uid=largest,
                largest_vertices=cache.records[largest]['vertices'], full_adam_moments_allocated=True)
            prepared = {}
            for uid in dict.fromkeys(group+[largest]):
                item = cache.get(uid)
                clean = transform_latent(sample_posterior(item['mu'], item['logvar'], rngs['posterior']), cache.stats)[None]
                noise = torch.randn(clean.shape, generator=rngs['flow_noise'])
                t = torch.rand(1, generator=rngs['time'])
                mask = torch.ones(clean.shape[:2], dtype=torch.bool)
                mixed, target = linear_flow_target(clean, noise, t, mask)
                fps = farthest_point_sample(item['points'][None, :, :3], 1024, item['point_mask'][None])
                prepared[uid] = tuple(v.cuda() for v in (mixed, t, item['vertices'][None], item['points'][None], mask, item['point_mask'][None], fps))+(target.cuda(),)
            layers = config['model']['num_layers']; cond_layers = config['model']['condition_layers']
            policies = [dict(flow_direct_blocks=[], condition_direct_blocks=[])]
            for count in (8, 16, 24, 30, layers):
                policies.append(dict(flow_direct_blocks=list(range(layers-count, layers)),
                    condition_direct_blocks=list(range(cond_layers))))
            total = torch.cuda.get_device_properties(0).total_memory
            limit = total - (6 << 30)
            report.update(memory_limit_bytes=limit, safety_margin_bytes=6 << 30)
            baseline_grads = None; baseline_predictions = None

            def backward(uids):
                predictions = []
                for uid in uids:
                    if budget.stop(60): raise TimeoutError('Preflight time reserve reached')
                    sample = prepared[uid]
                    prediction = model(*sample[:-1])
                    loss = equal_mesh_velocity_loss(prediction, sample[-1], sample[4])/len(uids)
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite probe loss')
                    loss.backward()
                    predictions.append(prediction.detach().cpu())
                return predictions

            for policy in policies:
                if budget.stop(90): raise TimeoutError('Preflight budget exhausted before policy selection')
                model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                model.set_execution_policy(**policy)
                row = dict(policy=policy, status='running')
                report['measurements'].append(row)
                atomic_json(out/'report.json', report)
                try:
                    backward(group[:1]); model.zero_grad(set_to_none=True)
                    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); tick = time.monotonic()
                    predictions = backward(group)
                    torch.cuda.synchronize(); row['five_mesh_forward_backward_seconds'] = time.monotonic()-tick
                    if baseline_grads is None:
                        baseline_grads = {name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()}
                        baseline_predictions = predictions
                        row['equivalence'] = 'reference'
                    else:
                        exact = True; worst = 0.
                        for a,b in zip(predictions, baseline_predictions):
                            torch.testing.assert_close(a,b,atol=0,rtol=0)
                        for name,p in model.named_parameters():
                            if budget.stop(60): raise TimeoutError('Preflight comparison reserve reached')
                            actual = p.grad.detach().cpu(); expected = baseline_grads[name]
                            exact = exact and torch.equal(actual, expected)
                            worst = max(worst, float((actual-expected).abs().max()))
                            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
                        row.update(equivalence='bitwise' if exact else 'within_declared_2e-6_tolerance', gradient_max_abs=worst)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'], error_if_nonfinite=True)
                    row['gradient_norm_preclip'] = float(norm)
                    if budget.stop(60): raise TimeoutError('Preflight largest-mesh reserve reached')
                    model.zero_grad(set_to_none=True)
                    tick = time.monotonic(); backward([largest])
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'], error_if_nonfinite=True)
                    torch.cuda.synchronize()
                    row.update(largest_forward_backward_seconds=time.monotonic()-tick,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        status='passed' if torch.cuda.max_memory_reserved()<=limit else 'memory_margin_rejected')
                except torch.cuda.OutOfMemoryError as error:
                    row.update(status='OOM_rejected', error=str(error), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                    # Increasing direct activation retention further cannot reduce the required peak.
                    atomic_json(out/'report.json', report)
                    break
                atomic_json(out/'report.json', report)
            if budget.stop(15): raise TimeoutError('Preflight selection reserve reached')
            valid = [r for r in report['measurements'] if r['status']=='passed']
            if not valid: raise RuntimeError('No execution policy passed numerical and memory gates')
            best = min(valid, key=lambda r:r['five_mesh_forward_backward_seconds'])
            report.update(complete=True, selected_policy=best['policy'], selected_measurement=best,
                speedup_vs_reference=report['measurements'][0]['five_mesh_forward_backward_seconds']/best['five_mesh_forward_backward_seconds'],
                gpu_budget=budget.snapshot(), adam_steps=sorted({int(s['step']) for s in optimizer.state.values()}))
            assert report['adam_steps']==[0]
            atomic_json(out/'execution_policy.json', best['policy'])
            atomic_json(out/'report.json', report)
            print(json.dumps({k:report[k] for k in ('complete','selected_policy','speedup_vs_reference','selected_measurement')},indent=2),flush=True)
        except BaseException as error:
            report.update(failure=repr(error), gpu_budget=budget.snapshot())
            atomic_json(out/'report.json', report)
            raise


if __name__ == '__main__': main()
