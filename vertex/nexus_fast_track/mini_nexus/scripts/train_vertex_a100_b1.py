#!/usr/bin/env python3
"""100 fixed regressions, then independent fresh-init R0/R1 noise learning."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
from mini_nexus.flow import flow_matching_batch
from mini_nexus.training import VertexStageSystem
from scripts.check_vertex_sampler import check_cells, reference_levels
from scripts.train_vertex_staged import autocast, noise_for, regression_probe
from scripts.train_vertex_overfit import append_json, gradient_norm, learning_rate_at_step, write_json
from scripts.vertex_b1_evidence import (cache_training_input, cached_probe, empty_baseline,
    projection_snapshot, projection_gradients, projection_updates)


def fixed_seed(uid, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}:{uid}:A'.encode()).digest()[:8], 'little') % (2**63)


def make_model(variant, seed, smoke=False):
    torch.manual_seed(seed)
    kwargs = dict(hidden_dim=24, condition_dim=32, num_layers=2, num_heads=3,
                  condition_heads=4, condition_layers=2, condition_tokens=4) if smoke else {}
    model = VertexStageSystem(**kwargs, use_checkpoint=True)
    if variant == 'R1':
        flow = model.flow
        torch.nn.init.normal_(flow.depth_embedding.weight, std=.02)
        for layer in (flow.time_embedding[0], flow.time_embedding[2]):
            torch.nn.init.normal_(layer.weight, std=.02)
            torch.nn.init.zeros_(layer.bias)
        for block in flow.blocks:
            torch.nn.init.zeros_(block.cross_attention.output.weight)
            torch.nn.init.zeros_(block.cross_attention.output.bias)
    elif variant != 'R0':
        raise ValueError('unknown initialization')
    return model


def token_energy(value):
    value = value.detach().float()
    energy = value.square()
    denominator = energy.sum(-1).clamp_min(1e-30)
    top = energy.topk(min(8, energy.shape[-1]), dim=-1).values
    return {'rms': float(energy.mean().sqrt()),
            'token_rms': energy.mean(-1).sqrt().cpu().tolist(),
            'token_top1_energy_fraction': (top[..., 0] / denominator).cpu().tolist(),
            'token_top8_energy_fraction': (top.sum(-1) / denominator).cpu().tolist()}


def response_metrics(delta_velocity, delta_x):
    dv, dx = delta_velocity.double(), delta_x.double()
    energy = dx.square().sum()
    if energy == 0:
        raise ValueError('nonzero perturbation required')
    return {'signed_gain': float((dv * dx).sum() / energy),
            'magnitude_gain': float(dv.norm() / dx.norm()),
            'relative_response_error': float((dv + 2 * dx).norm() / (2 * dx.norm()))}


def response_passes(row):
    return abs(row['signed_gain'] + 2) <= .2 and row['relative_response_error'] <= .1


def assert_b1_scope(samples, smoke=False):
    assert len(samples) == 1, 'B1 must consume exactly one mesh'
    if not smoke:
        assert {s.uid for s in samples} == {'nexus_2k_000105'}, 'B1 UID mismatch'


def a_passes(rows):
    return bool(rows) and all(r['velocity_mse'] <= 1e-4 and r['relative_mse'] <= 1e-3
                             and r['exact_occupancy'] and r['exact_coordinate_set'] for r in rows)


@torch.no_grad()
def fp32_response(model, sample, context, seed, *, trace=False):
    target = sample.octree_levels[8].target[None].to(context.device)
    noise = noise_for(target, seed)
    value = .5 * noise + .5 * target
    delta = .05 * noise_for(target, seed + 10_000_000)
    parents = sample.octree_levels[8].parent_codes[None].to(context.device)
    depth = torch.tensor([9], device=context.device)
    times = torch.tensor([.5], device=context.device)
    layers, handles = {}, []
    def capture(name):
        def hook(module, inputs, output):
            layers[name] = token_energy(output)
        return hook
    try:
        if trace:
            modules = {f'block_{i:02}': b for i, b in enumerate(model.flow.blocks)}
            modules.update(data_embedding=model.flow.data_embedding, position_embedding=model.flow.position_embedding,
                           depth_embedding=model.flow.depth_embedding, output_norm=model.flow.output_norm)
            handles = [m.register_forward_hook(capture(n)) for n, m in modules.items()]
        before = model.flow(value, times, parents, depth, context)
    finally:
        for handle in handles:
            handle.remove()
    after = model.flow(value + delta, times, parents, depth, context)
    return {'seed': seed, 'time': .5, **response_metrics(after - before, delta), 'layers': layers}


@torch.no_grad()
def evaluate(args, model, samples, out, step, phase, *, heldout=False):
    model.eval()
    rows, responses, seen = [], [], None
    if phase == 'A':
        arrays = out / 'arrays_latest'; arrays.mkdir(exist_ok=True)
        for sample in samples:
            with autocast(args):
                context = model.condition_encoder(sample.condition[None].to(args.device))
                row = regression_probe(model, sample, context, time_value=.5,
                    seed=fixed_seed(sample.uid, args.seed), save=arrays / f'{sample.uid}.npz')
            rows.append(row)
        passed = len(rows) == len(samples) and a_passes(rows)
    else:
        assert_b1_scope(samples, args.smoke_model)
        sample = samples[0]
        # Validation and final holdout have disjoint seed namespaces from training.
        seeds = list(range(9_000_000, 9_000_064)) if heldout else list(range(8_000_000, 8_000_016))
        with autocast(args):
            context = model.condition_encoder(sample.condition[None].to(args.device))
            probe_dir = out / ('holdout_arrays' if heldout else f'probes-{step:06d}')
            probe_dir.mkdir(exist_ok=True)
            for seed in seeds:
                row = regression_probe(model, sample, context, time_value=.5, seed=seed,
                                       save=probe_dir / f'{seed}.npz')
                target = sample.octree_levels[8].target[None].to(args.device)
                x, v, _, _ = flow_matching_batch(target, noise=noise_for(target, seed),
                                               time=torch.tensor([.5], device=args.device))
                row['empty_baseline'] = empty_baseline(x, v, target)
                rows.append(row)
            if (out / 'training_cache.npz').exists():
                seen = cached_probe(model, sample, context, out / 'training_cache.npz',
                                    probe_dir / 'training_cache_prediction.npz')
        # All operations below really run in FP32, including the point encoder.
        context = model.condition_encoder(sample.condition[None].to(args.device))
        response_seeds = seeds if heldout else seeds[:4]
        responses = [fp32_response(model, sample, context, seed, trace=i == 0)
                     for i, seed in enumerate(response_seeds)]
        passed = all(r['velocity_mse'] <= .01 and r['exact_coordinate_set'] for r in rows)
        passed = passed and all(response_passes(r) for r in responses)
    report = {'phase': phase, 'step': step, 'passed': passed, 'heldout': heldout,
              'smoke_model': args.smoke_model, 'probes': rows, 'responses_fp32': responses,
              'exact_coordinate_count': sum(r['exact_coordinate_set'] for r in rows),
              'mean_velocity_mse': sum(r['velocity_mse'] for r in rows) / len(rows),
              'worst_velocity_mse': max(r['velocity_mse'] for r in rows)}
    if phase == 'B1':
        consumed = [json.loads(line)['uid'] for line in (out / 'train.jsonl').read_text().splitlines()] if step else []
        assert set(consumed) <= {samples[0].uid}
        assert {r['uid'] for r in rows} == {samples[0].uid}
        report.update(experiment='B1_single_mesh_fixed_t', training_cache_probe=seen,
            scope={'train_uids': [s.uid for s in samples], 'eval_uids': [s.uid for s in samples],
                   'actually_consumed_uids': sorted(set(consumed)), 'consumed_updates': len(consumed),
                   'depth': 9, 'gt_parents': True, 'time': .5, 'fixed_condition': True,
                   'training_noise': '8 fresh independent draws per update',
                   'validation_seeds': list(range(8_000_000, 8_000_016)),
                   'holdout_seeds': list(range(9_000_000, 9_000_064))},
            training_config=json.loads((out / 'config.json').read_text()),
            training_cache=json.loads((out / 'training_cache.json').read_text()) if seen else None)
    filename = 'heldout_once.json' if heldout else f'evaluation-{step:06d}.json'
    write_json(out / filename, report)
    model.train()
    return report


def checkpoint(out, model, optimizer, config, step, streak):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': config, 'step': step, 'consecutive_passes': streak,
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if config['device'] == 'cuda' else None},
               out / 'checkpoint-last.pt.tmp')
    (out / 'checkpoint-last.pt.tmp').replace(out / 'checkpoint-last.pt')


def run(args, samples, phase, variant, should_stop):
    if phase == 'B1':
        assert_b1_scope(samples, args.smoke_model)
    out = args.output / ('A100' if phase == 'A' else f'B1_{variant}')
    out.mkdir()
    model = make_model(variant, args.seed, args.smoke_model).to(args.device)
    lr_target, accumulation = (1e-4, 1) if phase == 'A' else (1e-5, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_target, weight_decay=0., foreach=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(phase=phase, variant=variant, fresh_initialization=True, loaded_checkpoint=None,
        parameter_count=sum(p.numel() for p in model.parameters()), learning_rate=lr_target,
        gradient_accumulation=accumulation, weight_decay=0., warmup=100, clip=1.,
        fixed_time=.5, fixed_point_cloud=True, augmentation=False,
        uids=[s.uid for s in samples], manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        condition_sha256={s.uid: hashlib.sha256(s.condition.numpy().tobytes()).hexdigest() for s in samples},
        torch=torch.__version__, final_holdout='64 disjoint noises, consumed once; stop arm regardless of outcome',
        initialization_changes=[] if variant == 'R0' else ['depth std=.02', 'time MLP std=.02', 'cross output zero'],
        response_gate='FP32 signed gain in [-2.2,-1.8] AND relative response error <= .1',
        A_gate='all meshes MSE<=1e-4, relative<=1e-3, exact occupancy and coordinates; 3 consecutive checks')
    write_json(out / 'config.json', config)
    limit = args.a_updates if phase == 'A' else args.b_updates
    interval = args.a_eval_every if phase == 'A' else 50
    write_json(args.output / 'status.json', {'state': 'evaluating', 'phase': phase, 'variant': variant, 'step': 0})
    initial = evaluate(args, model, samples, out, 0, phase)
    history = [initial]
    streak, passed, step, final_holdout = 0, False, 0, None
    loss = batch = None
    while step < limit and not should_stop():
        if phase == 'A':
            epoch, offset = divmod(step, len(samples))
            order = torch.randperm(len(samples), generator=torch.Generator().manual_seed(args.seed + epoch))
            index = int(order[offset])
        else:
            index = 0
        sample = samples[index]
        if phase == 'B1':
            assert_b1_scope([sample], args.smoke_model)
        batch = collate_nexus2k_samples([sample]).to(args.device)
        level = batch.octree_levels[8]
        times = torch.tensor([.5], device=args.device)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate_at_step(step, lr_target, 100)
        optimizer.zero_grad(set_to_none=True)
        losses, noise_hashes = [], []
        started = time.monotonic()
        for micro in range(accumulation):
            noise_seed = fixed_seed(sample.uid, args.seed) if phase == 'A' else args.seed + 100_000_000 + step * 8 + micro
            noise = noise_for(level.target, noise_seed)
            noise_hashes.append(hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest())
            if phase == 'B1' and step == 0 and micro == 0:
                cached = cache_training_input(out / 'training_cache.npz', batch, noise, times, sample.uid, noise_seed)
                write_json(out / 'training_cache.json', cached)
            with autocast(args):
                loss = model(batch.condition, level, noise=noise, time=times)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            (loss / accumulation).backward()
            losses.append(float(loss))
        if phase == 'B1':
            gradients_before_clip = projection_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('non-finite gradient')
        row = {'step': step + 1, 'uid': sample.uid, 'depth': 9, 'time': .5,
               'noise_sha256': noise_hashes, 'loss': sum(losses) / accumulation,
               'lr': optimizer.param_groups[0]['lr'], 'weight_decay': 0., 'accumulation': accumulation,
               'gradient_norm_before_clip': float(norm)}
        if step < 5 or (step + 1) % interval == 0:
            row['vecset_gradient_norm_after_clip'] = gradient_norm(model.condition_encoder)
            row['dit_gradient_norm_after_clip'] = gradient_norm(model.flow)
        if phase == 'B1':
            before = projection_snapshot(model)
            row['projection_gradients_before_clip'] = gradients_before_clip
            row['projection_gradients_after_clip'] = projection_gradients(model)
        optimizer.step()
        if phase == 'B1':
            row['projection_updates'] = projection_updates(model, before)
            del before
        if args.device == 'cuda':
            torch.cuda.synchronize()
        row['seconds'] = time.monotonic() - started
        row['peak_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30 if args.device == 'cuda' else 0.
        append_json(out / 'train.jsonl', row)
        print(json.dumps({'phase': phase, 'variant': variant, **row}), flush=True)
        step += 1
        write_json(args.output / 'status.json', {'state': 'training', 'phase': phase, 'variant': variant, 'step': step})
        if step % interval == 0 or step == limit or should_stop():
            optimizer.zero_grad(set_to_none=True)
            write_json(args.output / 'status.json', {'state': 'evaluating', 'phase': phase, 'variant': variant, 'step': step})
            report = evaluate(args, model, samples, out, step, phase)
            history.append(report)
            streak = streak + 1 if report['passed'] else 0
            passed = streak >= 3 if phase == 'A' else report['passed']
            if passed and phase == 'B1':
                final_holdout = evaluate(args, model, samples, out, step, phase, heldout=True)
                passed = final_holdout['passed']
            write_json(args.output / 'status.json', {'state': 'saving', 'phase': phase, 'variant': variant, 'step': step})
            checkpoint(out, model, optimizer, config, step, streak)
            if passed or final_holdout is not None:
                break
            if phase == 'B1' and step == args.b_updates and args.b_max_updates > limit:
                recent = [h for h in history if step - 100 <= h['step'] <= step]
                improving = len(recent) >= 3 and all(x['mean_velocity_mse'] > y['mean_velocity_mse'] for x, y in zip(recent, recent[1:]))
                improving = improving and recent[-1]['mean_velocity_mse'] <= .9 * recent[0]['mean_velocity_mse']
                write_json(out / 'extension_decision.json', {'extend': improving,
                    'rule': 'last 3 evaluations strictly improve, >=10% MSE reduction over 100 updates',
                    'from': limit, 'to': args.b_max_updates if improving else limit})
                if improving:
                    limit = args.b_max_updates
    result = {'phase': phase, 'variant': variant, 'passed': passed, 'steps': step,
              'smoke_model': args.smoke_model, 'stopped_by_signal': should_stop(),
              'heldout_evaluated': final_holdout is not None, 'last_evaluation': history[-1]['step']}
    write_json(out / 'result.json', result)
    del loss, model, optimizer, batch
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['A', 'B1', 'pipeline'], default='pipeline')
    p.add_argument('--variant', choices=['R0', 'R1', 'both'], default='both')
    p.add_argument('--expected-samples', type=int, default=100)
    p.add_argument('--a-updates', type=int, default=20000)
    p.add_argument('--a-eval-every', type=int, default=500)
    p.add_argument('--b-updates', type=int, default=500)
    p.add_argument('--b-max-updates', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260914)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='bf16')
    p.add_argument('--smoke-model', action='store_true')
    a = p.parse_args()
    if min(a.expected_samples, a.a_updates, a.a_eval_every, a.b_updates) < 1 or a.b_max_updates < a.b_updates:
        p.error('invalid budgets')
    return a


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    torch.set_num_threads(8)
    try:
        dataset = Nexus2KManifestDataset(args.manifest, 'train')
        if len(dataset) != args.expected_samples:
            raise ValueError('manifest sample count mismatch')
        samples = [dataset[i] for i in range(len(dataset))]
        if args.phase == 'B1':
            assert_b1_scope(samples, args.smoke_model)
        anchor = next((s for s in samples if s.uid == 'nexus_2k_000105'), samples[0] if args.smoke_model else None)
        if anchor is None or (not args.smoke_model and len(anchor.vertices) != 8):
            raise ValueError('B1 anchor 000105 with 8 vertices required')
        oracles = []
        for i, sample in enumerate(samples):
            for level, (parents, target) in zip(sample.octree_levels, reference_levels(sample.quantized_vertices)):
                assert torch.equal(level.parent_codes, parents) and torch.equal(level.target, target)
            oracles.append(check_cells(sample.quantized_vertices, name=sample.uid, seeds=(17,), steps_list=(20,)))
            write_json(args.output / 'status.json', {'state': 'oracle', 'checked': i + 1, 'total': len(samples)})
        write_json(args.output / 'oracle.json', {'passed': True, 'oracle_only_not_model_score': True, 'cases': oracles})
        results = []
        if args.phase in ('A', 'pipeline'):
            result = run(args, samples, 'A', 'R0', lambda: stopped)
            results.append(result)
            if not result['passed'] or stopped:
                write_json(args.output / 'result.json', {'stages': results, 'B1_started': False})
                write_json(args.output / 'status.json', {'state': 'stopped' if stopped else 'gate_failed', 'phase': 'A'})
                return
        if args.phase in ('B1', 'pipeline'):
            for variant in (['R0', 'R1'] if args.variant == 'both' else [args.variant]):
                if stopped:
                    break
                results.append(run(args, [anchor], 'B1', variant, lambda: stopped))
        write_json(args.output / 'result.json', {'stages': results, 'B1_started': any(r['phase'] == 'B1' for r in results),
                                               'B2_C_D_started': False})
        write_json(args.output / 'status.json', {'state': 'stopped' if stopped else 'complete', 'stages': results})
    except Exception as error:
        write_json(args.output / 'status.json', {'state': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
