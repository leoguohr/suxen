#!/usr/bin/env python3
"""Gated A/B/C/D Vertex debugging. Oracle success is never a model score."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
from mini_nexus.flow import flow_matching_batch
from mini_nexus.octree import decode_leaf_centers, expand_occupied_children
from mini_nexus.training import VertexStageSystem
from mini_nexus.vertex_evaluation import binary_metrics, cell_metrics, generate_cells, sample_level, trace_denoising
from scripts.check_vertex_sampler import check_cells, exact_cells, reference_levels
from scripts.train_vertex_overfit import append_json, gradient_norm, learning_rate_at_step, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--uids', nargs='+', default=['nexus_2k_000105', 'nexus_2k_001208'])
    p.add_argument('--last-stage', choices=list('ABCD'), default='D')
    p.add_argument('--stage-steps', type=int, nargs=4, default=[1000, 2000, 3000, 3000])
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--smoke-model', action='store_true', help='Small model for CLI tests, never a full-model result.')
    args = p.parse_args()
    if args.eval_every < 1 or min(args.stage_steps) < 1 or args.learning_rate <= 0:
        p.error('positive step counts and learning rate required')
    if len(args.uids) != len(set(args.uids)) or not 1 <= len(args.uids) <= 4:
        p.error('require 1–4 distinct UIDs')
    if args.last_stage == 'D' and len(args.uids) < 2:
        p.error('D requires 2–4 distinct meshes')
    return args


def autocast(args):
    return torch.autocast(args.device, dtype=torch.bfloat16) if args.precision == 'bf16' else contextlib.nullcontext()


def noise_for(target, seed):
    return torch.randn(target.shape, device=target.device,
                       generator=torch.Generator(device=target.device).manual_seed(seed))


@torch.no_grad()
def regression_probe(model, sample, context, *, time_value, seed, save=None):
    level = sample.octree_levels[8]
    target = level.target.to(context.device)[None]
    parents = level.parent_codes.to(context.device)
    times = torch.tensor([time_value], device=context.device)
    noise = noise_for(target, seed)
    noisy, velocity, _, _ = flow_matching_batch(target, noise=noise, time=times)
    predicted = model.flow(noisy, times, parents[None], torch.tensor([9], device=context.device), context).float()
    mse = float((predicted - velocity).square().mean())
    baseline = float(velocity.square().mean())
    estimate = noisy + (1 - time_value) * predicted
    cells = expand_occupied_children(parents, estimate[0] >= .5)
    gt = sample.quantized_vertices.to(context.device)
    if save is not None:
        np.savez_compressed(save, noise=noise.cpu().numpy(), time=times.cpu().numpy(),
            parents=parents.cpu().numpy(), target_occupancy=target.cpu().numpy(),
            target_velocity=velocity.cpu().numpy(), predicted_velocity=predicted.cpu().numpy(),
            estimate=estimate.cpu().numpy(), predicted_cells=cells.cpu().numpy(), target_cells=gt.cpu().numpy())
    return {'uid': sample.uid, 'time': time_value, 'seed': seed, 'velocity_mse': mse,
            'zero_velocity_mse': baseline, 'relative_mse': mse / max(baseline, 1e-12),
            'exact_occupancy': bool(torch.equal(estimate >= .5, target.bool())),
            'exact_coordinate_set': exact_cells(cells, gt),
            'occupancy': binary_metrics(estimate >= .5, target.bool()), 'cells': cell_metrics(cells, gt)}


@torch.no_grad()
def evaluate_stage(args, model, samples, stage, step):
    model.eval()
    out = args.output / stage
    with autocast(args):
        contexts = [model.condition_encoder(s.condition[None].to(args.device)) for s in samples]
        probes, generations = [], []
        if stage in 'AB':
            for seed in ([args.seed + 100] if stage == 'A' else [700001, 700002, 700003, 700004]):
                for t in ([.5] if stage == 'A' else [.1, .3, .5, .7, .9]):
                    probes.append(regression_probe(model, samples[0], contexts[0], time_value=t, seed=seed,
                        save=out / f'probe-{step:06d}.npz' if stage == 'A' else None))
        if stage == 'B':
            sample, context = samples[0], contexts[0]
            level = sample.octree_levels[8]
            parents = level.parent_codes.to(args.device)
            for seed in [800001, 800002, 800003, 800004]:
                noise = noise_for(level.target.to(args.device)[None], seed)
                value = sample_level(model, context, parents, 9, noise, steps=20)
                cells = expand_occupied_children(parents, value[0] >= .5)
                gt = sample.quantized_vertices.to(args.device)
                np.savez_compressed(out / f'generated-{step:06d}-{seed}.npz',
                    predicted=cells.cpu().numpy(), target=gt.cpu().numpy(), continuous=value.float().cpu().numpy())
                generations.append({'uid': sample.uid, 'seed': seed, 'gt_parents': True,
                    'exact_coordinate_set': exact_cells(cells, gt), **cell_metrics(cells, gt)})
        if stage in 'CD':
            # Identical initial seed across objects explicitly tests switching conditions.
            for sample, context in zip(samples, contexts):
                for seed in [800001, 800002, 800003, 800004]:
                    cells, details = generate_cells(model, context, steps=20, seed=seed)
                    gt = sample.quantized_vertices.to(args.device)
                    np.savez_compressed(out / f'generated-{step:06d}-{sample.uid}-{seed}.npz',
                        predicted=np.empty((0, 3), dtype=np.int64) if cells is None else cells.cpu().numpy(),
                        target=gt.cpu().numpy(), status=details['status'])
                    generations.append({'uid': sample.uid, 'seed': seed, 'gt_parents': False,
                        'exact_coordinate_set': details['status'] == 'complete' and exact_cells(cells, gt),
                        **details, **({} if cells is None else cell_metrics(cells, gt))})
        trace = trace_denoising(model, samples[0].octree_levels[8], contexts[0], time=.5, seed=args.seed + 100)
    if stage == 'A':
        passed = all(p['velocity_mse'] <= 1e-4 and p['relative_mse'] <= 1e-3
                     and p['exact_occupancy'] and p['exact_coordinate_set'] for p in probes)
    elif stage == 'B':
        passed = all(p['velocity_mse'] <= .01 and p['exact_coordinate_set'] for p in probes)
        passed = passed and all(g['exact_coordinate_set'] for g in generations)
    else:
        passed = all(g['exact_coordinate_set'] for g in generations)
    report = {'stage': stage, 'step': step, 'gate_passed_this_evaluation': passed,
              'model_evaluation': True, 'oracle_only_not_model_score': False,
              'smoke_model': args.smoke_model, 'probes': probes, 'generations': generations, 'trace': trace}
    write_json(out / f'evaluation-{step:06d}.json', report)
    model.train()
    return report


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        torch.set_num_threads(8)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        dataset = Nexus2KManifestDataset(args.manifest, 'train')
        samples = [dataset[dataset.index_for_uid(uid)] for uid in args.uids]
        if not args.smoke_model and (samples[0].uid != 'nexus_2k_000105' or len(samples[0].vertices) != 8):
            raise ValueError('first experiment must use nexus_2k_000105 with 8 GT vertices')
        if args.last_stage == 'D':
            for i, s in enumerate(samples):
                for other in samples[:i]:
                    if exact_cells(s.quantized_vertices, other.quantized_vertices):
                        raise ValueError('D requires distinct GT coordinate sets')
        oracle_reports = []
        for sample in samples:
            # Compare loaded labels against an independent coordinate-to-bit implementation.
            for level, (parents, target) in zip(sample.octree_levels, reference_levels(sample.quantized_vertices)):
                assert torch.equal(level.parent_codes, parents) and torch.equal(level.target, target)
            torch.testing.assert_close(sample.vertices, decode_leaf_centers(sample.quantized_vertices, 9), rtol=0, atol=0)
            oracle_reports.append(check_cells(sample.quantized_vertices, name=sample.uid))
        write_json(args.output / 'oracle_before_training.json', {'passed': True, 'oracle_only_not_model_score': True, 'cases': oracle_reports})
        model_args = dict(hidden_dim=24, condition_dim=32, num_layers=2, num_heads=3,
                          condition_heads=4, condition_layers=2, condition_tokens=4) if args.smoke_model else {}
        model = VertexStageSystem(**model_args, use_checkpoint=True).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0., foreach=False)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(weight_decay=0., augmentation=False, fixed_point_cloud=True, fixed_t_A=.5,
            fixed_noise_seed_A=args.seed + 100, random_t_noise_from_stage='B',
            warmup_steps=100, gradient_clip=1., fresh_initialization=True,
            parameter_count=sum(p.numel() for p in model.parameters()),
            manifest_sha256=dataset.manifest_sha256, torch=torch.__version__,
            condition_sha256={s.uid: hashlib.sha256(s.condition.numpy().tobytes()).hexdigest() for s in samples},
            gates={'A': 'MSE<=1e-4, relative MSE<=1e-3, exact occupancy and coordinates; 3 consecutive checks',
                   'B': '20 held-out t/noise cases MSE<=.01 and exact coordinates; 4 pure-noise GT-parent samples exact',
                   'C': 'all 4 seeds produce exact depth-9 coordinate set using generated parents',
                   'D': 'all objects and all 4 shared seeds produce their own exact coordinate sets'},
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in ('mini_nexus', 'scripts') for p in sorted((ROOT / folder).glob('*.py'))})
        write_json(args.output / 'config.json', config)
        batches = [collate_nexus2k_samples([s]).to(args.device) for s in samples]
        fixed_level = batches[0].octree_levels[8]
        fixed_noise = noise_for(fixed_level.target, args.seed + 100)
        fixed_time = torch.tensor([.5], device=args.device)
        total_steps = 0
        stages = []
        for stage_index, stage in enumerate('ABCD'):
            out = args.output / stage
            out.mkdir()
            active_samples = samples if stage == 'D' else samples[:1]
            write_json(args.output / 'status.json', {'state': 'evaluating', 'stage': stage, 'step': 0})
            evaluate_stage(args, model, active_samples, stage, 0)
            consecutive = 0
            for step in range(1, args.stage_steps[stage_index] + 1):
                object_index = (step - 1) % len(active_samples)
                depth = 9 if stage in 'AB' else ((step - 1) // len(active_samples)) % 9 + 1
                batch = batches[object_index]
                level = batch.octree_levels[depth - 1]
                noise = fixed_noise if stage == 'A' else torch.randn_like(level.target)
                times = fixed_time if stage == 'A' else torch.rand(1, device=args.device)
                lr = learning_rate_at_step(total_steps, args.learning_rate, 100)
                for group in optimizer.param_groups: group['lr'] = lr
                write_json(args.output / 'status.json', {'state': 'training', 'stage': stage, 'step': step, 'total_steps': total_steps})
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                with autocast(args):
                    loss = model(batch.condition, level, noise=noise, time=times)
                if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                if not torch.isfinite(norm): raise FloatingPointError('non-finite gradient')
                record = {'stage': stage, 'step': step, 'total_steps': total_steps + 1,
                    'uid': batch.uids[0], 'depth': depth, 'time': float(times[0]),
                    'noise_sha256': hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest(),
                    'loss': float(loss), 'lr': lr, 'weight_decay': 0., 'gradient_norm_before_clip': float(norm)}
                if step <= 3 or step % args.eval_every == 0:
                    record.update(vecset_gradient_norm_after_clip=gradient_norm(model.condition_encoder),
                                  dit_gradient_norm_after_clip=gradient_norm(model.flow))
                optimizer.step()
                if args.device == 'cuda': torch.cuda.synchronize()
                record['seconds'] = time.monotonic() - started
                record['peak_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30 if args.device == 'cuda' else 0.
                append_json(out / 'train.jsonl', record); print(json.dumps(record), flush=True)
                total_steps += 1
                if step % args.eval_every == 0 or step == args.stage_steps[stage_index]:
                    optimizer.zero_grad(set_to_none=True)
                    report = evaluate_stage(args, model, active_samples, stage, step)
                    consecutive = consecutive + 1 if report['gate_passed_this_evaluation'] else 0
                    if consecutive >= (3 if stage == 'A' else 1): break
            passed = consecutive >= (3 if stage == 'A' else 1)
            result = {'stage': stage, 'passed': passed, 'steps': step, 'total_steps': total_steps,
                      'next_stage_allowed': passed, 'smoke_model': args.smoke_model}
            write_json(out / 'result.json', result)
            stages.append(result)
            # One resumable final state avoids keeping multiple 28 GB optimizer snapshots.
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': config, 'stage': stage, 'stage_step': step, 'total_steps': total_steps,
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else None}
            torch.save(checkpoint, args.output / 'checkpoint-last.pt.tmp')
            (args.output / 'checkpoint-last.pt.tmp').replace(args.output / 'checkpoint-last.pt')
            if not passed or stage == args.last_stage: break
        write_json(args.output / 'result.json', {'stages': stages, 'all_requested_stages_passed': all(s['passed'] for s in stages) and stages[-1]['stage'] == args.last_stage,
            'stopped_at': stages[-1]['stage'], 'smoke_model': args.smoke_model})
        write_json(args.output / 'status.json', {'state': 'complete' if stages[-1]['passed'] else 'gate_failed',
                                               'stage': stages[-1]['stage'], 'total_steps': total_steps})
    except Exception as error:
        write_json(args.output / 'status.json', {'state': 'failed', 'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
