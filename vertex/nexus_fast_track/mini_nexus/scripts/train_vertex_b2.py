"""Continue R1/Adam/RNG with independent random time and noise per microbatch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch


def persist_checkpoint(source, directory, update):
    """Publish a read-back-verified copy, keeping the previous backup until ready."""
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'checkpoint-last.pt.partial'
    expected = hashlib.sha256()
    with source.open('rb') as reader, temporary.open('wb') as writer:
        for block in iter(lambda: reader.read(16 * 1024**2), b''):
            writer.write(block)
            expected.update(block)
        writer.flush()
        os.fsync(writer.fileno())
    actual = hashlib.sha256()
    with temporary.open('rb') as reader:
        for block in iter(lambda: reader.read(16 * 1024**2), b''):
            actual.update(block)
    assert expected.hexdigest() == actual.hexdigest(), 'persistent backup hash mismatch'
    target = directory / 'checkpoint-last.pt'
    temporary.replace(target)
    report = {'b2_update': update, 'cumulative_step': 1000 + update, 'path': str(target),
              'bytes': target.stat().st_size, 'sha256': actual.hexdigest(), 'verified_readback': True}
    metadata = directory / 'backup_verified.json.partial'
    metadata.write_text(json.dumps(report, indent=2))
    metadata.replace(directory / 'backup_verified.json')
    return report


def time_response(delta_v, delta_x, time_value):
    gain = -1 / (1 - time_value)
    dv, dx = delta_v.double(), delta_x.double()
    return {'ideal_signed_gain': gain, 'signed_gain': float((dv * dx).sum() / dx.square().sum()),
            'relative_response_error': float((dv - gain * dx).norm() / (abs(gain) * dx.norm()))}


def verify_restored_optimizer(optimizer, saved):
    compared = 0
    for group, source in zip(optimizer.param_groups, saved['param_groups'], strict=True):
        assert group['lr'] == source['lr'] == 1e-5 and group['weight_decay'] == source['weight_decay'] == 0
        assert all(group[k] == v for k, v in source.items() if k != 'params')
        for parameter, ident in zip(group['params'], source['params'], strict=True):
            actual, expected = optimizer.state[parameter], saved['state'].get(ident, {})
            assert actual.keys() == expected.keys()
            for key, value in expected.items():
                if torch.is_tensor(value):
                    assert torch.equal(actual[key], value.to(actual[key].device)), key
                    compared += 1
                else:
                    assert actual[key] == value
    return compared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frozen-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--backup-dir', type=Path)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    assert args.updates > 0 and args.eval_every > 0
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
    from mini_nexus.octree import expand_occupied_children
    from mini_nexus.vertex_evaluation import cell_metrics, sample_level
    from scripts.check_vertex_sampler import exact_cells
    from scripts.train_vertex_a100_b1 import make_model
    from scripts.train_vertex_staged import autocast, noise_for, regression_probe
    from scripts.train_vertex_overfit import append_json, write_json
    from scripts.vertex_b1_evidence import projection_snapshot, projection_gradients, projection_updates
    from scripts.evaluate_vertex_b1_frozen import file_sha

    torch.set_num_threads(8)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    frozen = json.loads(args.frozen_report.read_text())
    assert frozen['geometry_regression_passed'], 'Independent B1 geometry gate is required'
    checkpoint_sha = file_sha(args.checkpoint)
    assert checkpoint_sha == frozen['checkpoint_sha256']
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    source = state['config']
    assert state['step'] == 1000 and source['variant'] == 'R1' and source['phase'] == 'B1'
    assert source['uids'] == ['nexus_2k_000105']
    manifest = args.manifest or Path(source['manifest'])
    data = Nexus2KManifestDataset(manifest, 'train')
    assert len(data) == 1
    sample = data[0]
    assert sample.uid == 'nexus_2k_000105' and len(sample.quantized_vertices) == 8
    assert hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest() == source['condition_sha256'][sample.uid]
    model = make_model('R1', source['seed'], source['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    optimizer_tensors = verify_restored_optimizer(optimizer, state['optimizer'])
    # Restore after every initialization/load operation. Validation uses separate generators.
    torch.set_rng_state(state['torch_rng'])
    if args.device == 'cuda':
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), state['cuda_rng']))
    assert torch.equal(torch.get_rng_state(), state['torch_rng'])
    args.precision = source['precision']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(phase='B2', variant='R1', uids=[sample.uid], source_checkpoint_sha256=checkpoint_sha,
                  source_step=1000, optimizer_restored_exactly=True, optimizer_tensors_compared=optimizer_tensors,
                  rng_restored_exactly=True, learning_rate=1e-5, new_warmup=False, accumulation=8,
                  time_sampling='independent U[0,1) per microbatch', smoke_model=source['smoke_model'],
                  source_config=source, parameter_count=sum(p.numel() for p in model.parameters()))
    write_json(args.output / 'config.json', config)
    del state
    batch = collate_nexus2k_samples([sample]).to(args.device)
    level = batch.octree_levels[8]
    parents = sample.octree_levels[8].parent_codes.to(args.device)
    gt = sample.quantized_vertices.to(args.device)

    @torch.no_grad()
    def sampling(context, folder, seeds):
        rows = []
        for seed in seeds:
            noise = noise_for(level.target, seed)
            for steps in [20, 40]:
                value = sample_level(model, context, parents, 9, noise.clone(), steps=steps)
                cells = expand_occupied_children(parents, value[0] >= .5)
                np.savez_compressed(folder / f'sample-{seed}-{steps}.npz', noise=noise.cpu().numpy(),
                                    estimate=value.cpu().numpy(), predicted_cells=cells.cpu().numpy(), target_cells=gt.cpu().numpy())
                rows.append({'seed': seed, 'steps': steps, 'uid': sample.uid, 'gt_parents': True,
                             'exact_coordinate_set': exact_cells(cells, gt), **cell_metrics(cells, gt)})
        return rows

    @torch.no_grad()
    def evaluate(update):
        model.eval()
        folder = args.output / f'arrays-{update:06d}'; folder.mkdir()
        by_time, responses = {}, []
        with autocast(args):
            context = model.condition_encoder(batch.condition)
            for t in [0., .1, .3, .5, .7, .9, .95]:
                probes = [regression_probe(model, sample, context, time_value=t, seed=seed,
                          save=folder / f'probe-{t}-{seed}.npz') for seed in range(11_000_000, 11_000_016)]
                by_time[str(t)] = {'probes': probes, 'mean_mse': sum(p['velocity_mse'] for p in probes) / 16,
                                  'worst_mse': max(p['velocity_mse'] for p in probes),
                                  'exact_count': sum(p['exact_coordinate_set'] for p in probes)}
            generated = sampling(context, folder, range(12_000_000, 12_000_008))
        context = model.condition_encoder(batch.condition)
        for t in [0., .1, .3, .5, .7, .9, .95]:
            for seed in range(11_000_000, 11_000_004):
                x = (1 - t) * noise_for(level.target, seed) + t * level.target
                # Scale perturbation with (1-t), so the ideal response does not explode near 1.
                dx = .1 * (1 - t) * noise_for(level.target, seed + 10_000_000)
                times = torch.tensor([t], device=args.device)
                before = model.flow(x, times, parents[None], torch.tensor([9], device=args.device), context)
                after = model.flow(x + dx, times, parents[None], torch.tensor([9], device=args.device), context)
                np.savez_compressed(folder / f'response-{t}-{seed}.npz', x=x.cpu().numpy(), delta_x=dx.cpu().numpy(),
                                    velocity_before=before.cpu().numpy(), velocity_after=after.cpu().numpy())
                responses.append({'time': t, 'seed': seed, **time_response(after - before, dx, t)})
        development_passed = all(p['velocity_mse'] <= .01 and p['exact_coordinate_set']
                                 for group in by_time.values() for p in group['probes'])
        development_passed &= all(r['exact_coordinate_set'] for r in generated if r['steps'] == 20)
        report = {'phase': 'B2', 'update': update, 'cumulative_step': 1000 + update,
                  'uid': sample.uid, 'by_time': by_time, 'sampling': generated,
                  'responses_fp32': responses, 'development_geometry_passed': development_passed,
                  'response_diagnostic_only': True, 'forty_step_is_diagnostic_only': True}
        write_json(args.output / f'evaluation-{update:06d}.json', report)
        model.train()
        return report

    write_json(args.output / 'status.json', {'state': 'evaluating', 'update': 0})
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if args.device == 'cuda' else []
    report = evaluate(0)
    assert torch.equal(cpu_rng, torch.get_rng_state()), 'validation changed training CPU RNG'
    if args.device == 'cuda':
        assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
    write_json(args.output / 'resume_verification.json', {
        'checkpoint_sha256': checkpoint_sha, 'model_loaded_strictly': True,
        'optimizer_tensors_equal': optimizer_tensors, 'optimizer_groups_equal': True,
        'rng_restored_and_preserved_through_initial_evaluation': True,
        'source_step': 1000, 'first_update_lr': optimizer.param_groups[0]['lr'], 'new_warmup': False})
    step = 0
    for step in range(1, args.updates + 1):
        if stopped:
            step -= 1
            break
        assert sample.uid == 'nexus_2k_000105'
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        times_used, noises, losses = [], [], []
        for micro in range(8):
            t = torch.rand((1,), device=args.device)
            noise = torch.randn_like(level.target)
            times_used.append(float(t));noises.append(hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest())
            with autocast(args):
                loss = model(batch.condition, level, noise=noise, time=t)
            assert torch.isfinite(loss), 'nonfinite loss'
            (loss / 8).backward();losses.append(float(loss))
        gradients = projection_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm), 'nonfinite gradient'
        before = projection_snapshot(model)
        optimizer.step()
        updates = projection_updates(model, before);del before
        if args.device == 'cuda':torch.cuda.synchronize()
        row = {'update': step, 'cumulative_step': 1000 + step, 'uid': sample.uid, 'times': times_used,
               'noise_sha256': noises, 'loss': sum(losses) / 8, 'lr': optimizer.param_groups[0]['lr'],
               'weight_decay': 0., 'accumulation': 8, 'gradient_norm_before_clip': float(norm),
               'projection_gradients_before_clip': gradients, 'projection_updates': updates,
               'seconds': time.monotonic() - started}
        append_json(args.output / 'train.jsonl', row);print(json.dumps(row), flush=True)
        write_json(args.output / 'status.json', {'state': 'training', 'phase': 'B2', 'update': step})
        if step % args.eval_every == 0 or step == args.updates or stopped:
            optimizer.zero_grad(set_to_none=True)
            write_json(args.output / 'status.json', {'state': 'evaluating', 'update': step})
            report = evaluate(step)
            write_json(args.output / 'status.json', {'state': 'saving', 'update': step})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                        'step': 1000 + step, 'b2_update': step, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else None},
                       args.output / 'checkpoint-last.pt.tmp')
            (args.output / 'checkpoint-last.pt.tmp').replace(args.output / 'checkpoint-last.pt')
            if args.backup_dir:
                write_json(args.output / 'status.json', {'state': 'backing_up', 'update': step})
                backup = persist_checkpoint(args.output / 'checkpoint-last.pt', args.backup_dir, step)
                write_json(args.output / 'backup_verified.json', backup)
    fresh = []
    if step == args.updates and not stopped:
        model.eval()
        folder = args.output / 'fresh_sampling_once';folder.mkdir()
        with torch.no_grad(), autocast(args):
            context = model.condition_encoder(batch.condition)
            fresh = sampling(context, folder, range(13_000_000, 13_000_008))
        write_json(args.output / 'fresh_sampling_once.json', {'samples': fresh, 'used_once': True})
    result = {'phase': 'B2', 'updates': step, 'stopped': stopped,
              'development_geometry_passed': report['development_geometry_passed'],
              'fresh_sampling_passed': bool(fresh) and all(r['exact_coordinate_set'] for r in fresh if r['steps'] == 20),
              'C_D_started': False}
    result['passed'] = result['development_geometry_passed'] and result['fresh_sampling_passed']
    write_json(args.output / 'result.json', result)
    write_json(args.output / 'status.json', {'state': 'stopped' if stopped else 'complete', **result})


if __name__ == '__main__':
    main()
