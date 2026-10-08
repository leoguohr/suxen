"""Continue verified B2/Adam/RNG across nine depths; evaluate local and generated parents."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch


def depth_for(update, micro):
    return 1 + ((update - 1) * 8 + micro) % 9


def gt_at_depth(vertices, depth):
    return torch.unique(vertices // (2 ** (9 - depth)), dim=0)


def occupancy_for(parents, gt):
    lookup = set(map(tuple, gt.cpu().tolist()))
    bits = torch.tensor([[i >> 2 & 1, i >> 1 & 1, i & 1] for i in range(8)], device=parents.device)
    children = parents[:, None] * 2 + bits[None]
    values = [tuple(cell) in lookup for cell in children.reshape(-1, 3).cpu().tolist()]
    return torch.tensor(values, device=parents.device, dtype=torch.float32).reshape(1, len(parents), 8)


@torch.no_grad()
def sample_tree(model, context, sample, seed, folder, *, gt_parents=False):
    from mini_nexus.vertex_evaluation import sample_level
    from mini_nexus.octree import expand_occupied_children
    from scripts.train_vertex_staged import noise_for
    from scripts.evaluate_vertex_b2_frozen import occupancy_metrics
    folder.mkdir(parents=True, exist_ok=False)
    parents = torch.zeros((1, 3), dtype=torch.long, device=context.device)
    rows, first_mismatch = [], None
    for depth in range(1, 10):
        if gt_parents: parents = sample.octree_levels[depth - 1].parent_codes.to(context.device)
        gt = gt_at_depth(sample.quantized_vertices.to(context.device), depth)
        # Capacity abort is a failed trajectory, never truncated or repaired generation.
        if len(parents) > 4096:
            np.savez_compressed(folder / f'depth-{depth}.npz', parents=parents.cpu().numpy(), target_cells=gt.cpu().numpy())
            return {'seed': seed, 'gt_parents': gt_parents, 'exact_coordinate_set': False,
                    'first_mismatch_depth': first_mismatch or depth, 'capacity_abort_depth': depth, 'levels': rows}
        target = occupancy_for(parents, gt)
        noise = noise_for(target, seed + depth * 1000)
        value = sample_level(model, context, parents, depth, noise, steps=20) if len(parents) else noise
        predicted = expand_occupied_children(parents, value[0] >= .5)
        if len(parents):
            metrics = occupancy_metrics(value, target, predicted, gt)
        else:
            metrics = {'exact_coordinate_set': len(gt) == 0, 'predicted_count': 0, 'target_count': len(gt),
                       'missing_count': len(gt), 'extra_count': 0, 'occupancy_mse': None,
                       'minimum_signed_threshold_margin': None}
        if not metrics['exact_coordinate_set'] and first_mismatch is None: first_mismatch = depth
        np.savez_compressed(folder / f'depth-{depth}.npz', parents=parents.cpu().numpy(), noise=noise.cpu().numpy(),
                            estimate=value.cpu().numpy(), target_occupancy=target.cpu().numpy(),
                            predicted_cells=predicted.cpu().numpy(), target_cells=gt.cpu().numpy())
        rows.append({'depth': depth, 'input_parent_count': len(parents), 'noise_seed': seed + depth * 1000,
                     'noise_sha256': hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(), **metrics})
        parents = predicted
    return {'seed': seed, 'gt_parents': gt_parents, 'exact_coordinate_set': rows[-1]['exact_coordinate_set'],
            'all_levels_exact': first_mismatch is None, 'first_mismatch_depth': first_mismatch, 'levels': rows}


def backup_checkpoint(source, directory, update):
    import os
    from scripts.evaluate_vertex_b1_frozen import file_sha
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / 'checkpoint-last.pt.partial'
    expected = hashlib.sha256()
    with source.open('rb') as reader, temporary.open('wb') as writer:
        for block in iter(lambda: reader.read(16 * 1024**2), b''):
            writer.write(block); expected.update(block)
        writer.flush(); os.fsync(writer.fileno())
    actual = file_sha(temporary)
    assert actual == expected.hexdigest()
    target = directory / 'checkpoint-last.pt'; temporary.replace(target)
    report = {'phase': 'C', 'c_update': update, 'cumulative_step': 2000 + update, 'path': str(target),
              'sha256': actual, 'bytes': target.stat().st_size, 'verified_readback': True}
    temp = directory / 'backup_verified.json.partial'; temp.write_text(json.dumps(report, indent=2))
    temp.replace(directory / 'backup_verified.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code-root', 'checkpoint', 'manifest', 'output', 'frozen-report']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--backup-dir', type=Path)
    parser.add_argument('--updates', type=int, default=1800)
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    assert args.updates > 0 and args.eval_every > 0
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    from scripts.evaluate_vertex_b1_frozen import file_sha
    from scripts.evaluate_vertex_b2_frozen import base_config
    from scripts.train_vertex_b2 import verify_restored_optimizer
    from scripts.train_vertex_a100_b1 import make_model
    from scripts.train_vertex_staged import autocast
    from scripts.train_vertex_overfit import write_json, append_json
    from scripts.vertex_b1_evidence import projection_snapshot, projection_gradients, projection_updates
    from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
    torch.set_num_threads(8)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    gate = json.loads(args.frozen_report.read_text())
    assert gate['sampling_passed'] and gate['exact_count'] == 64
    assert gate['source_step'] == 2000 and gate['source_b2_update'] == 1000
    sha = file_sha(args.checkpoint); assert sha == gate['checkpoint_sha256']
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    source = state['config']; original = base_config(source)
    assert state['step'] == 2000 and state['b2_update'] == 1000 and source['phase'] == 'B2'
    assert source['uids'] == ['nexus_2k_000105'] and source['variant'] == 'R1'
    dataset = Nexus2KManifestDataset(args.manifest, 'train'); assert len(dataset) == 1
    sample = dataset[0]; assert sample.uid == 'nexus_2k_000105' and len(sample.quantized_vertices) == 8
    assert hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest() == original['condition_sha256'][sample.uid]
    level_audit = []
    for depth in range(1, 10):
        level = sample.octree_levels[depth - 1]
        gt = gt_at_depth(sample.quantized_vertices, depth)
        assert torch.equal(occupancy_for(level.parent_codes, gt)[0], level.target)
        level_audit.append({'depth': depth, 'parents': len(level.parent_codes), 'occupied_children': int(level.target.sum()),
                            'gt_cells': gt.tolist()})
    assert [r['parents'] for r in level_audit] == [1] + [8] * 8
    assert all(r['occupied_children'] == 8 for r in level_audit)
    write_json(args.output / 'level_audit.json', level_audit)
    model = make_model('R1', original['seed'], original['smoke_model'])
    model.load_state_dict(state['model'], strict=True); model.to(args.device)
    assert sum(p.numel() for p in model.parameters()) == source['parameter_count']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False)
    optimizer.load_state_dict(state['optimizer'])
    compared = verify_restored_optimizer(optimizer, state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if args.device == 'cuda': torch.cuda.set_rng_state_all(state['cuda_rng'])
    assert torch.equal(torch.get_rng_state(), state['torch_rng'])
    if args.device == 'cuda':
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(), state['cuda_rng']))
    args.precision = original['precision']
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(phase='C', variant='R1', uids=[sample.uid], source_step=2000, source_b2_update=1000,
                  source_checkpoint_sha256=sha, source_config=source, smoke_model=original['smoke_model'],
                  parameter_count=source['parameter_count'], learning_rate=1e-5, weight_decay=0., accumulation=8,
                  gradient_clip=1., new_warmup=False, depth_schedule='1+((update-1)*8+micro)%9',
                  time_sampling='independent U[0,1) per microbatch', threshold=.5, sampling_steps=20,
                  development_seeds=list(range(15_000_000, 15_000_004)),
                  final_seeds=list(range(16_000_000, 16_000_008)), max_parents_before_capacity_abort=4096)
    write_json(args.output / 'config.json', config)
    del state
    batch = collate_nexus2k_samples([sample]).to(args.device)
    @torch.no_grad()
    def evaluate(update, seeds, name):
        model.eval()
        cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all() if args.device == 'cuda' else []
        folder = args.output / name; folder.mkdir()
        local, full = [], []
        with autocast(args):
            context = model.condition_encoder(batch.condition)
            for seed in seeds:
                local.append(sample_tree(model, context, sample, seed, folder / f'gt-{seed}', gt_parents=True))
                full.append(sample_tree(model, context, sample, seed, folder / f'full-{seed}'))
        report = {'phase': 'C', 'update': update, 'cumulative_step': 2000 + update, 'uid': sample.uid,
                  'local_gt_parents': local, 'full_generated_parents': full,
                  'local_all_layers_passed': all(r['all_levels_exact'] for r in local),
                  'full_tree_passed': all(r['exact_coordinate_set'] for r in full),
                  'diagnostics_are_not_gates': True}
        assert torch.equal(cpu_rng, torch.get_rng_state())
        if args.device == 'cuda': assert all(torch.equal(a,b) for a,b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
        write_json(args.output / f'{name}.json', report)
        model.train()
        return report
    write_json(args.output / 'status.json', {'state': 'evaluating', 'update': 0})
    report = evaluate(0, config['development_seeds'], 'evaluation-000000')
    write_json(args.output / 'resume_verification.json', {'source_step': 2000, 'source_b2_update': 1000,
               'checkpoint_sha256': sha, 'model_loaded_strictly': True, 'optimizer_tensors_equal': compared,
               'rng_restored_and_preserved_through_initial_evaluation': True, 'first_update_lr': 1e-5, 'new_warmup': False})
    step = 0
    for step in range(1, args.updates + 1):
        if stopped: step -= 1; break
        assert sample.uid == 'nexus_2k_000105'
        started = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        times, depths, noises, losses = [], [], [], []
        for micro in range(8):
            depth = depth_for(step, micro); level = batch.octree_levels[depth - 1]
            t = torch.rand((1,), device=args.device); noise = torch.randn_like(level.target)
            with autocast(args): loss = model(batch.condition, level, noise=noise, time=t)
            assert torch.isfinite(loss)
            (loss / 8).backward()
            times.append(float(t)); depths.append(depth); losses.append(float(loss))
            noises.append(hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest())
        gradients = projection_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert torch.isfinite(norm)
        before = projection_snapshot(model); optimizer.step()
        changes = projection_updates(model, before); del before
        row = {'update': step, 'cumulative_step': 2000 + step, 'uid': sample.uid, 'depths': depths, 'times': times,
               'noise_sha256': noises, 'micro_losses': losses, 'loss': sum(losses)/8, 'lr': optimizer.param_groups[0]['lr'],
               'weight_decay': 0., 'accumulation': 8, 'gradient_norm_before_clip': float(norm),
               'projection_gradients_before_clip': gradients, 'projection_updates': changes,
               'seconds': time.monotonic()-started}
        append_json(args.output / 'train.jsonl', row); print(json.dumps(row), flush=True)
        write_json(args.output / 'status.json', {'state': 'training', 'phase': 'C', 'update': step})
        if step % args.eval_every == 0 or step == args.updates or stopped:
            optimizer.zero_grad(set_to_none=True)
            write_json(args.output / 'status.json', {'state': 'evaluating', 'update': step})
            report = evaluate(step, config['development_seeds'], f'evaluation-{step:06d}')
            write_json(args.output / 'status.json', {'state': 'saving', 'update': step})
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': config,
                        'step': 2000 + step, 'c_update': step, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else None},
                       args.output / 'checkpoint-last.pt.tmp')
            (args.output / 'checkpoint-last.pt.tmp').replace(args.output / 'checkpoint-last.pt')
            if args.backup_dir:
                write_json(args.output / 'status.json', {'state': 'backing_up', 'update': step})
                write_json(args.output / 'backup_verified.json', backup_checkpoint(args.output / 'checkpoint-last.pt', args.backup_dir, step))
    final = None
    if step == args.updates and not stopped:
        write_json(args.output / 'status.json', {'state': 'final_evaluation', 'update': step})
        final = evaluate(step, config['final_seeds'], 'fresh_once')
    result = {'phase': 'C', 'updates': step, 'stopped': stopped, 'D_started': False,
              'full_tree_passed': bool(final and final['full_tree_passed']),
              'local_all_layers_passed': bool(final and final['local_all_layers_passed'])}
    write_json(args.output / 'result.json', result)
    write_json(args.output / 'status.json', {'state': 'stopped' if stopped else 'complete', **result})


if __name__ == '__main__':
    main()
