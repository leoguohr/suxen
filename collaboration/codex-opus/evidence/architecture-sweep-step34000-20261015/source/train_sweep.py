"""Seven independent A24000 continuations; no implicit restart or protocol change."""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
AB_ROOT = ROOT.parent / 'nexus_d9_ab_20260930'
SOURCE_SHA = '2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba'
BRANCHES = tuple(f'S{i}' for i in range(7))
SOURCE_STEP, UPDATES, MICROS, COUNT, DEPTH = 24000, 10000, 8, 50, 9
EVENTS = UPDATES * MICROS
NAMESPACE = 'nexus_vertex_arch_sweep_20261006_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_dir(path.parent)


def runtime():
    sys.path.insert(0, str(AB_ROOT))
    import train_ab
    return train_ab


def code_hashes(ab):
    own = {path.name: sha(path) for path in sorted(ROOT.glob('*.py'))}
    return {**own, **{'ab/' + name: digest for name, digest in ab.code_hashes().items()},
            'ab/evaluate_ab.py': sha(AB_ROOT / 'evaluate_ab.py'),
            'ab/samplers.py': sha(AB_ROOT / 'samplers.py')}


def event_at(index):
    assert 0 <= index < EVENTS
    occurrence, offset = divmod(index, COUNT * DEPTH)
    mesh, depth0 = divmod(offset, DEPTH)
    return mesh, depth0, occurrence


def paired_noise(shape, uid, depth, occurrence, index, device):
    key = f'{NAMESPACE}|{index}|{uid}|{depth}|{occurrence}'
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    moment = torch.rand((1,), device=device, dtype=torch.float32, generator=generator)
    return noise, moment, key, hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()


def pack_levels(levels, noises):
    """Keep each mesh on its own batch axis, with explicit zero padding and masks."""
    assert levels and len(levels) == len(noises)
    batch, maximum = len(levels), max(level.target.shape[1] for level in levels)
    template = levels[0]
    target = template.target.new_zeros((batch, maximum, 8))
    metadata = template.metadata.new_zeros((batch, maximum, template.metadata.shape[-1]))
    positions = template.positions.new_zeros((batch, maximum, 3))
    mask = template.mask.new_zeros((batch, maximum))
    depth = torch.empty((batch,), dtype=torch.long, device=target.device)
    noise = target.new_zeros(target.shape)
    for row, (level, epsilon) in enumerate(zip(levels, noises)):
        count = level.target.shape[1]
        assert level.target.shape[0] == 1 and bool(level.mask.all()) and count > 0
        target[row, :count] = level.target[0]
        metadata[row, :count] = level.metadata[0]
        positions[row, :count] = level.positions[0]
        mask[row, :count] = level.mask[0]
        depth[row] = level.depths.reshape(-1)[0]
        noise[row, :count] = epsilon[0]
    return type(template)(target, metadata, positions, depth, mask), noise


def capture_rng(device):
    return {'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': [torch.cuda.get_rng_state(device)] if device.type == 'cuda' else []}


def restore_rng(source, device):
    random.setstate(source['python_rng'])
    np.random.set_state(source['numpy_rng'])
    torch.set_rng_state(source['torch_rng'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(source['cuda_rng'][0], device)
    assert_same(capture_rng(device), {key: source[key] for key in capture_rng(device)})


def assert_same(actual, expected):
    """Read back all model, Adam, RNG, cursor and exposure values, not just headers."""
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.equal(actual.cpu(), expected.detach().cpu())
    elif isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype and np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for value, previous in zip(actual, expected):
            assert_same(value, previous)
    else:
        assert actual == expected, (actual, expected)


def save_checkpoint(model, optimizer, provenance, config, counts, output, update, device,
                    initial_steps, restore_audit):
    path = output / f'checkpoint-{SOURCE_STEP + update:06d}.pt'
    assert not path.exists()
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': None,
               'step': SOURCE_STEP + update, 'sweep_update': update, 'config': config,
               'source_checkpoint_sha256': SOURCE_SHA, 'source_provenance': provenance,
               'optimizer_initial_steps': initial_steps, 'restore_audit': restore_audit,
               'sample_depth_counts': counts.copy(), 'schedule_next_event': update * MICROS,
               **capture_rng(device)}
    temporary = path.with_suffix('.pt.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_dir(path.parent)
    digest = sha(path)
    loaded = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    assert_same(loaded, payload)
    del loaded, payload
    gc.collect()
    identity = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
                'step': SOURCE_STEP + update, 'sweep_update': update,
                'schedule_next_event': update * MICROS, 'full_model_adam_rng': True,
                'atomic_write_fsynced': True, 'readback_verified': True,
                'readback_scope': 'all model/Adam/RNG/config/cursor/exposure values exact',
                'final': update == UPDATES and config['phase'] == 'NEXUS_VERTEX_ARCH_SWEEP_10000'}
    write_json(path.with_suffix('.json'), identity)
    write_json(output / 'checkpoint_identity.json', identity)
    for old in sorted(output.glob('checkpoint-*.pt'))[:-2]:
        old.unlink()
        old.with_suffix('.json').unlink()
    sync_dir(output)
    return identity


def train_update(model, optimizer, conditions, levels, manifest, update, device, microgroup=1,
                 indices=None):
    from architecture_variants import cached_condition_batch
    assert microgroup in (1, 2, 4, 8)
    indices = list(range((update - 1) * MICROS, update * MICROS)) if indices is None else indices
    assert len(indices) == MICROS
    optimizer.zero_grad(set_to_none=True)
    rows, mean_loss = [], 0.0
    for start in range(0, MICROS, microgroup):
        group_levels, noises, moments, clouds, group_rows = [], [], [], [], []
        for micro, index in enumerate(indices[start:start + microgroup], start):
            mesh, depth0, occurrence = event_at(index)
            level, uid = levels[mesh][depth0], manifest['train_uids'][mesh]
            noise, moment, key, digest = paired_noise(level.target.shape, uid, depth0 + 1,
                                                     occurrence, index, device)
            group_levels.append(level)
            noises.append(noise)
            moments.append(moment)
            clouds.append(conditions[mesh])
            group_rows.append({'event_index': index, 'micro': micro, 'mesh_index': mesh, 'uid': uid,
                               'depth': depth0 + 1, 'occurrence': occurrence, 'rng_key': key,
                               'noise_sha256': digest, 'time': float(moment.item()),
                               'parents': level.target.shape[1]})
        level, noise = pack_levels(group_levels, noises)
        cloud = (clouds[0] if microgroup == 1 else
                 cached_condition_batch(model, conditions, [row['mesh_index'] for row in group_rows]))
        with torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext():
            loss = model(cloud, level, noise=noise, time=torch.cat(moments))
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite velocity loss; optimizer not stepped')
        # Original forward averages each mesh first. The update always has eight equal events.
        scaled = loss * (len(group_rows) / MICROS)
        scaled.backward()
        value = float(loss.detach())
        mean_loss += value * len(group_rows) / MICROS
        rows.extend({**row, 'packed_group_mean_loss': value, 'packed_group_size': len(group_rows)}
                    for row in group_rows)
    for name, parameter in model.named_parameters():
        assert parameter.dtype == torch.float32 and parameter.requires_grad
        assert parameter.grad is not None, name
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return mean_loss, float(norm), rows


def load_start(args, device):
    ab = runtime()
    from architecture_variants import restore_source, cache_fixed_features
    assert args.start_sha == SOURCE_SHA and sha(args.start) == SOURCE_SHA
    source = torch.load(args.start, map_location='cpu', mmap=True, weights_only=False)
    assert source['step'] == SOURCE_STEP and source['d9_update'] == 2000
    assert source['schedule_next_event'] == 16000 and source['scheduler'] is None
    assert source['sample_depth_counts'].shape == (COUNT, DEPTH)
    assert int(source['sample_depth_counts'].sum()) == 16000
    original = source['config']
    assert original['phase'] == 'NEXUS_D9_AB_2000' and original['branch'] == 'A'
    assert original['max_depth'] == original['rope_reference_depth'] == DEPTH
    assert original['data_manifest_sha256'] == sha(args.data / 'manifest.json')
    for name, digest in original['code_sha256'].items():
        assert sha(AB_ROOT / name) == digest, name
    model, optimizer, audit = restore_source(source, args.branch, device)
    model.train().requires_grad_(True)
    model.flow.use_checkpoint = False
    model.condition_encoder.use_checkpoint = False
    manifest, conditions, levels = ab.load_data(args.data, device)
    cache = cache_fixed_features(model, conditions)
    restore_rng(source, device)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group['lr'] == 1e-5 and group['weight_decay'] == 0 and group['foreach'] is False
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8
    initial = {name: int(optimizer.state[param]['step'].item())
               for name, param in model.named_parameters()}
    assert set(initial.values()) <= {0, SOURCE_STEP} and len(initial) == len(optimizer.state)
    return ab, source, model, optimizer, audit, manifest, conditions, levels, cache, initial


def evaluate_model(model, conditions, manifest, data, output, branch, identity, config):
    """Reuse original full-tree Euler exporter, saving every prediction before loading GT."""
    import evaluate_ab as evaluation
    native_export = evaluation.native_export
    output.mkdir(parents=True, exist_ok=False)
    model.eval().requires_grad_(False)
    rows = []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for seed in evaluation.SEEDS:
            for index, record in enumerate(manifest['records']):
                uid = record['uid']
                # Keep one frozen learned context at a time; no training context is cached.
                context = model.condition_encoder(conditions[index])
                folder = output / 'native' / f'seed-{seed}' / uid
                metadata = {'branch': branch, 'solver': 'source Euler left endpoint',
                            'checkpoint_sha256': identity['sha256'], 'checkpoint_step': identity['step'],
                            'condition_source': record['condition_file'],
                            'condition_sha256': record['condition_sha256'],
                            'normalization': record.get('normalization'),
                            'code_sha256': config['code_sha256'], 'sample_generation_index': len(rows),
                            'uid_order': manifest['train_uids'], 'evaluation_gt_read': False,
                            'paired_seed_scope': 'same seed and parent set imply same noise; diverged parents change shape'}
                report = native_export.export_tree(model, context, folder, uid=uid, seed=seed,
                                                   steps=20, max_depth=DEPTH, metadata=metadata)
                rows.append({'seed': seed, 'uid': uid, 'status': report['status'],
                             'generation_file': str((folder / 'generation.json').relative_to(output)),
                             'prediction_sha256': report['prediction_sha256']})
                write_json(output / 'generation_progress.json', {'completed': len(rows), 'expected': 100})
    assert len(rows) == 100
    for row in rows:
        report = json.loads((output / row['generation_file']).read_text())
        folder = (output / row['generation_file']).parent
        assert sha(folder / report['prediction']) == report['prediction_sha256']
        for level in report['levels']:
            assert sha(folder / level['file']) == level['sha256']
            assert level['denoiser_calls'] == 20 or level['input_parent_count'] == 0
    write_json(output / 'generation_manifest.json', {'expected': 100, 'gt_read': False,
               'status': 'all_predictions_saved_and_hashed_before_gt_scoring', 'rows': rows})
    scored = []
    for row in rows:
        record = manifest['records'][manifest['train_uids'].index(row['uid'])]
        label, raw = data / record['label_file'], AB_ROOT / record['source_label_file']
        assert sha(label) == record['label_sha256'] and sha(raw) == record['source_label_sha256']
        with np.load(label, allow_pickle=False) as item:
            target = item['leaves'].copy()
        with np.load(raw, allow_pickle=False) as item:
            raw_gt = item['gt_normalized'].copy()
        scored.append(evaluation.score_tree((output / row['generation_file']).parent, target, raw_gt))
    report = evaluation.summarize(scored)
    write_json(output / 'evaluation.json', report)
    write_json(output / 'summary.json', {'branch': branch, 'checkpoint_identity': identity,
               'paired_control_seeds': list(evaluation.SEEDS), 'steps': 20, 'threshold': .5,
               'sampler': 'source Euler left endpoint', 'gt_parents_used': False,
               'scope': '50 seen meshes x 2 paired seeds; original root-to-D9 generation',
               **{key: value for key, value in report.items() if key != 'rows'}})


def train(args, preflight=False):
    assert torch.cuda.device_count() == 1, 'one UUID-isolated GPU per branch; no DDP'
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    assert not args.output.exists(), 'fresh branch only; no implicit restart'
    device = torch.device('cuda:0')
    loaded = load_start(args, device)
    ab, source, model, optimizer, audit, manifest, conditions, levels, cache, initial = loaded
    provenance = {'checkpoint_path': str(args.start), 'checkpoint_sha256': SOURCE_SHA,
                  'step': source['step'], 'd9_update': source['d9_update'], 'config': source['config'],
                  'schedule_next_event': source['schedule_next_event'],
                  'sample_depth_counts': source['sample_depth_counts']}
    config = {'phase': 'NEXUS_VERTEX_ARCH_SWEEP_SMOKE' if preflight else 'NEXUS_VERTEX_ARCH_SWEEP_10000',
              'branch': args.branch, 'source_step': SOURCE_STEP, 'source_checkpoint_sha256': SOURCE_SHA,
              'updates': UPDATES, 'micros_per_update': MICROS, 'microgroup': args.microgroup,
              'max_depth': DEPTH, 'rope_reference_depth': DEPTH, 'optimizer': 'Adam',
              'lr': 1e-5, 'weight_decay': 0, 'clip': 1.0, 'scheduler': None,
              'loss': 'original masked per-mesh velocity MSE; eight equal event weights per update',
              'precision': 'BF16 network; FP32 parameters/Adam/flow state', 'recompute': False,
              'schedule': 'UID-major depth1..9 within every 450 events; new branch cursor0',
              'source_cursor_provenance_only': 16000, 'sweep_start_cursor': 0,
              'rng_namespace': NAMESPACE, 'rng': 'private event-keyed CUDA generators; identical across branches',
              'data_manifest_sha256': sha(args.data / 'manifest.json'), 'code_sha256': code_hashes(ab),
              'learned_condition_features_cached': False, 'immutable_feature_cache': cache,
              'save_every': 200, 'startup_save_update': 1, 'retain_full_checkpoints': 2,
              'final_checkpoint': 'latest recovery retained permanently; no duplicate weights file',
              'evaluation_updates': [] if preflight else [UPDATES],
              'torch_version': torch.__version__, 'automatic_retry': False}
    args.output.mkdir(parents=True)
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'startup.json', {'restore_audit': audit, 'optimizer_initial_steps': initial,
               'strict_model': True, 'rng_readback': True, 'new_branch_cursor': 0,
               'source_cursor': 16000, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
               'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'gpu_name': torch.cuda.get_device_name(device)})
    del source, loaded
    gc.collect()
    counts = np.zeros((COUNT, DEPTH), dtype=np.int64)
    stopped, completed, identity = False, 0, None

    def stop(_number, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for update in range(1, (args.smoke_updates if preflight else UPDATES) + 1):
            indices = None
            if preflight and args.stress_largest:
                mesh, depth0 = max(((i, d) for i in range(COUNT) for d in range(DEPTH)),
                                  key=lambda pair: levels[pair[0]][pair[1]].target.shape[1])
                indices = [(update - 1) * MICROS * COUNT * DEPTH + micro * COUNT * DEPTH
                           + mesh * DEPTH + depth0 for micro in range(MICROS)]
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            loss, norm, rows = train_update(model, optimizer, conditions, levels, manifest,
                                            update, device, args.microgroup, indices)
            torch.cuda.synchronize(device)
            completed = update
            for row in rows:
                counts[row['mesh_index'], row['depth'] - 1] += 1
            row = {'sweep_update': update, 'step': SOURCE_STEP + update, 'loss': loss,
                   'gradient_norm_before_clip': norm, 'seconds': time.monotonic() - started, 'micros': rows,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 1024**3}
            with (args.output / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            write_json(args.output / 'status.json', {'state': 'training', 'sweep_update': update,
                       'training_complete': False, 'last_loss': loss, 'checkpoint': identity})
            if update == 1 or update % 200 == 0 or stopped or (preflight and update == args.smoke_updates):
                identity = save_checkpoint(model, optimizer, provenance, config, counts, args.output,
                                           update, device, initial, audit)
                write_json(args.output / 'status.json', {'state': 'training', 'sweep_update': update,
                           'training_complete': False, 'last_loss': loss, 'checkpoint': identity})
            print(json.dumps({'branch': args.branch, **{key: value for key, value in row.items()
                              if key != 'micros'}, 'checkpoint': identity if update == 1 else None}), flush=True)
            if stopped:
                write_json(args.output / 'status.json', {'state': 'stopped_by_signal', 'sweep_update': update,
                           'training_complete': False, 'checkpoint': identity, 'automatic_restart': False})
                return
        if preflight:
            write_json(args.output / 'status.json', {'state': 'complete_discarded', 'sweep_update': completed,
                       'smoke_only': True, 'formal_updates_counted': 0, 'checkpoint': identity})
            return
        assert int(counts.sum()) == EVENTS and int(counts.min()) == 177 and int(counts.max()) == 178
        for name, parameter in model.named_parameters():
            assert int(optimizer.state[parameter]['step'].item()) == initial[name] + UPDATES, name
        write_json(args.output / 'coverage.json', {'uids': manifest['train_uids'], 'counts': counts.tolist(),
                   'total_events': EVENTS, 'minimum': int(counts.min()), 'maximum': int(counts.max()),
                   'next_event': EVENTS, 'new_optimizer_updates': UPDATES, 'cumulative_step': SOURCE_STEP + UPDATES})
        write_json(args.output / 'status.json', {'state': 'evaluation', 'sweep_update': UPDATES,
                   'training_complete': True, 'evaluation_complete': False, 'checkpoint': identity})
        del optimizer, levels
        gc.collect()
        torch.cuda.empty_cache()
        evaluate_model(model, conditions, manifest, args.data, args.output / 'evaluation',
                       args.branch, identity, config)
        write_json(args.output / 'status.json', {'state': 'complete', 'sweep_update': UPDATES,
                   'training_complete': True, 'evaluation_complete': True, 'checkpoint': identity})
    except Exception as error:
        write_json(args.output / 'status.json', {'state': 'failed_no_automatic_retry',
                   'sweep_update': completed, 'training_complete': completed == UPDATES,
                   'checkpoint': identity, 'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('train', 'preflight'))
    parser.add_argument('--start', type=Path, default=AB_ROOT / 'experiment/A/checkpoint-024000.pt')
    parser.add_argument('--start-sha', default=SOURCE_SHA)
    parser.add_argument('--data', type=Path, default=AB_ROOT / 'prepared_d9')
    parser.add_argument('--branch', choices=BRANCHES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microgroup', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--smoke-updates', type=int, choices=(1, 2), default=1)
    parser.add_argument('--stress-largest', action='store_true')
    args = parser.parse_args()
    assert args.command == 'preflight' or (args.smoke_updates == 1 and not args.stress_largest)
    torch.set_num_threads(8)
    train(args, args.command == 'preflight')


if __name__ == '__main__':
    main()
