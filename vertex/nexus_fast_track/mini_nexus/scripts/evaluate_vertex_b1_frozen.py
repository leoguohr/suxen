"""Evaluate the fixed R1 step-1000 checkpoint once; never train or alter old results."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def separated_gates(probes, responses):
    geometry = len(probes) == 64 and all(p['exact_coordinate_set'] and p['velocity_mse'] <= .01 for p in probes)
    diagnostic = len(responses) == 64 and all(abs(r['signed_gain'] + 2) <= .2 and r['relative_response_error'] <= .1 for r in responses)
    return dict(geometry_regression_passed=geometry, local_response_diagnostic_passed=diagnostic,
                strict_passed_on_this_followup=geometry and diagnostic)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    from mini_nexus.data_2k import Nexus2KManifestDataset
    from scripts.train_vertex_a100_b1 import make_model, response_metrics, token_energy
    from scripts.train_vertex_staged import autocast, noise_for, regression_probe
    from scripts.train_vertex_overfit import write_json

    torch.set_num_threads(8)
    checkpoint_path = args.checkpoint or args.run_dir / 'checkpoint-last.pt'
    checkpoint_sha = file_sha(checkpoint_path)
    original_signature = (checkpoint_path.stat().st_size, checkpoint_path.stat().st_mtime_ns)
    state = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    config = state['config']
    assert state['step'] == 1000 and config['phase'] == 'B1' and config['variant'] == 'R1'
    assert config['uids'] == ['nexus_2k_000105']
    manifest = args.manifest or Path(config['manifest'])
    dataset = Nexus2KManifestDataset(manifest, 'train')
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.uid == 'nexus_2k_000105' and len(sample.quantized_vertices) == 8
    actual_condition_sha = hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest()
    assert actual_condition_sha == config['condition_sha256'][sample.uid]
    model = make_model('R1', config['seed'], config['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model = model.to(args.device).eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == config['parameter_count']
    versions = [p._version for p in model.parameters()]
    args.precision = config['precision']
    metadata = {'checkpoint_sha256': checkpoint_sha, 'checkpoint_path': str(checkpoint_path),
                'source_step': 1000, 'variant': 'R1', 'training_config': config,
                'manifest_sha256': file_sha(manifest), 'actual_condition_sha256': actual_condition_sha,
                'parameters_updated': False, 'old_strict_result_unchanged': True,
                'seeds': list(range(9_000_000, 9_000_064)), 'uid': sample.uid,
                'time': .5, 'depth': 9, 'gt_parents': True, 'smoke_model': config['smoke_model']}
    write_json(args.output / 'identity.json', metadata)
    write_json(args.output / 'status.json', {'state': 'evaluating', 'checked': 0})
    arrays = args.output / 'arrays'; arrays.mkdir()
    probes, responses, original_four = [], [], []
    with autocast(args):
        context = model.condition_encoder(sample.condition[None].to(args.device))
        for seed in metadata['seeds']:
            probes.append(regression_probe(model, sample, context, time_value=.5, seed=seed,
                                           save=arrays / f'geometry-{seed}.npz'))
    # Independently capture raw FP32 perturbation inputs and outputs, including the original four.
    context = model.condition_encoder(sample.condition[None].to(args.device))
    target = sample.octree_levels[8].target[None].to(args.device)
    parents = sample.octree_levels[8].parent_codes[None].to(args.device)
    times = torch.tensor([.5], device=args.device)
    depth = torch.tensor([9], device=args.device)
    for index, seed in enumerate(metadata['seeds'] + list(range(8_000_000, 8_000_004))):
        noise = noise_for(target, seed)
        x = .5 * noise + .5 * target
        dx = .05 * noise_for(target, seed + 10_000_000)
        layers, handles = {}, []
        if seed == 8_000_002:
            def capture(name):
                def hook(module, inputs, output):
                    layers[name] = token_energy(output)
                return hook
            for i, block in enumerate(model.flow.blocks):
                handles.append(block.register_forward_hook(capture(f'block_{i:02d}')))
        try:
            before = model.flow(x, times, parents, depth, context)
        finally:
            for handle in handles:
                handle.remove()
        after = model.flow(x + dx, times, parents, depth, context)
        np.savez_compressed(arrays / f'response-{seed}.npz',
                            x=x.cpu().numpy(), delta_x=dx.cpu().numpy(),
                            velocity_before=before.cpu().numpy(), velocity_after=after.cpu().numpy(),
                            target=target.cpu().numpy(), noise=noise.cpu().numpy(), parents=parents.cpu().numpy())
        row = {'seed': seed, 'time': .5, **response_metrics(after - before, dx), 'layers': layers}
        (responses if seed >= 9_000_000 else original_four).append(row)
        write_json(args.output / 'status.json', {'state': 'evaluating', 'responses_checked': index + 1})
    assert versions == [p._version for p in model.parameters()]
    assert original_signature == (checkpoint_path.stat().st_size, checkpoint_path.stat().st_mtime_ns)
    report = {**metadata, **separated_gates(probes, responses), 'probes': probes,
              'responses_fp32': responses, 'original_four_responses_fp32': original_four,
              'mean_velocity_mse': sum(p['velocity_mse'] for p in probes) / 64,
              'worst_velocity_mse': max(p['velocity_mse'] for p in probes),
              'exact_coordinate_count': sum(p['exact_coordinate_set'] for p in probes),
              'geometry_failures': [p['seed'] for p in probes if not p['exact_coordinate_set'] or p['velocity_mse'] > .01],
              'local_response_failures': [r['seed'] for r in responses if abs(r['signed_gain'] + 2) > .2 or r['relative_response_error'] > .1],
              'note': 'These 64 seeds are now consumed; never reuse them as an unseen terminal test.'}
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'status.json', {'state': 'complete', **separated_gates(probes, responses)})
    print(json.dumps({k: v for k, v in report.items() if k in ['geometry_regression_passed', 'local_response_diagnostic_passed',
                     'strict_passed_on_this_followup', 'mean_velocity_mse', 'worst_velocity_mse', 'exact_coordinate_count',
                     'geometry_failures', 'local_response_failures']}, indent=2))


if __name__ == '__main__':
    main()
