"""Freeze B2 step 2000 and sample 64 new noises once; geometry is the only gate."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def base_config(config):
    while 'source_config' in config:
        config = config['source_config']
    return config


def occupancy_metrics(value, target, predicted, gt):
    pred_set, gt_set = set(map(tuple, predicted.cpu().tolist())), set(map(tuple, gt.cpu().tolist()))
    error = value.float() - target.float()
    return {'exact_coordinate_set': pred_set == gt_set, 'predicted_count': len(pred_set),
            'target_count': len(gt_set), 'missing_count': len(gt_set - pred_set),
            'extra_count': len(pred_set - gt_set), 'occupancy_mse': float(error.square().mean()),
            'occupancy_mae': float(error.abs().mean()), 'occupancy_max_abs_error': float(error.abs().max()),
            'minimum_signed_threshold_margin': float(((2 * target - 1) * (value - .5)).min()),
            'minimum_absolute_threshold_distance': float((value - .5).abs().min())}


def sampling_gate(rows):
    return len(rows) == 64 and all(row['exact_coordinate_set'] for row in rows)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code-root', 'checkpoint', 'manifest', 'output', 'paired-noise']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    from scripts.evaluate_vertex_b1_frozen import file_sha
    from scripts.train_vertex_a100_b1 import make_model
    from scripts.train_vertex_staged import noise_for
    from scripts.train_vertex_overfit import write_json, append_json
    from mini_nexus.data_2k import Nexus2KManifestDataset
    from mini_nexus.vertex_evaluation import sample_level
    from mini_nexus.octree import expand_occupied_children
    torch.set_num_threads(8)
    sha = file_sha(args.checkpoint)
    assert sha == args.expected_sha
    signature = (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = state['config']; source = base_config(config)
    assert state['step'] == 2000 and state['b2_update'] == 1000 and config['phase'] == 'B2'
    assert config['variant'] == 'R1' and config['uids'] == ['nexus_2k_000105']
    data = Nexus2KManifestDataset(args.manifest, 'train'); assert len(data) == 1
    sample = data[0]; assert sample.uid == 'nexus_2k_000105' and len(sample.quantized_vertices) == 8
    condition_sha = hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest()
    assert condition_sha == source['condition_sha256'][sample.uid]
    model = make_model('R1', source['seed'], source['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(args.device).eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == config['parameter_count']
    del state
    versions = [p._version for p in model.parameters()]
    precision = source['precision']
    def amp(mode):
        return torch.autocast(args.device, dtype=torch.bfloat16) if mode == 'bf16' else nullcontext()
    contexts = {}
    for mode in dict.fromkeys([precision, 'bf16', 'fp32']):
        with amp(mode): contexts[mode] = model.condition_encoder(sample.condition[None].to(args.device))
    target = sample.octree_levels[8].target[None].to(args.device)
    parents = sample.octree_levels[8].parent_codes.to(args.device)
    gt = sample.quantized_vertices.to(args.device)
    metadata = {'phase': 'B2_frozen64', 'checkpoint_sha256': sha, 'source_step': 2000,
                'source_b2_update': 1000, 'uid': sample.uid, 'depth': 9, 'gt_parents': True,
                'steps': 20, 'threshold': .5, 'precision': precision, 'noise_device': args.device,
                'seeds': list(range(14_000_000, 14_000_064)), 'parameters_updated': False,
                'condition_sha256': condition_sha, 'smoke_model': source['smoke_model'],
                'diagnostics_are_not_gates': True, 'paired_source_file_sha256': file_sha(args.paired_noise)}
    write_json(args.output / 'identity.json', metadata)
    arrays = args.output / 'arrays'; arrays.mkdir()
    def run(noise, seed, steps, mode, name):
        noise_sha = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
        with amp(mode): value = sample_level(model, contexts[mode], parents, 9, noise.clone(), steps=steps)
        cells = expand_occupied_children(parents, value[0] >= .5)
        np.savez_compressed(arrays / f'{name}.npz', noise=noise.cpu().numpy(), parents=parents.cpu().numpy(),
                            target_occupancy=target.cpu().numpy(), estimate=value.cpu().numpy(),
                            predicted_cells=cells.cpu().numpy(), target_cells=gt.cpu().numpy())
        row = {'seed': seed, 'steps': steps, 'precision': mode, 'noise_sha256': noise_sha,
               **occupancy_metrics(value, target, cells, gt)}
        append_json(args.output / 'samples.jsonl', {'name': name, **row})
        return row
    rows = []
    for seed in metadata['seeds']:
        noise = noise_for(target, seed)
        # Preserve the actual CUDA draw before sampling, including in an interrupted run.
        np.savez_compressed(arrays / f'input-{seed}.npz', noise=noise.cpu().numpy())
        rows.append(run(noise, seed, 20, precision, f'fresh-{seed}'))
        write_json(args.output / 'status.json', {'state': 'evaluating', 'completed': len(rows)})
    paired = []
    saved = np.load(args.paired_noise)
    paired_noise = torch.from_numpy(saved['noise'].copy()).to(args.device)
    assert paired_noise.shape == target.shape
    for steps in [20, 40]:
        for mode in ['bf16', 'fp32']:
            paired.append(run(paired_noise, 13_000_002, steps, mode, f'paired-13000002-{steps}-{mode}'))
    failures = []
    for row in rows:
        if row['exact_coordinate_set']: continue
        seed = row['seed']
        noise = torch.from_numpy(np.load(arrays / f'input-{seed}.npz')['noise'].copy()).to(args.device)
        for steps, mode in [(20, 'fp32'), (40, precision), (40, 'fp32')]:
            failures.append(run(noise, seed, steps, mode, f'failure-{seed}-{steps}-{mode}'))
    assert versions == [p._version for p in model.parameters()]
    assert signature == (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
    report = {**metadata, 'sampling_passed': sampling_gate(rows), 'exact_count': sum(r['exact_coordinate_set'] for r in rows),
              'samples': rows, 'paired_13000002': paired, 'failure_diagnostics': failures,
              'failed_seeds': [r['seed'] for r in rows if not r['exact_coordinate_set']],
              'note': 'Seeds 14000000..14000063 are now consumed, not an unseen test for future tuning.'}
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'status.json', {'state': 'complete', 'sampling_passed': report['sampling_passed'], 'exact_count': report['exact_count']})
    print(json.dumps({'sampling_passed': report['sampling_passed'], 'exact_count': report['exact_count']}), flush=True)


if __name__ == '__main__':
    main()
