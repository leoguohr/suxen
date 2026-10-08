"""Recover C final evaluation from a frozen checkpoint after loss of instance logs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['code-root', 'checkpoint', 'manifest', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.code_root.resolve()))
    from mini_nexus.data_2k import Nexus2KManifestDataset
    from scripts.evaluate_vertex_b1_frozen import file_sha
    from scripts.evaluate_vertex_b2_frozen import base_config
    from scripts.train_vertex_a100_b1 import make_model
    from scripts.train_vertex_c import sample_tree
    from scripts.train_vertex_staged import autocast
    from scripts.train_vertex_overfit import write_json
    torch.set_num_threads(8)
    sha = file_sha(args.checkpoint); assert sha == args.expected_sha
    signature = (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = state['config']; original = base_config(config)
    assert state['step'] == 3800 and state['c_update'] == 1800 and config['phase'] == 'C'
    assert config['variant'] == 'R1' and config['uids'] == ['nexus_2k_000105']
    assert config['development_seeds'] == list(range(15_000_000, 15_000_004))
    assert config['final_seeds'] == list(range(16_000_000, 16_000_008))
    dataset = Nexus2KManifestDataset(args.manifest, 'train'); assert len(dataset) == 1
    sample = dataset[0]; assert sample.uid == 'nexus_2k_000105' and len(sample.quantized_vertices) == 8
    condition_sha = hashlib.sha256(sample.condition.numpy().tobytes()).hexdigest()
    assert condition_sha == original['condition_sha256'][sample.uid]
    model = make_model('R1', original['seed'], original['smoke_model'])
    model.load_state_dict(state['model'], strict=True)
    model.to(args.device).eval().requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == config['parameter_count']
    del state
    versions = [p._version for p in model.parameters()]
    args.precision = config['precision']
    identity = {'phase': 'C_recovery_evaluation', 'c_update': 1800, 'cumulative_step': 3800,
                'checkpoint_sha256': sha, 'training_config': config, 'parameters_updated': False,
                'condition_sha256': condition_sha, 'original_final_execution_unknown': True,
                'unseen_final_test_claimed': False, 'precision': args.precision,
                'note': 'Recovery rerun of predeclared seeds; original instance logs unavailable. No training or new seed selection.'}
    write_json(args.output / 'identity.json', identity)
    reports = {}
    with autocast(args):
        context = model.condition_encoder(sample.condition[None].to(args.device))
        for name, seeds in [('development_recovery', config['development_seeds']), ('final_recovery', config['final_seeds'])]:
            folder = args.output / name; folder.mkdir()
            local, full = [], []
            for seed in seeds:
                local.append(sample_tree(model, context, sample, seed, folder / f'gt-{seed}', gt_parents=True))
                full.append(sample_tree(model, context, sample, seed, folder / f'full-{seed}'))
                write_json(args.output / 'status.json', {'state': 'evaluating', 'group': name, 'completed': len(full), 'total': len(seeds)})
            report = {'phase': 'C_recovery_evaluation', 'c_update': 1800, 'cumulative_step': 3800,
                      'uid': sample.uid, 'local_gt_parents': local, 'full_generated_parents': full,
                      'local_all_layers_passed': all(x.get('all_levels_exact', False) for x in local),
                      'full_tree_passed': all(x['exact_coordinate_set'] for x in full),
                      'full_tree_exact_count': sum(x['exact_coordinate_set'] for x in full),
                      'seed_count': len(seeds), 'diagnostics_are_not_gates': True}
            reports[name] = report
            write_json(args.output / f'{name}.json', report)
    assert versions == [p._version for p in model.parameters()]
    assert signature == (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
    result = {**identity, 'development_full_exact': reports['development_recovery']['full_tree_exact_count'],
              'final_recovery_full_exact': reports['final_recovery']['full_tree_exact_count'],
              'final_recovery_full_tree_passed': reports['final_recovery']['full_tree_passed'],
              'final_recovery_local_all_layers_passed': reports['final_recovery']['local_all_layers_passed']}
    write_json(args.output / 'result.json', result)
    write_json(args.output / 'status.json', {'state': 'complete', 'final_recovery_full_exact': result['final_recovery_full_exact']})
    print(json.dumps({k:v for k,v in result.items() if k.startswith('final_')}), flush=True)


if __name__ == '__main__':
    main()
