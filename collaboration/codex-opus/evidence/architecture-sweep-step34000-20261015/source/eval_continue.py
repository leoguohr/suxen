"""Evaluate saved S0/S1 endpoints; start S2..S6 with the frozen original trainer."""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parent
GPU_UUIDS = ('GPU-6cf268fa-8e3b-ecfc-c88c-22bc22cc98c3',
            'GPU-6977060b-ce26-90fc-e2ba-8af1370f8c0e')
TERMINAL_SHA = {
    'S0': '3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935',
    'S1': '2126ce1be93405a45087af60bfed9e17db4e0377597cdd925b6c45e0146ce193',
}
FIRST_WRAPPER_SHA = {
    'resume_sweep.py': '57efa7248f192e8b4ecab245060a52dd8ebba90d7a13bd61b925326626b61502',
    'resume_queue.py': '1464e7485ec50d74cbfd9d250be3d2b84e212b66e76938026fd33b111f406f83',
}
PREVIOUS_WRAPPER_SHA = {
    'resume_sweep.py': '7b728718114eec32825328f5a5ae8f0c94a04b13c97c23a23c0cb7cd50afd133',
    'resume_queue.py': 'ea750edab13c74f6addea66658a0fc494ee779bf9d4e4a4664fa76105b4e0369',
}
CURRENT_WRAPPERS = ('eval_continue.py', 'eval_continue_queue.py')
EVIDENCE_SUFFIXES = ('.json', '.jsonl', '.log', '.md')
SCORING_SHA = {
    'replay.py': 'b4c21112f6aa386a89d6fbac16abcf859193f0741651a91f37d75f546caaf86a',
    'models.py': '679c6482a47ca40ed3bcb5e86eb6f407db105a5fb6cc41a6c8708b6e64127a0a',
    'teacher_ae.py': '740de899b2d7ea47ec56edf61101d45dee6251956505426e16c1f5cec9f6416c',
    'indicators.py': '9d3fc97d83194c7249f5f0542d5c1591e06de3010061f4be0db051012aef989b',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(args):
    """Keep installed Torch/NumPy ahead of the already prepared dependency directory."""
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    base = str(args.base_root.resolve())
    if base not in sys.path:
        sys.path.insert(0, base)
    deps = str(args.deps.resolve())
    assert args.deps.is_dir(), 'prepared dependency directory must exist'
    if deps not in sys.path:
        sys.path.append(deps)
    import train_sweep as train
    assert train.ROOT.resolve() == args.base_root.resolve(), 'wrong frozen trainer imported'
    train.runtime()
    import evaluate_ab
    import diffusers
    import numpy as np
    import torch
    import samplers
    scheduler = Path(inspect.getfile(diffusers.DPMSolverMultistepScheduler))
    assert diffusers.__version__ == samplers.DIFFUSERS_VERSION
    assert sha(scheduler) == samplers.SCHEDULER_SOURCE_SHA256
    assert evaluate_ab.SEEDS == (97029000, 97029001) and evaluate_ab.DEPTH == 9
    points = np.array([[0., 0., 0.], [1., 2., 3.]], dtype=np.float32)
    metric = evaluate_ab.xyz_metric(points[::-1].copy(), points)
    assert metric['status'] == 'applicable_equal_count' and metric['matched_coordinate_rmse'] == 0.
    for name, digest in SCORING_SHA.items():
        expected = train.AB_ROOT / 'source_runtime/teacher_candidate' / name
        assert Path(sys.modules[Path(name).stem].__file__).resolve() == expected.resolve()
        assert sha(expected) == digest
    environment = {'deps': deps, 'dependency_path_mode': 'append_after_installed_packages',
                   'torch_version': torch.__version__, 'torch_path': torch.__file__,
                   'numpy_version': np.__version__, 'numpy_path': np.__file__,
                   'diffusers_version': diffusers.__version__, 'diffusers_path': diffusers.__file__,
                   'scheduler_source_sha256': sha(scheduler),
                   'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'],
                   'xyz_metric_cpu_permuted_two_point_check': metric,
                   'complete_evaluate_ab_import_cpu': True}
    return train, environment


def code_sources(args, config, train):
    """Resolve each generation of wrapper against its own directory, including same names."""
    sources = {}
    for name, digest in config['code_sha256'].items():
        path = train.AB_ROOT / name[3:] if name.startswith('ab/') else args.base_root / name
        assert sha(path) == digest, f'frozen original code differs: {name}'
        sources[name] = {'path': str(path), 'sha256': digest, 'role': 'frozen_original'}
    assert {'train_sweep.py', 'architecture_variants.py', 'run_sweep_queue.py'} <= sources.keys()
    first = config.get('recovery400', config.get('recovery'))
    previous = config['recovery7600']
    assert first['resume_update'] == 400 and first['runtime_wrapper_sha256'] == FIRST_WRAPPER_SHA
    assert previous['resume_update'] == 7600 and previous['runtime_wrapper_sha256'] == PREVIOUS_WRAPPER_SHA
    assert first['next_event'] == 3200 and first['remaining_updates'] == 9600
    assert previous['next_event'] == 60800 and previous['remaining_updates'] == 2400
    assert first['training_configuration_changed'] is previous['training_configuration_changed'] is False
    assert previous['previous_recovery']['metadata'] == first
    assert previous['previous_recovery']['sha256'] == FIRST_WRAPPER_SHA
    for role, root, expected in (('historical_recovery400', args.first_code, FIRST_WRAPPER_SHA),
                                 ('historical_recovery7600', args.previous_code, PREVIOUS_WRAPPER_SHA)):
        for name, digest in expected.items():
            path = root / name
            assert sha(path) == digest, f'{role} code differs: {name}'
            sources[f'{role}/{name}'] = {'path': str(path), 'sha256': digest, 'role': role}
    for name in CURRENT_WRAPPERS:
        path = ROOT / name
        sources[f'actual_runtime_wrapper/{name}'] = {
            'path': str(path), 'sha256': sha(path), 'role': 'actual_runtime_wrapper'}
    for name, digest in SCORING_SHA.items():
        relative = f'ab/source_runtime/teacher_candidate/{name}'
        path = train.AB_ROOT / relative[3:]
        assert sha(path) == digest, f'actual scoring code differs: {name}'
        sources[relative] = {'path': str(path), 'sha256': digest, 'role': 'actual_scoring_runtime'}
    return sources


def terminal_headers(args, branch):
    folder = args.original_runs / branch
    checkpoint = folder / 'checkpoint-034000.pt'
    identity = json.loads(checkpoint.with_suffix('.json').read_text())
    assert identity == json.loads((folder / 'checkpoint_identity.json').read_text())
    assert Path(identity['path']) == checkpoint and checkpoint.is_file()
    assert identity['sha256'] == TERMINAL_SHA[branch]
    assert identity['step'] == 34000 and identity['sweep_update'] == 10000
    assert identity['schedule_next_event'] == 80000
    assert all(identity[key] is True for key in
               ('final', 'full_model_adam_rng', 'atomic_write_fsynced', 'readback_verified'))
    assert checkpoint.stat().st_size == identity['bytes']
    status = json.loads((folder / 'status.json').read_text())
    assert status['training_complete'] is True and status['sweep_update'] == 10000
    assert status.get('evaluation_complete') is not True, 'already evaluated branch needs no recovery'
    return checkpoint, identity, json.loads((folder / 'config.json').read_text())


def restore_terminal_model(source, branch, device, train, *, small=False):
    """Read and verify Adam ages, but never construct or restore an optimizer."""
    import numpy as np
    import torch
    from architecture_variants import construct_model
    assert branch in TERMINAL_SHA
    assert source['step'] == 34000 and source['sweep_update'] == 10000
    assert source['schedule_next_event'] == 80000 and source['scheduler'] is None
    assert source['source_checkpoint_sha256'] == train.SOURCE_SHA
    config = source['config']
    for key, value in {'phase': 'NEXUS_VERTEX_ARCH_SWEEP_10000', 'branch': branch,
                       'source_step': 24000, 'source_checkpoint_sha256': train.SOURCE_SHA,
                       'updates': 10000, 'micros_per_update': 8, 'microgroup': 2,
                       'max_depth': 9, 'rope_reference_depth': 9, 'recompute': False,
                       'scheduler': None, 'sweep_start_cursor': 0,
                       'source_cursor_provenance_only': 16000, 'rng_namespace': train.NAMESPACE}.items():
        assert config[key] == value, key
    counts = np.zeros((train.COUNT, train.DEPTH), dtype=np.int64)
    for index in range(train.EVENTS):
        mesh, depth, _ = train.event_at(index)
        counts[mesh, depth] += 1
    train.assert_same(source['sample_depth_counts'], counts)
    provenance = source['source_provenance']
    assert provenance['checkpoint_sha256'] == train.SOURCE_SHA and provenance['step'] == 24000
    assert provenance['d9_update'] == 2000 and provenance['schedule_next_event'] == 16000
    assert int(provenance['sample_depth_counts'].sum()) == 16000
    with torch.device('meta'):
        model = construct_model(branch, small=small)
    expected = model.state_dict()
    assert expected.keys() == source['model'].keys()
    for name, value in expected.items():
        saved = source['model'][name]
        assert value.shape == saved.shape and value.dtype == saved.dtype, name
    model.load_state_dict({name: value.to(device=device, copy=True)
                           for name, value in source['model'].items()}, strict=True, assign=True)
    model.eval().requires_grad_(False)
    model.flow.use_checkpoint = False
    model.condition_encoder.use_checkpoint = False
    current = dict(model.named_parameters())
    initial = source['optimizer_initial_steps']
    assert list(initial) == list(current)
    assert initial == {name: (0 if branch == 'S1' and name.startswith('flow.final_modulation.')
                             else 24000) for name in current}
    saved_optimizer = source['optimizer']
    assert len(saved_optimizer['param_groups']) == 1
    group = saved_optimizer['param_groups'][0]
    ids = group['params']
    assert ids == list(range(len(current))) and set(saved_optimizer['state']) == set(ids)
    for name, value in {'lr': 1e-5, 'weight_decay': 0, 'betas': (.9, .999),
                        'eps': 1e-8, 'foreach': False, 'amsgrad': False, 'maximize': False,
                        'capturable': False, 'differentiable': False}.items():
        assert group[name] == value, name
    for (name, parameter), param_id in zip(current.items(), ids):
        state = saved_optimizer['state'][param_id]
        assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'}
        assert state['step'].numel() == 1 and float(state['step']) == initial[name] + 10000
        for key in ('exp_avg', 'exp_avg_sq'):
            assert state[key].shape == parameter.shape and state[key].dtype == parameter.dtype
    train.assert_same(model.state_dict(), source['model'])
    assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
    return model


def verify_training_history(folder, manifest, train):
    updates = 0
    with (folder / 'train.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            updates += 1
            assert row['sweep_update'] == updates and row['step'] == 24000 + updates
            assert len(row['micros']) == 8
            for micro, event in enumerate(row['micros']):
                index = (updates - 1) * 8 + micro
                mesh, depth, occurrence = train.event_at(index)
                uid = manifest['train_uids'][mesh]
                assert event['event_index'] == index and event['micro'] == micro
                assert event['mesh_index'] == mesh and event['uid'] == uid
                assert event['depth'] == depth + 1 and event['occurrence'] == occurrence
                assert event['rng_key'] == f'{train.NAMESPACE}|{index}|{uid}|{depth + 1}|{occurrence}'
    assert updates == 10000
    return {'path': str(folder / 'train.jsonl'), 'sha256': sha(folder / 'train.jsonl'),
            'continuous_updates': updates, 'events': 80000, 'last_cumulative_step': 34000}


def archive_evidence(source, target):
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in EVIDENCE_SUFFIXES:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def annotate_generation_metadata(environment):
    """Add runtime provenance while forwarding every original sampling argument exactly."""
    import evaluate_ab
    original = evaluate_ab.native_export.export_tree
    identity = {'path': str(ROOT / 'eval_continue.py'), 'sha256': sha(ROOT / 'eval_continue.py')}

    def export(*arguments, **keywords):
        keywords['metadata'] = {**keywords['metadata'], 'actual_runtime_wrapper': identity,
                                'dependency_environment': environment}
        return original(*arguments, **keywords)

    evaluate_ab.native_export.export_tree = export
    return evaluate_ab.native_export, original


def evaluate_endpoint(args, train, environment):
    import torch
    from architecture_variants import cache_fixed_features
    assert args.branch in TERMINAL_SHA and not args.output.exists()
    assert torch.cuda.device_count() == 1 and os.environ.get('CUDA_VISIBLE_DEVICES') in GPU_UUIDS
    checkpoint, identity, config = terminal_headers(args, args.branch)
    assert sha(checkpoint) == TERMINAL_SHA[args.branch]
    sources = code_sources(args, config, train)
    source = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    train.assert_same(source['config'], config)
    assert config['torch_version'] == torch.__version__
    assert config['data_manifest_sha256'] == sha(args.data / 'manifest.json')
    model = restore_terminal_model(source, args.branch, torch.device('cuda:0'), train)
    manifest, conditions, levels = train.runtime().load_data(args.data, torch.device('cuda:0'))
    cache = cache_fixed_features(model, conditions)
    train.assert_same(cache, config['immutable_feature_cache'])
    train.restore_rng(source, torch.device('cuda:0'))
    coverage = json.loads((args.original_runs / args.branch / 'coverage.json').read_text())
    train.assert_same(coverage['counts'], source['sample_depth_counts'].tolist())
    assert coverage['uids'] == manifest['train_uids']
    assert coverage['total_events'] == coverage['next_event'] == 80000
    assert coverage['new_optimizer_updates'] == 10000 and coverage['cumulative_step'] == 34000
    history = verify_training_history(args.original_runs / args.branch, manifest, train)
    del source, levels
    gc.collect()
    torch.cuda.empty_cache()
    args.output.mkdir(parents=True)
    archive_evidence(args.original_runs / args.branch, args.output / 'previous_attempt')
    for name in ('train.jsonl', 'config.json', 'startup.json', 'coverage.json',
                 'checkpoint_identity.json', 'checkpoint-034000.json'):
        shutil.copy2(args.original_runs / args.branch / name, args.output / name)
    assert sha(args.output / 'train.jsonl') == history['sha256']
    runtime_config = copy.deepcopy(config)
    runtime_config['training_code_sha256'] = copy.deepcopy(config['code_sha256'])
    runtime_config['code_sha256'] = {name: item['sha256'] for name, item in sources.items()}
    runtime_config['evaluation_recovery'] = {
        'mode': 'frozen_terminal_model_evaluation_only', 'training_updates_this_attempt': 0,
        'checkpoint_reference': identity, 'checkpoint_copied': False,
        'actual_runtime_wrapper': str(ROOT / 'eval_continue.py'),
        'code_sources': sources, 'environment': environment, 'training_history': history}
    train.write_json(args.output / 'evaluation_runtime_config.json', runtime_config)
    train.write_json(args.output / 'endpoint_reference.json', runtime_config['evaluation_recovery'])
    status = {'state': 'evaluation', 'sweep_update': 10000, 'training_complete': True,
              'evaluation_complete': False, 'checkpoint': identity, 'training_updates_this_attempt': 0}
    train.write_json(args.output / 'status.json', status)
    try:
        train.evaluate_model(model, conditions, manifest, args.data, args.output / 'evaluation',
                             args.branch, identity, runtime_config)
        train.write_json(args.output / 'evaluation/runtime_identity.json', runtime_config['evaluation_recovery'])
        train.write_json(args.output / 'status.json', {**status, 'state': 'complete', 'evaluation_complete': True})
    except Exception as error:
        train.write_json(args.output / 'status.json', {**status, 'state': 'failed_no_automatic_retry',
                         'error': repr(error), 'traceback': traceback.format_exc()})
        raise


def run(args):
    train, environment = bootstrap(args)
    if args.command == 'import-check':
        print(json.dumps(environment, indent=2), flush=True)
        return
    import torch
    torch.set_num_threads(8)
    assert args.microgroup == 2 and args.start_sha == train.SOURCE_SHA
    exporter, original_export = annotate_generation_metadata(environment)
    try:
        if args.command == 'evaluate-endpoint':
            evaluate_endpoint(args, train, environment)
            return
        assert args.branch in train.BRANCHES[2:], 'only unstarted S2..S6 may invoke training'
        assert torch.cuda.device_count() == 1 and os.environ.get('CUDA_VISIBLE_DEVICES') in GPU_UUIDS
        config = json.loads((args.original_runs / 'S0/config.json').read_text())
        sources = code_sources(args, config, train)
        original_hashes = train.code_hashes

        def effective_hashes(ab):
            return {**original_hashes(ab), **{name: item['sha256'] for name, item in sources.items()
                                           if item['role'] != 'frozen_original'}}

        train.code_hashes = effective_hashes
        try:
            train.train(args, preflight=False)
        finally:
            train.code_hashes = original_hashes
    finally:
        exporter.export_tree = original_export


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('command', choices=('evaluate-endpoint', 'train', 'import-check'))
    for name in ('base-root', 'original-runs', 'first-code', 'previous-code', 'deps'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--branch', choices=tuple(f'S{i}' for i in range(7)))
    result.add_argument('--output', type=Path)
    result.add_argument('--data', type=Path)
    result.add_argument('--start', type=Path)
    result.add_argument('--start-sha', default='2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba')
    result.add_argument('--microgroup', type=int, choices=(2,), default=2)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
