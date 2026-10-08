"""Fresh CPU audit of saved Soft4 control; no model import/forward or optimizer step."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import datetime
import hashlib
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def same(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def read(path):
    return json.loads(Path(path).read_text())


def audit(args, result):
    base = args.project_root.resolve()
    expected = read(args.historical_audit)
    def local(path):
        return base / Path(path).relative_to('/guohaoran/nexus_fast_track')
    source, control = local(expected['source']), local(expected['control'])
    data_root = base / 'diagnostics/teacher_cad50_fresh512_20260921'
    parent_dir = source.parent
    cfg = read(control / 'config.json')
    assert sha(source) == expected['source_sha256']
    assert sha(control / 'config.json') == expected['control_config_sha256']
    assert sha(control / 'updates.jsonl') == expected['control_updates_sha256']
    assert cfg['source_sha256'] == expected['source_sha256'] and cfg['mode'] == 'full'
    assert cfg['stop_weight_grad'] is False
    assert cfg['lrs'] == dict(encoder_mu=3e-6, decoder=3e-5, edge_head=3e-5, face_head=3e-5)
    assert cfg['betas'] == [.9, .999] and cfg['eps'] == 1e-8 and cfg['weight_decay'] == 0 and cfg['clip'] == 1
    assert cfg['new_updates'] == 100 and cfg['checkpoints'] == [0, 25, 50, 75, 100]
    assert cfg['initial_completed_updates'] == 2500 and cfg['final_completed_updates'] == 2600
    assert cfg['microbatch'] == 1 and cfg['meshes_per_update'] == 50
    assert cfg['sampling'] is False and cfg['KL'] == 0 and cfg['logvar_frozen']
    assert cfg['totals'] == dict(meshes=50, vertices=2872, edges=8364, faces=5576, pairs=235741, face_pool=13946)
    env = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version())
    assert {**cfg['environment'], 'python': cfg['environment']['python'].split()[0]}.items() >= env.items()
    result.update(source=str(source), source_sha256=sha(source), control=str(control), environment=env,
                  control_config_sha256=sha(control / 'config.json'), control_updates_sha256=sha(control / 'updates.jsonl'))
    data = read(data_root / 'data/manifest.json')
    pools = read(data_root / 'pool_manifest.json')
    assert sha(data_root / 'data/manifest.json') == cfg['data_manifest_sha256']
    assert sha(data_root / 'pool_manifest.json') == cfg['pool_manifest_sha256']
    for row in data['meshes']:
        assert sha(data_root / 'data' / row['path']) == row['sha256'], row['path']
    for row in pools['records']:
        assert sha(data_root / row['path']) == row['sha256'], row['path']
    assert data['uids'] == cfg['uids'] and len(data['uids']) == 50
    for name, digest in cfg['code_sha256'].items():
        assert sha(control / name) == digest, name
    dependencies = []
    for row in expected['dependencies']:
        current = local(row['path'])
        archive = control / 'source_archive' / current.relative_to(base)
        assert sha(current) == sha(archive) == row['sha256'], str(current)
        dependencies.append(dict(path=str(current), sha256=row['sha256']))
    archived = list((control / 'source_archive').rglob('*.py'))
    assert len(archived) == len(dependencies)
    result.update(data_pool_verified=True, code_hashes_match_original_run=True, dependencies=dependencies)
    print('PASS identities, environment, effective source, 50 data and 50 pools', flush=True)
    logs = [json.loads(x) for x in (control / 'updates.jsonl').read_text().splitlines()]
    assert [x['new_update'] for x in logs] == list(range(1, 101))
    for step, row in enumerate(logs, 1):
        assert row['completed_updates'] == 2500 + step and row['state_before'] == 2499 + step
        assert [m['uid'] for m in row['meshes']] == data['uids']
        assert row['lrs'] == cfg['lrs'] and set(row['adam_steps'].values()) == {2500 + step}
        assert row['participation_per_mesh'] == 2500 + step
        assert row['clip_coefficient'] == min(1., 1 / (row['total_grad_norm'] + 1e-6))
        assert np.isfinite(row['total_loss_before']) and np.isfinite(row['total_grad_norm'])
        assert all(np.isfinite(v['delta_l2']) for v in row['actual_updates'].values())
    done = read(control / 'complete.json')
    assert done['new_updates'] == 100 and done['completed_updates'] == 2600 and done['stopped_at_budget']
    assert not (control / 'failure.json').exists()
    parent = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    start = torch.load(control / 'checkpoint-new0000-step2500.pt', map_location='cpu', mmap=True, weights_only=False)
    for key in ['model', 'optimizer', 'rng', 'participation', 'completed_updates']:
        assert same(start[key], parent[key]), key
    assert cfg['trainable_groups'] == parent['config']['trainable_groups']
    parameter_map = []
    assert [g['name'] for g in parent['optimizer']['param_groups']] == list(cfg['trainable_groups'])
    for group in parent['optimizer']['param_groups']:
        names = cfg['trainable_groups'][group['name']]
        assert len(names) == len(group['params'])
        for name, pid in zip(names, group['params']):
            full_name = 'autoencoder.' + name
            parameter = parent['model'][full_name]
            state = parent['optimizer']['state'][pid]
            assert state['exp_avg'].shape == state['exp_avg_sq'].shape == parameter.shape
            assert int(state['step']) == 2500
            parameter_map.append(dict(group=group['name'], name=full_name, optimizer_id=pid, shape=list(parameter.shape)))
    assert len(parameter_map) == len(parent['optimizer']['state'])
    result.update(model_adam_rng_progress_groups_exact=True, parameter_map=parameter_map, all100_updates_logged=True)
    del start
    print('PASS complete parent/start model, Adam, RNG, progress and parameter map', flush=True)
    checkpoints, prediction_count, trends = [], 0, []
    parent_eval = read(parent_dir / 'eval-new0500.json')
    for record in expected['checkpoint_identity_checks']:
        step = record['step']
        ev = read(control / f'eval-new{step:04d}.json')
        path = local(record['path'])
        assert local(ev['checkpoint']) == path
        assert sha(path) == record['sha256'] == ev['checkpoint_sha256']
        assert path.stat().st_size == record['bytes']
        cp = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert cp['completed_updates'] == 2500 + step and list(cp['participation']) == data['uids']
        assert set(cp['participation'].values()) == {2500 + step}
        assert len(cp['optimizer']['state']) == len(parameter_map)
        assert all(int(s['step']) == 2500 + step for s in cp['optimizer']['state'].values())
        for g, pg in zip(cp['optimizer']['param_groups'], parent['optimizer']['param_groups']):
            assert same(g, pg)
        for name in parent['model']:
            if name.startswith('autoencoder.log_variance.'):
                assert same(cp['model'][name], parent['model'][name])
        assert ev['new_step'] == step and ev['completed_updates'] == 2500 + step
        assert [r['uid'] for r in ev['meshes']] == data['uids']
        for row in ev['meshes']:
            pred_path = control / row['prediction_path']
            assert sha(pred_path) == row['prediction_sha256'] and row['face']['complete']
            with np.load(pred_path) as q, np.load(data_root / 'pools' / (row['uid'] + '.npz')) as pool:
                n = len(q['vertices'])
                pe, ge, fi, gf = [q[k] for k in ['predicted_edge_ids', 'gt_edges', 'actual_face_candidate_ids', 'gt_faces']]
                ek = lambda a: a[:, 0].astype(np.int64) * n + a[:, 1]
                fk = lambda a: (a[:, 0].astype(np.int64) * n + a[:, 1]) * n + a[:, 2]
                assert np.array_equal(q['vertices'], pool['vertices'])
                assert np.array_equal(ge, np.unique(np.sort(pool['edges'].T, axis=1), axis=0))
                assert np.array_equal(gf, np.sort(pool['positive'], axis=1))
                assert ((pe[:, 0] < pe[:, 1]) & (pe[:, 0] >= 0) & (pe[:, 1] < n)).all()
                assert (fi[:, 0] < fi[:, 1]).all() and (fi[:, 1] < fi[:, 2]).all()
                assert len(np.unique(ek(pe))) == len(pe) and (q['predicted_edge_logits'] > 0).all()
                tp = int(np.isin(ek(pe), ek(ge)).sum())
                assert [tp, len(pe)-tp, len(ge)-tp] == [row['edge'][k] for k in ['tp', 'fp', 'fn']]
                labels, positive = np.isin(fk(fi), fk(gf)), q['actual_face_candidate_logits'] > 0
                assert np.isfinite(q['actual_face_candidate_logits']).all()
                assert len(np.unique(fk(fi))) == len(fi) and np.array_equal(labels, q['actual_face_candidate_gt'])
                tp, fp = int((positive & labels).sum()), int((positive & ~labels).sum())
                assert [tp, fp, len(gf)-tp] == [row['face'][k] for k in ['tp', 'fp', 'fn']]
                adjacency = [set() for _ in range(n)]
                for i, j in pe:
                    adjacency[int(i)].add(int(j))
                assert sum(len(adjacency[i] & adjacency[j]) for i in range(n) for j in adjacency[i]) == len(fi)
                for i, j in [(0, 1), (0, 2), (1, 2)]:
                    assert np.isin(fi[:, i].astype(np.int64) * n + fi[:, j], ek(pe)).all()
                covered = np.isin(fk(gf), fk(fi))
                assert np.array_equal(covered, q['gt_face_covered'])
                assert int((~covered).sum()) == row['missing_gt_face_candidates'] == row['face']['fn_missing_candidate']
                assert row['face']['fn_missing_candidate'] + row['face']['fn_present_but_negative'] == row['face']['fn']
                in_pool = np.isin(fk(fi), fk(np.concatenate([pool['positive'], pool['mixed']])))
                assert int((positive & ~labels & in_pool).sum()) == row['face']['actual_fp_inside_training_pool']
                assert int((positive & ~labels & ~in_pool).sum()) == row['face']['actual_fp_outside_training_pool']
                assert row['joint_perfect'] == all(row[k][m] == 0 for k in ['edge', 'face'] for m in ['fp', 'fn'])
                if step == 0:
                    ref = next(x for x in parent_eval['meshes'] if x['uid'] == row['uid'])
                    with np.load(parent_dir.parent / ref['prediction_path']) as old:
                        assert q.files == old.files and all(np.array_equal(q[k], old[k]) for k in q.files)
            prediction_count += 1
        for kind in ['edge', 'face']:
            counts = {k: sum(r[kind][k] for r in ev['meshes']) for k in ['tp', 'fp', 'fn', 'tn']}
            assert all(ev['counts'][kind][k] == v for k, v in counts.items())
            assert ev['counts'][kind]['micro_f1'] == 2*counts['tp']/max(2*counts['tp']+counts['fp']+counts['fn'], 1)
        assert ev['perfect_uids'] == [r['uid'] for r in ev['meshes'] if r['joint_perfect']]
        assert ev['joint_perfect'] == len(ev['perfect_uids'])
        if step == 0:
            assert ev['counts'] == parent_eval['counts'] and ev['perfect_uids'] == parent_eval['perfect_uids']
        checkpoints.append(record)
        trends.append(dict(step=step, counts=ev['counts'], joint_perfect=ev['joint_perfect']))
        del cp
        print('PASS checkpoint, complete 50-mesh prediction recount', step, flush=True)
    ready_path = control / 'ready.json'
    ready = read(ready_path)
    assert ready['passed'] and ready['source_sha256'] == expected['source_sha256']
    assert ready['optimizer_updates'] == 0 and ready['model_optimizer_rng_exact']
    assert ready['baseline'] == parent_eval['counts']
    assert ready['device_check']['uid'] == 'teacher_cad50_20'
    assert prediction_count == 250 and sha(source) == expected['source_sha256']
    assert not torch.cuda.is_initialized()
    result.update(passed=True, reuse_control=True, stored_control_evidence_passed=True, checkpoint_identity_checks=checkpoints,
                  prediction_files_hashed=prediction_count, predictions_recounted=prediction_count,
                  complete_triangle_enumerations=True, step0_full50_prediction_arrays_equal_parent=True,
                  control_ready_path=str(ready_path), control_ready_sha256=sha(ready_path),
                  control_device_check=ready['device_check'], control_ready_archive=ready,
                  control_trends=trends, new_branch_numerical_gate='pending separately: exact zero-init forward, old preclip gradient, Adam/RNG mapping',
                  optimizer_updates=0, model_forwards=0, gpu_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--historical-audit', type=Path, default=Path(__file__).with_name('HISTORICAL_CONTROL_REUSE_AUDIT.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(passed=False, reviewed_by='GPT-6 Astra / xhigh', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  script_sha256=sha(__file__), historical_audit_sha256=sha(args.historical_audit))
    try:
        audit(args, result)
    except BaseException:
        result['error'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('FRESH_CONTROL_CPU_AUDIT_PASS', flush=True)
