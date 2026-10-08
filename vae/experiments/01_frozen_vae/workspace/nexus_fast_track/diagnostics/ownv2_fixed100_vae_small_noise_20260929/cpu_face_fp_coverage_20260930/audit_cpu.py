"""Saved-prediction coverage audit. CPU only; no model or optimizer is loaded."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch


def no_cuda(*args, **kwargs):
    raise RuntimeError('This audit must never initialize CUDA')


torch.cuda._lazy_init = no_cuda
torch.cuda.init = no_cuda
torch.set_num_threads(1)


def read(path):
    return json.loads(Path(path).read_text())


def keys(ids, n):
    a = np.asarray(ids, dtype=np.int64).reshape(-1, 3)
    return (a[:, 0]*n+a[:, 1])*n+a[:, 2]


def relation(start, end):
    if start and end:
        return 'fp_at_both_endpoints'
    return 'start_only_fp' if start else 'end_only_fp'


def coverage(rows):
    h = Counter(r['hit_count'] for r in rows)
    return dict(count=len(rows), hit0=h[0], hit1=h[1], hit_ge2=sum(v for k, v in h.items() if k >= 2),
                histogram={str(k): h[k] for k in range(26)})


def self_test():
    assert int(keys([[2544, 2545, 2546]], 2547)[0]) > 2**31
    assert relation(True, True) == 'fp_at_both_endpoints'
    assert relation(True, False) == 'start_only_fp'
    assert relation(False, True) == 'end_only_fp'
    h = coverage([dict(hit_count=x) for x in [0, 0, 1, 3]])
    assert (h['hit0'], h['hit1'], h['hit_ge2']) == (2, 1, 1)
    pools = [{1, 2}, {2, 3}, {2, 4}]
    assert [sum(k in p for p in pools) for k in [0, 1, 2]] == [0, 1, 3]
    print('synthetic_tests_passed; CUDA_initialized=', torch.cuda.is_initialized(), flush=True)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(root, output):
    started = time.time()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'code'))
    import data_objective as d
    import stream_faces as sf
    expected_code = {
        'data_objective.py': '3e52edf63e0332bd3bf777826d5eb9e384f781b55ef87d67640a5b7ea06718e6',
        'stream_faces.py': '2fe8da1cc00aac733b8b0536a8169c0ee0c15b5ac000ff9265322af524fe3942',
        'train_vae.py': 'eb33ee0d1ce76ba55872fe2bd7b0d1d07993f6eea77ff856ec1ed42c3467433b'}
    hashes = {}

    def verified(path, expected=None):
        actual = d.sha(path)
        if expected is not None:
            assert actual == expected, (str(path), actual, expected)
        hashes[str(path)] = actual
        return actual

    for name, expected in expected_code.items():
        verified(root/'code'/name, expected)
    verified(root/'run/updates.jsonl', 'a2edb6872309fe8c5be32bc611876b7fad736db187391013e89f87eee4b63c6e')
    data_root = Path('/guohaoran/nexus_fast_track/diagnostics/shared_edge_head_fixed100_20260917')
    items, manifest = d.load_dataset(data_root)
    verified(data_root/'selection.json', d.SELECTION_SHA)
    verified(data_root/'overfit100_manifest.csv', d.MANIFEST_SHA)
    logs = [json.loads(line) for line in (root/'run/updates.jsonl').read_text().splitlines() if line]
    assert [r['update'] for r in logs] == list(range(34221, 34721))
    records = {uid: [] for uid in items}
    for index, r in enumerate(logs):
        epoch, group = 1711+index//20, index % 20
        assert (r['epoch'], r['group'], r['new_update']) == (epoch, group, index+1)
        assert r['uids'] == d.epoch_batches(manifest['uids'], epoch, seed=0)[group]
        assert [m['uid'] for m in r['meshes']] == r['uids']
        for m in r['meshes']:
            records[m['uid']].append((epoch, r['update'], m))
    assert all([e for e, _, _ in x] == list(range(1711, 1736)) for x in records.values())
    conditions = ['mu']+[f'noise-{x}' for x in range(861001, 861006)]
    step_hashes = {34220: 'ed9b29decf0c7e788d36e9273f572e034ef15a00371ce317ae6c9efdbeedde04',
                   34720: '7fb1e2a5d128762e147a4b87524d4cc212909625b706841c80e3519576e05562'}
    complete = {}
    for step, expected in step_hashes.items():
        for condition in conditions:
            p = root/f'run/evaluations/update-{step:08d}'/condition/'complete.json'
            verified(p)
            c = read(p)
            assert c['complete'] and c['native_forward'] and c['optimizer_updates_during_evaluation'] == 0
            assert c['identity']['checkpoint_sha256'] == expected
            assert [m['uid'] for m in c['meshes']] == manifest['uids']
            complete[step, condition] = {m['uid']: m for m in c['meshes']}

    def load_prediction(step, condition, uid, item):
        directory = root/f'run/evaluations/update-{step:08d}'/condition/uid
        expected_identity = dict(checkpoint_sha256=step_hashes[step], manifest_sha256=d.MANIFEST_SHA,
                                 group=condition, evaluation_seed=None if condition == 'mu' else int(condition[6:]), uid=uid)
        meta = read(directory/'network-output.json')
        result = read(directory/'result.json')
        state = read(directory/'face_shards/progress.json')
        for name in ('network-output.json', 'result.json', 'face_shards/progress.json'):
            verified(directory/name)
        assert meta['identity'] == result['identity'] == state['binding']['identity'] == expected_identity
        assert result['complete'] and state['complete'] and meta['native_forward']
        assert result == complete[step, condition][uid]
        verified(directory/'network-output.npz', meta['sha256'])
        n = len(item['vertices'])
        gt = item['gt_faces'].numpy()
        gt_edges = item['edges'].numpy()
        with np.load(directory/'network-output.npz', allow_pickle=False) as z:
            assert np.array_equal(z['vertices'], item['vertices'].numpy())
            assert np.array_equal(z['gt_faces'], gt) and np.array_equal(z['gt_edges'], gt_edges)
            assert np.array_equal(z['local_vertex_indices'], np.arange(n))
            edges = z['predicted_edges'].copy()
        assert edges.shape[1:] == (2,) and np.all(edges[:, 0] < edges[:, 1])
        assert len(np.unique(edges, axis=0)) == len(edges)
        adjacency = np.zeros((n, n), dtype=bool)
        adjacency[edges[:, 0], edges[:, 1]] = True
        assert hashlib.sha256(adjacency.tobytes()+gt.tobytes()).hexdigest() == state['binding']['input_sha256']
        gt_edge_keys = gt_edges[:, 0]*n+gt_edges[:, 1]
        etp = int(np.isin(edges[:, 0].astype(np.int64)*n+edges[:, 1], gt_edge_keys).sum())
        ec = dict(tp=etp, fp=len(edges)-etp, fn=len(gt_edges)-etp, tn=n*(n-1)//2-len(gt_edges)-(len(edges)-etp))
        assert ec == result['edge'] == meta['edge']
        enumerator = iter(sf.triangle_chunks(adjacency, chunk_size=state['binding']['chunk_size']))
        gt_keys = keys(gt, n)
        negative_candidates, fp = {}, set()
        counts = Counter()
        for shard in range(state['shards']):
            path = directory/f'face_shards/part-{shard:08d}.npz'
            verified(path, state['last_shard_sha256'] if shard == state['shards']-1 else None)
            ids_expected, _ = next(enumerator)
            with np.load(path, allow_pickle=False) as z:
                ids, logits, labels = z['ids'], z['logits'], z['labels']
                assert np.array_equal(ids, ids_expected) and np.isfinite(logits).all()
                ik = keys(ids, n)
                assert np.array_equal(labels, np.isin(ik, gt_keys))
                positive = logits > 0
                counts.update(tp=int((positive & labels).sum()), fp=int((positive & ~labels).sum()),
                              tn=int((~positive & ~labels).sum()), candidates=len(ids))
                negative_candidates.update({int(k): float(v) for k, v in zip(ik[~labels], logits[~labels])})
                fp.update(int(k) for k in ik[positive & ~labels])
        assert next(enumerator, None) is None
        counts['fn'] = len(gt)-counts['tp']
        covered = adjacency[gt[:, 0], gt[:, 1]] & adjacency[gt[:, 0], gt[:, 2]] & adjacency[gt[:, 1], gt[:, 2]]
        assert int((~covered).sum()) == state['fn_missing'] == result['face_fn_missing']
        assert int(covered.sum())-counts['tp'] == state['fn_present'] == result['face_fn_present']
        assert all(counts[k] == state[k] for k in ('tp', 'fp', 'tn', 'fn', 'candidates'))
        assert {k: counts[k] for k in ('tp', 'fp', 'tn', 'fn')} == result['face']
        return dict(scores=negative_candidates, fp=fp, edge_strict=ec['fp'] == ec['fn'] == 0, result=result)

    rows, replay, per_mesh = [], [], []
    for number, (uid, item) in enumerate(items.items(), 1):
        n = len(item['vertices'])
        saved = {(step, condition): load_prediction(step, condition, uid, item)
                 for step in step_hashes for condition in conditions}
        target_keys = set().union(*(p['fp'] for p in saved.values()))
        hit_epochs = {k: [] for k in target_keys}
        for epoch, update, m in records[uid]:
            negatives, digest = d.negative_faces(item, epoch, seed=0)
            assert digest == m['negative_sha256'], (uid, epoch, 'negative replay mismatch')
            assert len(negatives) == m['face_negatives'] and len(item['gt_faces']) == m['face_positives']
            negkeys = set(int(k) for k in keys(negatives.numpy(), n))
            hit = target_keys & negkeys
            for k in hit:
                hit_epochs[k].append(epoch)
            replay.append(dict(uid=uid, epoch=epoch, update=update, negatives=len(negatives),
                               recorded_sha256=m['negative_sha256'], replay_sha256=digest, match=True,
                               endpoint_fp_union_hits=len(hit)))
        gt_e = set(int(k) for k in item['edges'].numpy() @ np.array([n, 1], dtype=np.int64))
        for condition in conditions:
            a, b = saved[34220, condition], saved[34720, condition]
            mesh_rows = []
            for k in sorted(a['fp'] | b['fp']):
                i, rem = divmod(k, n*n)
                j, v = divmod(rem, n)
                sa, sb = a['scores'].get(k), b['scores'].get(k)
                row = dict(condition=condition, uid=uid, vertices=n, i=i, j=j, k=v,
                    start_candidate_present=sa is not None, end_candidate_present=sb is not None,
                    start_logit=sa, end_logit=sb, start_fp=k in a['fp'], end_fp=k in b['fp'],
                    relation=relation(k in a['fp'], k in b['fp']),
                    disappeared_reason=('candidate_absent' if sb is None else 'present_predicted_negative') if k in a['fp']-b['fp'] else '',
                    start_edge_strict=a['edge_strict'], end_edge_strict=b['edge_strict'],
                    all_three_edges_in_gt=all(x in gt_e for x in (i*n+j, i*n+v, j*n+v)),
                    hit_count=len(hit_epochs[k]), hit_epochs=';'.join(map(str, hit_epochs[k])))
                rows.append(row)
                mesh_rows.append(row)
            end_rows = [r for r in mesh_rows if r['end_fp']]
            cov = coverage(end_rows)
            per_mesh.append(dict(condition=condition, uid=uid, vertices=n,
                start_face_fp=len(a['fp']), end_face_fp=len(b['fp']),
                both_endpoint_fp=len(a['fp'] & b['fp']), start_only_fp=len(a['fp']-b['fp']), end_only_fp=len(b['fp']-a['fp']),
                end_edge_strict=b['edge_strict'], end_edge_fp=b['result']['edge']['fp'], end_edge_fn=b['result']['edge']['fn'],
                end_face_fn=b['result']['face']['fn'], end_fp_hit0=cov['hit0'], end_fp_hit1=cov['hit1'], end_fp_hit_ge2=cov['hit_ge2']))
        if number % 5 == 0:
            print(f'audited {number}/100 meshes; {len(replay)}/2500 replay hashes matched; elapsed={time.time()-started:.1f}s', flush=True)

    by_condition = {}
    for condition in conditions:
        subset = [r for r in rows if r['condition'] == condition]
        end = [r for r in subset if r['end_fp']]
        by_condition[condition] = dict(
            start_fp=coverage([r for r in subset if r['start_fp']]), end_fp=coverage(end),
            both_endpoint_fp=coverage([r for r in subset if r['start_fp'] and r['end_fp']]),
            disappeared_fp=coverage([r for r in subset if r['start_fp'] and not r['end_fp']]),
            new_fp=coverage([r for r in subset if not r['start_fp'] and r['end_fp']]),
            end_fp_edge_strict=coverage([r for r in end if r['end_edge_strict']]),
            end_fp_all_edges_gt=coverage([r for r in end if r['all_three_edges_in_gt']]),
            disappeared_reasons=dict(Counter(r['disappeared_reason'] for r in subset if r['disappeared_reason'])))
    assert by_condition['mu']['start_fp']['count'] == 1508
    assert by_condition['mu']['end_fp']['count'] == 1509
    assert by_condition['mu']['end_fp_edge_strict']['count'] == 602
    assert not torch.cuda.is_initialized()
    summary = dict(status='complete', seconds=time.time()-started, root=str(root),
        model_loaded=False, gpu_forward_calls=0, optimizer_updates=0, cuda_initialized=False,
        start_update=34220, end_update=34720, epochs=list(range(1711, 1736)),
        training_updates_verified=len(logs), negative_replays_verified=len(replay), prediction_records_verified=1200,
        data_manifest=manifest, source_checkpoint_hashes=step_hashes, conditions=by_condition,
        limitations=['Only epochs1711-1735 are audited, not earlier training history.',
          'A sampling hit proves a direct negative loss term, not its gradient magnitude or FP state at that time.',
          'FP at both endpoints does not imply FP throughout training.',
          'Missing endpoint candidate has no saved logit and is not classified-negative.',
          'Noise conditions share training and candidates; do not treat repeated candidates as independent trials.'])
    write_csv(output/'face_fp_coverage.csv', rows)
    write_csv(output/'negative_replay_verification.csv', replay)
    write_csv(output/'per_mesh_coverage.csv', per_mesh)
    verified(Path(__file__))
    for name, value in [('face_fp_coverage_summary.json', summary), ('source_hashes.json', hashes), ('missing_files.json', [])]:
        (output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete', mu=by_condition['mu'], seconds=summary['seconds']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        assert args.root is not None and args.output is not None
        main(args.root, args.output)
