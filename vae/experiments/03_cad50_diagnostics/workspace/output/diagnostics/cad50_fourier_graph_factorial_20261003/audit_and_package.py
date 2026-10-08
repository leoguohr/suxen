"""Read existing four-cell evidence on CPU; do not run model forwards or updates."""
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'cad50_graph_activation_ablation_20261003'
OUT = ROOT / 'delivery_20261005'
DATA = Path('/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921/data')
CELLS = {'Fourier_LN_post': OLD/'runs/V2_control', 'Fourier_LN_pre': OLD/'runs/Graph_LN_pre',
         'XYZ_LN_post': ROOT/'runs/XYZ_LN_post', 'XYZ_LN_pre': ROOT/'runs/XYZ_LN_pre'}
STEPS = (0, 500, 1000, 1500, 2000)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def counts(predicted, gt):
    return dict(tp=int((predicted & gt).sum()), fp=int((predicted & ~gt).sum()),
                fn=int((~predicted & gt).sum()))


def main():
    OUT.mkdir(exist_ok=True)
    manifest = read(DATA/'manifest.json')
    assert sha(DATA/'manifest.json') == '4742e72bde899b88633cb70603a256d98b83b7bd85cd301b7a090938081a2ac6'
    uid_order = manifest['uids']
    assert uid_order == [f'teacher_cad50_{i:02d}' for i in range(50)]
    meshes = {}
    for record in manifest['meshes']:
        path = DATA/record['path']
        assert sha(path) == record['sha256']
        with np.load(path, allow_pickle=False) as q:
            meshes[record['uid']] = dict(n=len(q['vertices']), gt=np.unique(np.sort(q['faces'], axis=1), axis=0))
    verification, trajectories, flat, schedules, configs = {}, {}, [], {}, {}
    for label, folder in CELLS.items():
        complete, cfg = read(folder/'complete.json'), read(folder/'config.json')
        assert complete['completed_updates'] == cfg['max_updates'] == 2000
        logs = [json.loads(line) for line in (folder/'updates.jsonl').read_text().splitlines()]
        assert [r['update'] for r in logs] == list(range(1, 2001))
        seen = collections.Counter()
        schedule = []
        for i, row in enumerate(logs):
            assert row['epoch'] == i//10 and row['batch_index'] == i%10
            assert len(row['uids']) == len(set(row['uids'])) == 5
            assert row['adam_step'] == i+1 and row['participations'] == (i+1)*5
            assert row['lr'] == 1e-4*min((i+1)/100, 1)
            assert all(math.isfinite(row[k]) for k in ['loss_before', 'gradient_norm_before_clip', 'clip_coefficient'])
            assert row['gradient_norm_before_clip'] > 0
            assert [m['uid'] for m in row['meshes']] == row['uids']
            seen.update(row['uids'])
            schedule.append((row['epoch'], row['batch_index'], row['uids'], [m['negative_sha256'] for m in row['meshes']]))
        assert set(seen) == set(uid_order) and set(seen.values()) == {200}
        for epoch in range(200):
            assert sorted(itertools.chain.from_iterable(x['uids'] for x in logs[10*epoch:10*(epoch+1)])) == uid_order
        schedules[label], configs[label] = schedule, cfg
        trajectories[label] = []
        for step in STEPS:
            path = folder/'evaluations'/f'step-{step:05d}'
            e = read(path/'evaluation.json')
            assert e['complete'] and [r['uid'] for r in e['meshes']] == uid_order
            strict = []
            for row in e['meshes']:
                assert row['complete'] and row['checkpoint_sha256'] == e['checkpoint']['sha256']
                assert read(path/row['uid']/'metrics.json') == row
                assert row['face']['fn'] == row['face_fn_missing']+row['face_fn_present']
                assert row['joint_strict'] == all(row[t][k] == 0 for t in ['edge','face'] for k in ['fp','fn'])
                if row['joint_strict']:
                    strict.append(row['uid'])
                flat.append(dict(cell=label, step=step, uid=row['uid'], vertices=row['vertices'],
                    edge_tp=row['edge']['tp'], edge_fp=row['edge']['fp'], edge_fn=row['edge']['fn'],
                    face_tp=row['face']['tp'], face_fp=row['face']['fp'], face_fn=row['face']['fn'],
                    face_fn_missing=row['face_fn_missing'], face_fn_present=row['face_fn_present'],
                    joint_strict=row['joint_strict']))
            assert e['joint_strict_uids'] == strict and e['joint_strict'] == len(strict)
            for task in ['edge','face']:
                actual = {k:sum(r[task][k] for r in e['meshes']) for k in ['tp','fp','fn']}
                actual['micro_f1'] = 2*actual['tp']/max(2*actual['tp']+actual['fp']+actual['fn'],1)
                assert actual == e['counts'][task]
            trajectories[label].append(dict(step=step, counts=e['counts'], strict=e['joint_strict'],
                strict_uids=strict, large16=e['large16'], checkpoint=e['checkpoint']))
        # Independently count all saved final logits and verify complete actual triangles.
        final = e
        for row in final['meshes']:
            uid, n = row['uid'], row['vertices']
            path = folder/'evaluations/step-02000'/uid
            with np.load(path/'edge.npz', allow_pickle=False) as q:
                pairs, logits, truth = q['pair_ids'], q['logits'], q['gt']
                assert np.array_equal(pairs, np.column_stack(np.triu_indices(n,1)))
                assert np.isfinite(logits).all() and counts(logits>0, truth) == row['edge']
                adjacency = np.zeros((n,n), dtype=bool)
                chosen = pairs[logits>0]
                adjacency[chosen[:,0],chosen[:,1]] = True
            expected = [(i,int(j),int(k)) for i in range(n) for j in np.flatnonzero(adjacency[i])
                        for k in np.flatnonzero(adjacency[i]&adjacency[j])]
            expected = np.asarray(expected,dtype=np.int64).reshape(-1,3)
            saved, tp, fp = [], 0, 0
            gt = meshes[uid]['gt']; gt_keys = (gt[:,0]*n+gt[:,1])*n+gt[:,2]
            shards = sorted(path.glob('face-*.npz'))
            assert len(shards) == row['face_shards']
            for shard in shards:
                with np.load(shard, allow_pickle=False) as q:
                    ids, logits, truth = q['ids'], q['logits'], q['gt']
                    assert np.isfinite(logits).all()
                    assert np.array_equal(truth,np.isin((ids[:,0]*n+ids[:,1])*n+ids[:,2],gt_keys))
                    c = counts(logits>0,truth); tp += c['tp']; fp += c['fp']; saved.append(ids)
            actual = np.concatenate(saved) if saved else np.empty((0,3), dtype=np.int64)
            assert np.array_equal(actual,expected) and len(actual) == row['actual_face_candidates']
            assert dict(tp=tp,fp=fp,fn=len(gt)-tp) == row['face']
            with np.load(path/'gt_face.npz',allow_pickle=False) as q:
                assert np.array_equal(q['ids'],gt)
                covered = adjacency[gt[:,0],gt[:,1]] & adjacency[gt[:,0],gt[:,2]] & adjacency[gt[:,1],gt[:,2]]
                assert np.array_equal(q['covered'],covered)
                assert int((~covered).sum()) == row['face_fn_missing']
                assert int((covered & (q['logits']<=0)).sum()) == row['face_fn_present']
        cpinfo = final['checkpoint']; cppath = Path(cpinfo['path'])
        assert cppath.stat().st_size == cpinfo['bytes'] and sha(cppath) == cpinfo['sha256']
        cp = torch.load(cppath,map_location='cpu',mmap=True,weights_only=False)
        assert cp['completed_updates'] == 2000 and set(cp['participation'].values()) == {200}
        assert cp['next_epoch'] == 200 and cp['next_batch'] == 0 and 'rng' in cp
        adam_steps = {int(state['step']) for state in cp['optimizer']['state'].values()}
        assert adam_steps == {2000}
        verification[label] = dict(updates=2000, every_uid_participations=200,
            all_five_evaluations_complete=True, final_prediction_counts_reproduced=True,
            final_triangles_complete=True, final_checkpoint_hash_verified=cpinfo,
            adam_state_entries=len(cp['optimizer']['state']), adam_steps=sorted(adam_steps),
            checkpoint_cursor=[cp['next_epoch'],cp['next_batch']], rng_keys=list(cp['rng']))
        del cp
        print('VERIFIED',label, final['joint_strict'],final['counts'],flush=True)
    assert all(s==schedules['Fourier_LN_post'] for s in schedules.values())
    keys = ['data','initial_tensor_hash','lr','betas','eps','weight_decay','warmup_updates','clip',
            'sampling','KL','logvar_frozen','dropout','microbatch','meshes_per_update','objective']
    assert all(all(c[k] == configs['Fourier_LN_post'][k] for k in keys) for c in configs.values())
    write(OUT/'VERIFICATION.json',dict(cells=verification, all_UID_and_negative_schedules_equal=True,
        common_training_config_equal=True, gpu_forward_or_optimizer_updates=0,
        scope='CPU inspection of existing logs, full final prediction arrays and complete resumable checkpoints'))
    write(OUT/'TRAJECTORIES.json',trajectories)
    with (OUT/'PER_MESH.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    # Package original evidence as it exists; historical provisional metadata stays unchanged.
    sources = {}
    for prefix, base in [('factorial',ROOT),('fourier_controls',OLD)]:
        for path in base.iterdir():
            if path.is_file() and path.suffix in ['.py','.json','.md','.diff','.log']:
                sources[f'{prefix}/{path.name}'] = path
    for base, prefix in [(ROOT/'performance','factorial/performance'),(ROOT/'reference','factorial/reference')]:
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in ['.json','.md']:
                sources[f'{prefix}/{path.relative_to(base)}'] = path
    for label, folder in CELLS.items():
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ['.py','.json','.jsonl','.npz','.log'] and 'cold_final' not in path.parts:
                sources[f'cells/{label}/{path.relative_to(folder)}'] = path
    for path in DATA.rglob('*'):
        if path.is_file() and path.suffix in ['.npz','.json','.py','.md']:
            sources[f'data/{path.relative_to(DATA)}'] = path
    for path in OUT.iterdir():
        if path.is_file() and path.suffix in ['.json','.csv','.md','.png'] and path.name != 'SHA256SUMS.json':
            sources[f'analysis/{path.name}'] = path
    write(OUT/'EXCLUDED_WEIGHTS.json',{k:v['final_checkpoint_hash_verified'] for k,v in verification.items()})
    sources['analysis/EXCLUDED_WEIGHTS.json'] = OUT/'EXCLUDED_WEIGHTS.json'
    write(OUT/'SHA256SUMS.json',{name:sha(path) for name,path in sources.items()})
    sources['analysis/SHA256SUMS.json'] = OUT/'SHA256SUMS.json'
    archive=ROOT/'cad50_fourier_graph_factorial_results_20261005.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,path in sorted(sources.items()):
            z.write(path,name)
    assert archive.stat().st_size < 900*1024*1024
    delivery=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(sources))
    write(OUT/'DELIVERY.json',delivery);print('DELIVERY',json.dumps(delivery),flush=True)


if __name__ == '__main__':
    main()
