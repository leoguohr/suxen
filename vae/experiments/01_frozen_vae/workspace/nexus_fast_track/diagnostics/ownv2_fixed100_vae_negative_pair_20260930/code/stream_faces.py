"""Checkpoint-bound, atomic, resumable triangle evaluation; no CUDA dependency."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def triangle_chunks(adjacency, cursor=(0, 0, 0), chunk_size=32768):
    """cursor=(vertex i, index of neighbor j, offset in common neighbors k)."""
    n = len(adjacency)
    assert adjacency.shape == (n, n) and not np.tril(adjacency).any()
    assert chunk_size > 0
    i, jpos, kpos = cursor
    buf = np.empty((chunk_size, 3), dtype=np.int32); used = 0
    while i < n:
        js = np.flatnonzero(adjacency[i])
        while jpos < len(js):
            j = int(js[jpos]); ks = np.flatnonzero(adjacency[i] & adjacency[j])
            while kpos < len(ks):
                take = min(chunk_size-used, len(ks)-kpos)
                buf[used:used+take, 0] = i; buf[used:used+take, 1] = j
                buf[used:used+take, 2] = ks[kpos:kpos+take]
                used += take; kpos += take
                if used == chunk_size:
                    yield buf.copy(), (i, jpos, kpos)
                    used = 0
            jpos += 1; kpos = 0
        i += 1; jpos = kpos = 0
    if used: yield buf[:used].copy(), (n, 0, 0)


def stream_faces(adjacency, gt_faces, score, directory, identity, stop=lambda: False,
                 chunk_size=32768, after_shard=None):
    """Only progress.json commits a shard. An orphan shard is overwritten on resume."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    progress = directory/'progress.json'; n = len(adjacency)
    gt = np.asarray(gt_faces, dtype=np.int64).reshape(-1, 3)
    keys = (gt[:, 0]*n+gt[:, 1])*n+gt[:, 2]
    covered = adjacency[gt[:, 0], gt[:, 1]] & adjacency[gt[:, 0], gt[:, 2]] & adjacency[gt[:, 1], gt[:, 2]]
    input_hash = hashlib.sha256(adjacency.tobytes()+gt.tobytes()).hexdigest()
    binding = dict(identity=identity, input_sha256=input_hash, chunk_size=chunk_size)
    if progress.exists():
        state = json.loads(progress.read_text()); assert state['binding'] == binding
        if state['shards']:
            last = directory/f"part-{state['shards']-1:08d}.npz"
            assert file_sha(last) == state['last_shard_sha256']
        if state['complete']: return state
    else:
        state = dict(binding=binding, cursor=[0, 0, 0], shards=0, candidates=0,
                     tp=0, fp=0, tn=0, complete=False)
        atomic_json(progress, state)
    for ids, cursor in triangle_chunks(adjacency, state['cursor'], chunk_size):
        if stop(): return state
        logits = np.asarray(score(ids), dtype=np.float32)
        assert logits.shape == (len(ids),) and np.isfinite(logits).all()
        ik = (ids[:, 0].astype(np.int64)*n+ids[:, 1])*n+ids[:, 2]
        labels = np.isin(ik, keys); pred = logits > 0
        path = directory/f"part-{state['shards']:08d}.npz"
        atomic_npz(path, ids=ids, logits=logits, labels=labels)
        if after_shard is not None: after_shard(path)
        state = dict(state, cursor=list(cursor), shards=state['shards']+1,
                     candidates=state['candidates']+len(ids),
                     tp=state['tp']+int((pred & labels).sum()),
                     fp=state['fp']+int((pred & ~labels).sum()),
                     tn=state['tn']+int((~pred & ~labels).sum()),
                     last_shard_sha256=file_sha(path))
        atomic_json(progress, state)
    missing = int((~covered).sum())
    state.update(complete=True, cursor=[n, 0, 0], fn=len(gt)-state['tp'],
                 fn_missing=missing, fn_present=int(covered.sum())-state['tp'])
    assert state['fn'] == state['fn_missing']+state['fn_present']
    atomic_json(progress, state)
    return state


def aggregate(meshes):
    assert meshes and all(m['complete'] for m in meshes)
    result = {}
    for task in ('edge', 'face'):
        x = {k: sum(m[task][k] for m in meshes) for k in ('tp','fp','fn','tn')}
        x['micro_f1'] = 2*x['tp']/max(2*x['tp']+x['fp']+x['fn'], 1)
        result[task] = x
        result[task+'_perfect_uids'] = [m['uid'] for m in meshes if m[task]['fp'] == m[task]['fn'] == 0]
    result['perfect_uids'] = [m['uid'] for m in meshes if all(m[t]['fp'] == m[t]['fn'] == 0 for t in ('edge','face'))]
    result['joint_perfect'] = len(result['perfect_uids'])
    result['face_fn_missing'] = sum(m['face_fn_missing'] for m in meshes)
    result['face_fn_present'] = sum(m['face_fn_present'] for m in meshes)
    return result
