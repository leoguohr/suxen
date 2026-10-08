"""Original fixed100 arrays, stateless sampling and unchanged Hard4."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

EDGE_SCALE = 0.9306077080970389
FACE_SCALE = 0.39804385828730726
FACE_FACTOR = 0.25
GROUPS = ('tp', 'tn', 'fp', 'fn')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


SELECTION_SHA = '6a88ce78eee5caa00e27e4d7d8d9247132e6725a44697bbb16180ca38fee14c6'
MANIFEST_SHA = 'b90e14f5d318dc0d32ad806c631af7f66be28aa644ff3caf357d34ce3e208b93'


def load_dataset(source):
    """Read original arrays without reordering; pairs materialize one mesh at a time."""
    import csv
    source = Path(source)
    assert sha(source/'selection.json') == SELECTION_SHA
    assert sha(source/'overfit100_manifest.csv') == MANIFEST_SHA
    selected = json.loads((source/'selection.json').read_text())['uids']
    records = list(csv.DictReader((source/'overfit100_manifest.csv').open()))
    assert [r['uid'] for r in records] == selected
    assert len(selected) == len(set(selected)) == 100
    items, totals = {}, dict(meshes=100, vertices=0, edges=0, faces=0, pairs=0)
    for r in records:
        for kind in ('mesh', 'topology'):
            assert sha(r[kind+'_path']) == r[kind+'_sha256'], (r['uid'], kind)
        with np.load(r['mesh_path'], allow_pickle=False) as a, np.load(r['topology_path'], allow_pickle=False) as b:
            v, f = a['vertices_norm'].copy(), a['faces'].copy()
            gt = np.unique(np.sort(f, axis=1), axis=0)
            edges = np.unique(np.sort(b['edge_index'].T, axis=1), axis=0)
            assert v.dtype == np.float32 and f.dtype == np.int64 and np.isfinite(v).all()
            assert f.min() >= 0 and f.max() < len(v) and len(gt) == len(f)
            assert (np.diff(gt, axis=1) > 0).all()
            assert np.array_equal(gt, np.unique(np.sort(b['face_set'], axis=1), axis=0))
        derived = np.unique(np.sort(np.concatenate((f[:,[0,1]], f[:,[0,2]], f[:,[1,2]])), axis=1), axis=0)
        assert np.array_equal(edges, derived)
        counts = dict(vertices=len(v), edges=len(edges), faces=len(gt), pairs=len(v)*(len(v)-1)//2)
        assert counts == dict(vertices=int(r['vertices']), edges=int(r['gt_edges']), faces=int(r['gt_faces']), pairs=int(r['edge_pairs']))
        for k, value in counts.items(): totals[k] += value
        items[r['uid']] = dict(uid=r['uid'], vertices=torch.from_numpy(v), faces=torch.from_numpy(f),
            gt_faces=torch.from_numpy(gt), edges=torch.from_numpy(edges))
    assert totals == dict(meshes=100, vertices=106325, edges=309194, faces=204330, pairs=84669234)
    return items, dict(uids=selected, totals=totals, selection_sha256=SELECTION_SHA,
        manifest_sha256=MANIFEST_SHA, source=str(source),
        records=[{k:r[k] for k in ('uid','mesh_path','mesh_sha256','topology_path','topology_sha256')} for r in records])


def materialize_item(item):
    n = len(item['vertices'])
    pairs = torch.triu_indices(n, n, 1).T.contiguous()
    keys = item['edges'].numpy() @ np.array([n, 1])
    labels = np.isin(pairs.numpy() @ np.array([n, 1]), keys)
    return dict(item, pairs=pairs, edge_labels=torch.from_numpy(labels))


def seed_for(kind, epoch, uid='', seed=0):
    text = f'own512-v2-v1/{kind}/{seed}/{epoch}/{uid}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def epoch_batches(uids, epoch, seed=0):
    ordered = np.asarray(uids)[np.random.default_rng(seed_for('order', epoch, seed=seed)).permutation(len(uids))].tolist()
    assert len(ordered) == 100 and len(set(ordered)) == 100
    return [ordered[i:i+5] for i in range(0, 100, 5)]


def negative_faces(item, epoch, seed=0):
    n = len(item['vertices'])
    gt = {tuple(row) for row in item['gt_faces'].cpu().tolist()}
    available = math.comb(n, 3)-len(gt)
    wanted = min(math.ceil(1.5*len(gt)), available)
    rng = np.random.default_rng(seed_for('negative', epoch, item['uid'], seed))
    if wanted == available:
        chosen = {x for x in itertools.combinations(range(n),3) if x not in gt}
    else:
        chosen = set()
        while len(chosen) < wanted:
            draws = np.sort(rng.integers(n, size=(max(128,4*(wanted-len(chosen))),3)), axis=1)
            for row in draws:
                key = tuple(int(x) for x in row)
                if key[0] < key[1] < key[2] and key not in gt:
                    chosen.add(key)
                    if len(chosen) == wanted:
                        break
    result = np.asarray(sorted(chosen), dtype=np.int64).reshape(-1,3)
    assert len(result) == wanted
    return torch.from_numpy(result), array_sha(result)


def edge_logits(embedding, pairs):
    delta = embedding[pairs[:,0]]-embedding[pairs[:,1]]
    space, time = delta.chunk(2, dim=-1)
    return (space.square().sum(-1)-time.square().sum(-1))*EDGE_SCALE


def face_logits(embedding, triples):
    first, second, third = [embedding[triples[:,i]] for i in range(3)]
    areas = []
    for a,b,c in zip(first.chunk(2,-1), second.chunk(2,-1), third.chunk(2,-1)):
        u, v = b-a, c-a
        areas.append((u.square().sum(-1)*v.square().sum(-1)-(u*v).sum(-1).square()).clamp_min(0))
    return FACE_SCALE*(FACE_FACTOR*(areas[0]-areas[1]))


def hard4_sums(logits, labels):
    positive, truth = logits.detach()>0, labels == 1
    masks = (truth & positive, ~truth & ~positive, ~truth & positive, truth & ~positive)
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels.to(torch.float32), reduction='none')
    return torch.stack([bce[m].sum(dtype=torch.float32) for m in masks]), torch.stack([m.sum() for m in masks])


def hard4_chunks(logits_fn, embedding, ids, labels, chunk):
    numerator = embedding.new_zeros(4)
    counts = torch.zeros(4, dtype=torch.long, device=embedding.device)
    for start in range(0,len(ids),chunk):
        logits = logits_fn(embedding, ids[start:start+chunk])
        n,c = hard4_sums(logits, labels[start:start+chunk])
        numerator, counts = numerator+n, counts+c
    loss = (numerator/counts.clamp_min(1)).sum()/4
    return loss, dict(counts=dict(zip(GROUPS, counts.detach().cpu().tolist())),
        group_bce_sums=numerator.detach().cpu().tolist(), loss=float(loss.detach()))


def objective(outputs, item, negatives, pair_chunk=32768, face_chunk=32768):
    device = outputs['edge'].device
    le, es = hard4_chunks(edge_logits, outputs['edge'], item['pairs'].to(device),
        item['edge_labels'].to(device), pair_chunk)
    positive = item['gt_faces'].to(device)
    triples = torch.cat((positive, negatives.to(device)))
    labels = torch.cat((torch.ones(len(positive), device=device),torch.zeros(len(negatives),device=device)))
    lf, fs = hard4_chunks(face_logits, outputs['face'], triples, labels, face_chunk)
    return le+lf, dict(edge=es, face=fs, edge_loss=float(le.detach()), face_loss=float(lf.detach()),
        face_positives=len(positive), face_negatives=len(negatives))


def resume_cursor(uids, completed, seed=0):
    assert completed >= 0
    epoch, group = divmod(completed, 20)
    order = [u for batch in epoch_batches(uids, epoch, seed) for u in batch]
    return dict(completed_updates=completed, epoch=epoch, group=group, next_mesh_position=5*group,
        order=order, order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),
        next_uids=order[5*group:5*group+5], seed=seed)
