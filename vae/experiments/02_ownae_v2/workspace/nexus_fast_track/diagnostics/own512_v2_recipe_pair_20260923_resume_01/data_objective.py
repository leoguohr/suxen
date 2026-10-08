"""Read-only CAD50 data, paired stateless sampling and differentiable Hard4."""
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
DATA_SHA = '4742e72bde899b88633cb70603a256d98b83b7bd85cd301b7a090938081a2ac6'
POOL_SHA = '86e7895f7ce44f7740d96a359570a3617c8967414c69aaf38dd70c02a177c61b'
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


def load_dataset(data_root, pool_root):
    data_root, pool_root = Path(data_root), Path(pool_root)
    assert sha(data_root/'manifest.json') == DATA_SHA
    pool_manifest_path = pool_root.resolve().parent/'pool_manifest.json'
    assert sha(pool_manifest_path) == POOL_SHA
    source = json.loads((data_root/'manifest.json').read_text())
    pool_manifest = json.loads(pool_manifest_path.read_text())
    assert source['uids'] == [f'teacher_cad50_{i:02d}' for i in range(50)]
    items, totals, records = {}, dict(meshes=50, vertices=0, edges=0, faces=0, pairs=0), []
    for row, pool_row in zip(source['meshes'], pool_manifest['records']):
        uid = row['uid']; assert uid == pool_row['uid']
        path, pool_path = data_root/row['path'], pool_root/(uid+'.npz')
        assert sha(path) == row['sha256'] and sha(pool_path) == pool_row['sha256']
        with np.load(path, allow_pickle=False) as q, np.load(pool_path, allow_pickle=False) as p:
            v, f = q['vertices'].copy(), q['faces'].copy()
            gt = np.unique(np.sort(f, axis=1), axis=0)
            edges = np.unique(np.sort(q['edge_index'].T, axis=1), axis=0)
            assert v.dtype == np.float32 and f.dtype == np.int64
            original_ids = q['original_vertex_indices'].copy()
            # Source-cache local order is authoritative; this is its preserved
            # map back to the earlier original numbering, not an identity map.
            assert original_ids.shape == (len(v),) and len(np.unique(original_ids)) == len(v)
            assert np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v)
            assert len(gt) == len(f) and (np.diff(gt, axis=1) > 0).all()
            assert np.array_equal(v, p['vertices'])
            assert np.array_equal(gt, np.unique(np.sort(p['positive'], axis=1), axis=0))
            assert np.array_equal(edges, np.unique(np.sort(p['edges'].T, axis=1), axis=0))
        derived = np.unique(np.sort(np.concatenate((f[:,[0,1]], f[:,[0,2]], f[:,[1,2]])), axis=1), axis=0)
        assert np.array_equal(edges, derived)
        pairs = torch.triu_indices(len(v), len(v), offset=1).T.contiguous()
        labels = np.isin(pairs.numpy()@np.array([len(v),1]), edges@np.array([len(v),1]))
        items[uid] = dict(uid=uid, vertices=torch.from_numpy(v), faces=torch.from_numpy(f),
            gt_faces=torch.from_numpy(gt), edges=torch.from_numpy(edges), pairs=pairs,
            edge_labels=torch.from_numpy(labels), original_vertex_indices=torch.from_numpy(original_ids))
        for name, value in [('vertices',len(v)),('edges',len(edges)),('faces',len(gt)),('pairs',len(pairs))]:
            totals[name] += value
        records.append(dict(uid=uid, data_sha256=row['sha256'], reference_pool_sha256=pool_row['sha256'],
            vertices=len(v), edges=len(edges), faces=len(gt), pairs=len(pairs),
            original_vertex_indices_sha256=array_sha(original_ids)))
    assert totals == dict(meshes=50, vertices=2872, edges=8364, faces=5576, pairs=235741)
    return items, dict(data_manifest_sha256=DATA_SHA, reference_pool_manifest_sha256=POOL_SHA,
        source=str(data_root.resolve()), totals=totals, uids=source['uids'], records=records,
        training_negatives='fresh stateless uniform non-GT triples; old pool negatives not used')


def seed_for(kind, epoch, uid='', seed=0):
    text = f'own512-v2-v1/{kind}/{seed}/{epoch}/{uid}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')


def epoch_batches(uids, epoch, seed=0):
    ordered = np.asarray(uids)[np.random.default_rng(seed_for('order', epoch, seed=seed)).permutation(len(uids))].tolist()
    assert len(ordered) == 50 and len(set(ordered)) == 50
    return [ordered[i:i+5] for i in range(0, 50, 5)]


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
