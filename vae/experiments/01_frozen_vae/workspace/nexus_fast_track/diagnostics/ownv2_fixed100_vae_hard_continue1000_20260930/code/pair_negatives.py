"""Fixed parent-mu hard negatives replace uniform slots; total count is unchanged."""
import json
from pathlib import Path
import numpy as np
import torch
from data_objective import negative_faces, array_sha, seed_for, sha


def load_hard_pool(path, items, parent_sha256):
    manifest = json.loads(Path(path).read_text())
    assert manifest['parent_checkpoint_sha256'] == parent_sha256
    assert manifest['source_condition'] == 'mu'
    assert list(manifest['meshes']) == list(items)
    pools = {}
    for uid, item in items.items():
        row = manifest['meshes'][uid]
        ids = np.asarray(row['ids'], dtype=np.int64).reshape(-1, 3)
        assert row['gt_faces'] == len(item['gt_faces']) and row['vertices'] == len(item['vertices'])
        assert len(ids) <= len(item['gt_faces'])
        assert len(np.unique(ids, axis=0)) == len(ids)
        assert (np.diff(ids, axis=1) > 0).all()
        assert ((ids >= 0) & (ids < len(item['vertices']))).all()
        assert not set(map(tuple, ids)) & set(map(tuple, item['gt_faces'].numpy()))
        assert array_sha(ids) == row['ids_sha256']
        pools[uid] = torch.from_numpy(ids)
    assert sum(len(v) for v in pools.values()) == manifest['selected_total'] == 1509
    manifest['file_sha256'] = sha(path)
    return pools, manifest


def select_negatives(item, epoch, branch, pools, seed=0):
    uniform, uniform_hash = negative_faces(item, epoch, seed)
    hard = pools[item['uid']]
    assert branch in ('A_uniform', 'B_hard')
    if branch == 'A_uniform' or len(hard) == 0:
        selected = uniform
    else:
        h = hard.numpy()
        hard_set = set(map(tuple, h))
        available = np.asarray([row for row in uniform.numpy() if tuple(row) not in hard_set],
                               dtype=np.int64).reshape(-1, 3)
        wanted = len(uniform)-len(h)
        assert 0 <= wanted <= len(available)
        rng = np.random.default_rng(seed_for('hard-negative-replacement-v1', epoch, item['uid'], seed))
        chosen = available[rng.choice(len(available), size=wanted, replace=False)]
        result = np.concatenate((h, chosen), axis=0)
        result = result[np.lexsort((result[:, 2], result[:, 1], result[:, 0]))]
        assert len(result) == len(uniform) and len(np.unique(result, axis=0)) == len(result)
        assert hard_set.issubset(set(map(tuple, result)))
        selected = torch.from_numpy(np.ascontiguousarray(result))
    return selected, dict(uniform_negative_sha256=uniform_hash,
                          negative_sha256=array_sha(selected.numpy()),
                          hard_negative_count=len(hard) if branch == 'B_hard' else 0,
                          total_negative_count=len(selected))


def self_test():
    import random
    item = dict(uid='synthetic', vertices=torch.zeros((8, 3)),
                gt_faces=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64))
    pools = {'synthetic': torch.tensor([[0, 1, 3], [0, 1, 4]], dtype=torch.int64)}
    py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    for epoch in (0, 1, 1736, 1760):
        u, digest = negative_faces(item, epoch)
        a, ma = select_negatives(item, epoch, 'A_uniform', pools)
        b, mb = select_negatives(item, epoch, 'B_hard', pools)
        assert torch.equal(u, a) and ma['negative_sha256'] == digest == mb['uniform_negative_sha256']
        assert len(a) == len(b) == 3
        assert set(map(tuple, pools['synthetic'].tolist())).issubset(set(map(tuple, b.tolist())))
        assert not set(map(tuple, b.tolist())) & set(map(tuple, item['gt_faces'].tolist()))
        assert torch.equal(b, select_negatives(item, epoch, 'B_hard', pools)[0])
        empty = {'synthetic': torch.empty((0, 3), dtype=torch.int64)}
        assert torch.equal(a, select_negatives(item, epoch, 'B_hard', empty)[0])
    assert py == random.getstate() and torch.equal(th, torch.get_rng_state())
    now = np.random.get_state()
    assert np_state[0] == now[0] and np.array_equal(np_state[1], now[1]) and np_state[2:] == now[2:]
    print('pair_negatives CPU tests passed: same count, GT exclusion, deterministic replacement, RNG unchanged')


if __name__ == '__main__':
    self_test()
