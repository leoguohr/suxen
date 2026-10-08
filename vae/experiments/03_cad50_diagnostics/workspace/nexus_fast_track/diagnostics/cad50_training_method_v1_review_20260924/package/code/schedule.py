"""CAD50 data scheduling and negative sampling -- stateless, seed-derived.

Extracted from the archived `data_objective.py`. Network-agnostic.

Design property: every random decision is derived from a SHA256 of a
fixed-format recipe string, never from ambient RNG state. Consequence:
any update can be reproduced from (kind, epoch, uid) alone, which is what
makes crash-resume bit-exact.

Recipes (verbatim):
    order    : f'own512-v2-v1/order/{seed}/{epoch}/'
    negative : f'own512-v2-v1/negative/{seed}/{epoch}/{uid}'
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np
import torch

BATCH_SIZE = 5
UPDATES_PER_EPOCH = 10          # 50 meshes / 5 per batch
NEGATIVE_RATIO = 1.5            # wanted = ceil(1.5 * |gt_faces|)


def seed_for(kind: str, epoch: int, uid: str = "", seed: int = 0) -> int:
    """Derive a 64-bit seed from a recipe string. Stateless by construction."""
    text = f"own512-v2-v1/{kind}/{seed}/{epoch}/{uid}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def array_sha(array) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def epoch_batches(uids, epoch: int, seed: int = 0):
    """One epoch = a seeded permutation of all uids, cut into 10 batches of 5."""
    order = np.random.default_rng(seed_for("order", epoch, seed=seed)).permutation(len(uids))
    ordered = np.asarray(uids)[order].tolist()
    assert len(ordered) == 50 and len(set(ordered)) == 50
    return [ordered[i:i + BATCH_SIZE] for i in range(0, 50, BATCH_SIZE)]


def batch_for_update(uids, total_update: int, base_update: int = 19356, seed: int = 0):
    """Map a global update index to (epoch, batch_index, batch_uids).

    IMPORTANT: the mapping is over the GLOBAL update counter, not a local one.
    A continuation therefore starts MID-EPOCH. The archived run resumed at
    epoch=1935, batch_index=6 -- a wrong assumption about this caused a real
    crash (see evidence/TRAIN_FAILURE_attempt1.json).
    """
    local = total_update - base_update
    epoch, batch_index = divmod(total_update - 1, UPDATES_PER_EPOCH)
    return epoch, batch_index, epoch_batches(uids, epoch, seed)[batch_index], local


def negative_faces(item, epoch: int, seed: int = 0):
    """Stateless uniform non-GT triples.

    wanted = min(ceil(1.5 * |gt_faces|), C(n,3) - |gt_faces|)

    Rejection sampling: draw n-triples, sort, drop duplicates and GT.
    If `wanted` equals the entire available space, enumerate exhaustively.
    """
    n = len(item["vertices"])
    gt = {tuple(row) for row in item["gt_faces"].cpu().tolist()}
    available = math.comb(n, 3) - len(gt)
    wanted = min(math.ceil(NEGATIVE_RATIO * len(gt)), available)

    rng = np.random.default_rng(seed_for("negative", epoch, item["uid"], seed))

    if wanted == available:
        chosen = {x for x in itertools.combinations(range(n), 3) if x not in gt}
    else:
        chosen = set()
        while len(chosen) < wanted:
            draws = np.sort(rng.integers(n, size=(max(128, 4 * (wanted - len(chosen))), 3)), axis=1)
            for row in draws:
                key = tuple(int(x) for x in row)
                if key[0] < key[1] < key[2] and key not in gt:
                    chosen.add(key)
                    if len(chosen) == wanted:
                        break

    result = np.asarray(sorted(chosen), dtype=np.int64).reshape(-1, 3)
    assert len(result) == wanted
    return torch.from_numpy(result), array_sha(result)


def merge_negatives(random_negatives, fixed_hard_negatives, device=None):
    """Negative set = unique(random union hard), sorted.

    `fixed_hard_negatives` are the parent model's face false positives
    (47 of them in the archived run) and are NOT refreshed during training.
    """
    merged = torch.unique(
        torch.cat((random_negatives, fixed_hard_negatives)), sorted=True, dim=0
    )
    if device is not None:
        merged = merged.to(device)
    return merged
