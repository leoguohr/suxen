"""CPU checks with a tiny S0-shaped model. Run before any GPU job:

    CUDA_VISIBLE_DEVICES= python tests/test_cpu.py

Checks: data integrity (SHA256), packed forward == original VertexDiT.forward per item,
per-item loss == original per-object velocity MSE, the detached-condition chunked
backward gives the same gradients as one plain backward, object scheduling covers
every object once per epoch, loss weights sum to 1, EMA swap/save round-trips,
and generation + scoring (perfect GT replay scores exact; random model runs).
"""
from __future__ import annotations

import os
# nexus-algo pairs protobuf 4.x with system onnx 1.16; torch.optim lazily imports onnx.
# S0 ran with the same setting. Must be set before torch is imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402
from common import (build_octree_levels, cache_fixed_features, construct_model, expand_occupied_children,  # noqa: E402
                    generate_tree, load_data, score_tree, summarize)
import dense_train  # noqa: E402
from packed import chunk_loss, make_chunks, pack_rows, packed_flow_forward  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)
DEVICE = torch.device("cpu")
# The nexus-algo torch build returns NaN from its CPU fused attention for some shapes
# (found by Codex on 2026-10-09). Tests run on CPU only, so force the math kernel here.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class RecordingSGD(torch.optim.SGD):
    """lr=0 optimizer that records gradients at step() (train_update frees them afterwards)."""
    def __init__(self, params):
        super().__init__(params, lr=0.0)
        self.recorded = None

    def step(self, closure=None):
        self.recorded = [None if q.grad is None else q.grad.detach().clone()
                         for group in self.param_groups for q in group["params"]]
        return super().step(closure)


def tiny_model():
    model = construct_model("S0", small=True).float()
    with torch.no_grad():  # AdaLN and the head are zero-initialised; make the test non-trivial
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def make_args(**overrides):
    args = argparse.Namespace(objects_per_update=3, coarse_depths=5, coarse_copies=3, row_tokens=1024,
                              chunk_tokens=2048, checkpoint_activations=1, clip=1e9, lr=0.0, warmup=1,
                              seed=7, loss_weighting="item", decay_fraction=0.0, min_lr=0.0,
                              max_updates=10**9)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    started = time.time()
    manifest, conditions, leaves, levels, raw_gt = load_data(DEVICE)
    check("data: 50 objects, SHA256 verified", len(conditions) == 50)
    sizes = [sum(len(c) for c, _ in lv) for lv in levels]
    model = tiny_model()
    cache_fixed_features(model, conditions)

    # 1) packed forward == original forward, item by item
    args = make_args()
    small = sorted(range(50), key=lambda i: sizes[i])[:2] + [sorted(range(50), key=lambda i: sizes[i])[20]]
    items = dense_train.build_items(args, small, levels, update=1, device=DEVICE)
    contexts = torch.cat([model.condition_encoder(conditions[o]) for o in small])
    rows = pack_rows(items, args.row_tokens)
    chunks = make_chunks(items, rows, args.chunk_tokens, DEVICE)
    worst, worst_loss, seen = 0.0, 0.0, 0
    with torch.no_grad():
        for chunk in chunks:
            pred = packed_flow_forward(model.flow, chunk.noisy, chunk.codes, chunk.depths, chunk.item_ids,
                                       chunk.item_times, contexts[chunk.row_object])
            _, per_item = chunk_loss(pred, chunk)
            for local, index in enumerate(chunk.item_index):
                item = items[index]
                noisy = (1 - item.time) * item.noise + item.time * item.target
                ref = model.flow(noisy[None], torch.tensor([item.time]), item.codes[None],
                                 torch.tensor([item.depth]), contexts[item.obj:item.obj + 1])[0]
                got = pred[chunk.item_ids == local]
                worst = max(worst, float((got - ref).abs().max()))
                ref_loss = float(((ref - (item.target - item.noise)) ** 2).sum() / (len(item.codes) * 8))
                worst_loss = max(worst_loss, abs(ref_loss - float(per_item[local])) / max(ref_loss, 1e-12))
                seen += 1
    check("packed forward == original forward", seen == len(items) and worst < 1e-4,
          f"items={seen} max|diff|={worst:.2e}")
    check("per-item loss == original per-object velocity MSE", worst_loss < 1e-4, f"max rel diff={worst_loss:.2e}")
    total_weight = sum(i.weight for i in items)
    check("loss weights sum to 1", abs(total_weight - 1) < 1e-9, f"{total_weight}")
    depth_weight = {}
    for i in items:
        depth_weight[(i.obj, i.depth)] = depth_weight.get((i.obj, i.depth), 0) + i.weight
    check("each (object, depth) has equal weight", max(depth_weight.values()) - min(depth_weight.values()) < 1e-12)

    # 2) detached-condition chunked backward == plain backward, for both loss weightings
    for weighting, ckpt in (("item", 1), ("item", 0), ("token", 1)):
        args = make_args(checkpoint_activations=ckpt, row_tokens=4096, chunk_tokens=8192, loss_weighting=weighting)
        update = 2
        objects = dense_train.objects_for_update(args, update, 50)
        items = dense_train.build_items(args, objects, levels, update, DEVICE)
        chunks = make_chunks(items, pack_rows(items, args.row_tokens), args.chunk_tokens, DEVICE)
        denominator = 8.0 * sum(len(i.codes) for i in items) if weighting == "token" else None
        model.zero_grad(set_to_none=True)
        batch = dense_train.cached_condition_batch(model, conditions, objects)
        condition = model.condition_encoder(batch)
        reference, squared, count = 0, 0.0, 0
        for chunk in chunks:
            pred = packed_flow_forward(model.flow, chunk.noisy, chunk.codes, chunk.depths, chunk.item_ids,
                                       chunk.item_times, condition[chunk.row_object])
            reference = reference + chunk_loss(pred, chunk, denominator)[0]
            valid = chunk.item_ids >= 0
            squared += float(((pred - chunk.velocity) ** 2).sum(-1)[valid].sum())
            count += int(valid.sum()) * 8
        if weighting == "token":
            check("token loss == plain mean over all tokens", abs(float(reference) - squared / count) < 1e-6 * max(1, squared / count),
                  f"{float(reference):.6e} vs {squared / count:.6e}")
        reference.backward()
        grad = lambda q: q.grad if q.grad is not None else torch.zeros_like(q)  # noqa: E731
        expected = [grad(q).clone() for q in model.parameters()]
        optimizer = RecordingSGD(model.parameters())
        row = dense_train.train_update(model, optimizer, args, conditions, levels, update, DEVICE)
        got = [g if g is not None else torch.zeros_like(e) for g, e in zip(optimizer.recorded, expected)]
        worst = max(float((g - e).abs().max() / (e.abs().max() + 1e-12)) for g, e in zip(got, expected))
        check(f"chunked backward == plain backward ({weighting}, checkpoint={ckpt})",
              worst < 1e-4 and abs(row["loss"] - float(reference)) < 1e-5 * max(1, float(reference)),
              f"max rel grad diff={worst:.2e}, chunks={row['chunks']}")
        vecset = [g for g, (n, _) in zip(got, model.named_parameters()) if n.startswith("condition_encoder.")]
        check("VecSet receives gradient", sum(float(g.abs().sum()) for g in vecset) > 0)
        check("gradients freed after the update", all(q.grad is None for q in model.parameters()))

    # 2b) learning-rate schedule: warmup, constant, linear decay over the last 20%
    sched = make_args(lr=1e-5, warmup=20, max_updates=100, decay_fraction=0.2, min_lr=0.0)
    values = [dense_train.lr_at(sched, u) for u in (10, 20, 50, 80, 90, 100)]
    expected_lr = [5e-6, 1e-5, 1e-5, 1e-5, 5e-6, 0.0]
    check("LR schedule", all(abs(a - b) < 1e-12 for a, b in zip(values, expected_lr)), f"{values}")

    # 3) object schedule covers each object once per epoch
    args = make_args(objects_per_update=8)
    picked = [o for u in range(1, 8) for o in dense_train.objects_for_update(args, u, 50)][:50]
    check("schedule: every object once per epoch", sorted(picked) == list(range(50)))

    # 4) EMA swap and save round-trip
    ema = dense_train.EMA(model, 0.9)
    before = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update()
    dense_train.ema_swap(ema)
    swapped = [p.detach().clone() for p in model.parameters()]
    dense_train.ema_swap(ema)
    restored_ok = all(torch.equal(p, b + 1.0) for p, b in zip(model.parameters(), before))
    ema_ok = all(torch.allclose(s, b + 0.1, atol=1e-6) for s, b in zip(swapped, before))
    check("EMA update and in-place swap", restored_ok and ema_ok)
    state = dense_train.ema_state(model, ema)
    fresh = construct_model("S0", small=True)
    fresh.load_state_dict(state, strict=True)
    check("EMA state_dict loads strictly", True)

    # 5) scoring: a perfect replay of GT scores exact; a random model generates and scores
    obj = sorted(range(50), key=lambda i: sizes[i])[3]
    cells = torch.from_numpy(leaves[obj])
    fake_levels, parents = [], torch.zeros((1, 3), dtype=torch.long)
    for lv in build_octree_levels(cells, 9):
        lookup = {tuple(c): r for r, c in enumerate(lv.parent_codes.tolist())}
        occupancy = torch.stack([lv.target[lookup[tuple(c)]] for c in parents.tolist()]) >= 0.5
        predicted = expand_occupied_children(parents, occupancy)
        fake_levels.append({"depth": lv.depth, "parents": parents.numpy(), "estimate": occupancy.float()[None].numpy(),
                            "occupancy": occupancy.numpy(), "predicted_cells": predicted.numpy()})
        parents = predicted
    scored = score_tree("complete", fake_levels, parents.numpy(), leaves[obj], raw_gt[obj])
    ratio = scored["d9_gt_xyz"]["spacing_ratio"]
    check("perfect GT replay scores exact at every depth", scored["all_levels_exact"] and scored["count_correct"]
          and ratio is not None and ratio < 1e-6, f"ratio={ratio}")
    status, gen_levels, q = generate_tree(model, model.condition_encoder(conditions[obj]), 97029000, steps=2,
                                          max_depth=4)
    report = summarize([score_tree(status, gen_levels, q, leaves[obj], raw_gt[obj]), scored])
    check("random-model generation + summarize run", report["trees"] == 2, f"status={status}")
    print(f"ALL CPU TESTS PASSED in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
