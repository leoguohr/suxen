"""Dense 50-object Vertex overfit training, warm-started from S0@34000. One GPU per run.

Same model (S0), same objective (rectified-flow velocity MSE, per-item mean, uniform t),
same data. What changes is how much signal each optimizer update carries:
  * every update takes K objects and ALL nine depths of each (VecSet runs once per object);
  * depths 1..C (cheap, and where 84/93 of S0's failures start) get R stratified-t copies;
  * a fresh AdamW (wd 0) with linear warmup, an EMA of the weights, a wall-clock deadline;
  * a cheap coarse probe (root -> depth 5, all 50 objects, own parents) every few minutes.
Each (object, depth) has equal total weight in the loss, as in the original protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import time
import traceback

import torch

from common import (HERE, S0_SHA256, append_jsonl, build_model, cache_fixed_features, coarse_probe,
                    load_data, load_weights, sha, write_json)
from packed import Item, chunk_loss, make_chunks, pack_rows, packed_flow_forward
from architecture_variants import cached_condition_batch


def seeded(key: str, device) -> torch.Generator:
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return torch.Generator(device=device).manual_seed(seed)


def object_order(seed: int, count: int, epoch: int) -> list[int]:
    g = torch.Generator().manual_seed(seed * 1000003 + epoch)
    return torch.randperm(count, generator=g).tolist()


def objects_for_update(args, update: int, count: int) -> list[int]:
    """Cycle through shuffled epochs of all objects, K at a time (deterministic, resumable)."""
    start = (update - 1) * args.objects_per_update
    picked = []
    for position in range(start, start + args.objects_per_update):
        epoch, offset = divmod(position, count)
        picked.append(object_order(args.seed, count, epoch)[offset])
    return picked


def build_items(args, objects, levels, update, device) -> list[Item]:
    g = seeded(f"dense|{args.seed}|{update}", device)
    items = []
    unit_weight = 1.0 / (len(objects) * len(levels[0]))
    for k, obj in enumerate(objects):
        for depth0, (codes, target) in enumerate(levels[obj]):
            depth = depth0 + 1
            copies = args.coarse_copies if depth <= args.coarse_depths else 1
            u = torch.rand((copies,), generator=g, device=device)
            times = ((torch.arange(copies, device=device) + u) / copies).tolist()
            for t in times:
                noise = torch.randn(target.shape, generator=g, device=device)
                items.append(Item(obj=k, depth=depth, codes=codes, target=target, time=float(t),
                                  noise=noise, weight=unit_weight / copies))
    return items


def train_update(model, optimizer, args, conditions, levels, update, device):
    objects = objects_for_update(args, update, len(conditions))
    items = build_items(args, objects, levels, update, device)
    rows = pack_rows(items, args.row_tokens)
    chunks = make_chunks(items, rows, args.chunk_tokens, device)
    optimizer.zero_grad(set_to_none=True)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")
    with autocast:
        batch = cached_condition_batch(model, conditions, objects)
        condition_full = model.condition_encoder(batch)
    condition_leaf = condition_full.detach().float().requires_grad_(True)
    total = 0.0
    depth_sum = torch.zeros(9, device=device)
    depth_n = torch.zeros(9, device=device)
    tokens = 0
    for chunk in chunks:
        with autocast:
            prediction = packed_flow_forward(model.flow, chunk.noisy, chunk.codes, chunk.depths,
                                             chunk.item_ids, chunk.item_times,
                                             condition_leaf[chunk.row_object],
                                             use_checkpoint=args.checkpoint_activations)
        loss, per_item = chunk_loss(prediction, chunk)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss; optimizer not stepped")
        loss.backward()
        total += float(loss.detach())
        depth_sum.index_add_(0, chunk.item_depths - 1, per_item)
        depth_n.index_add_(0, chunk.item_depths - 1, torch.ones_like(per_item))
        tokens += chunk.tokens
    condition_full.backward(condition_leaf.grad.to(condition_full.dtype))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
    lr = args.lr * min(1.0, update / max(args.warmup, 1))
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    per_depth = (depth_sum / depth_n.clamp_min(1)).tolist()
    return {"loss": total, "per_depth_mse": per_depth, "grad_norm": float(norm), "lr": lr,
            "objects": objects, "items": len(items), "rows": len(rows), "chunks": len(chunks),
            "padded_tokens": tokens}


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.params = [p for p in model.parameters()]
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - self.decay)


def ema_swap(ema: EMA):
    """Exchange live weights and EMA weights in place (no extra memory). Call twice to undo."""
    with torch.no_grad():
        for i, p in enumerate(ema.params):
            live = p.data
            p.data = ema.shadow[i]
            ema.shadow[i] = live


def ema_state(model, ema: EMA):
    """Full state_dict with parameters replaced by their EMA (buffers, if any, copied)."""
    shadow = {n: s for (n, _), s in zip(model.named_parameters(), ema.shadow)}
    return {k: (shadow[k] if k in shadow else v).detach().cpu() for k, v in model.state_dict().items()}


def save(path: Path, payload: dict):
    tmp = path.with_suffix(".pt.tmp")
    with tmp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def checkpoint_payload(model, ema, optimizer, args, update, full):
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "ema": ema_state(model, ema), "update": update, "args": vars(args) | {"output": str(args.output),
               "init": str(args.init)}, "architecture": "S0", "source_checkpoint_sha256": args.init_sha}
    if full:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def run_probe(model, ema, conditions, leaves, args, update, log):
    started = time.monotonic()
    raw = coarse_probe(model, conditions, leaves, max_depth=args.probe_depth)
    ema_swap(ema)
    try:
        smooth = coarse_probe(model, conditions, leaves, max_depth=args.probe_depth)
    finally:
        ema_swap(ema)
    row = {"update": update, "raw": raw, "ema": smooth, "seconds": time.monotonic() - started,
           "wall_minutes": (time.time() - args.start_wall) / 60}
    append_jsonl(log, row)
    print(f"[probe u{update}] raw exact-through-depth {raw['exact_through_depth']} | "
          f"ema {smooth['exact_through_depth']} ({row['seconds']:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", type=Path, required=True, help="S0 checkpoint-034000.pt (or a dense checkpoint)")
    p.add_argument("--init-sha", default=S0_SHA256, help="expected SHA256 of --init; 'skip' to skip the check")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--objects-per-update", type=int, default=8)
    p.add_argument("--coarse-depths", type=int, default=5)
    p.add_argument("--coarse-copies", type=int, default=4)
    p.add_argument("--row-tokens", type=int, default=4096)
    p.add_argument("--chunk-tokens", type=int, default=8192)
    p.add_argument("--checkpoint-activations", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--train-minutes", type=float, default=None, help="wall-clock training budget (required unless --preflight)")
    p.add_argument("--max-updates", type=int, default=10**9)
    p.add_argument("--probe-minutes", type=float, default=30.0)
    p.add_argument("--probe-depth", type=int, default=5)
    p.add_argument("--save-minutes", type=float, default=60.0)
    p.add_argument("--preflight", action="store_true", help="3 updates on the largest objects, no saves")
    args = p.parse_args()
    args.start_wall = time.time()
    assert args.preflight or args.train_minutes, "--train-minutes is required"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to ONE GPU"
    device = torch.device("cuda:0")
    torch.set_num_threads(8)
    if not args.preflight:
        assert not args.output.exists(), "output must be a new directory"
    args.output.mkdir(parents=True, exist_ok=args.preflight)
    status_path = args.output / "status.json"
    if args.init_sha != "skip":
        print("checking SHA256 of --init (28 GB, ~1-2 min)...", flush=True)
        actual = sha(args.init)
        assert actual == args.init_sha, f"--init SHA256 mismatch: {actual}"
    manifest, conditions, leaves, levels, _ = load_data(device)
    weights, _state = load_weights(args.init, "raw")
    model = build_model(weights, device)
    del weights, _state
    model.train().requires_grad_(True)
    model.flow.use_checkpoint = False
    model.condition_encoder.use_checkpoint = False
    cache_fixed_features(model, conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.0, fused=True)
    ema = EMA(model, args.ema_decay)
    write_json(args.output / "config.json", vars(args) | {"output": str(args.output), "init": str(args.init),
               "uids": manifest["train_uids"], "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(device),
               "parameters": sum(p.numel() for p in model.parameters())})
    if args.preflight:
        started = time.monotonic()
        probe = coarse_probe(model, conditions, leaves, max_depth=args.probe_depth)
        print(f"[preflight probe, S0 weights] exact-through-depth {probe['exact_through_depth']} "
              f"(S0 reference for seed 97029000: [49, 43, 30, 16, 9]) in {time.monotonic() - started:.0f}s",
              flush=True)
        append_jsonl(args.output / "preflight.jsonl", {"probe": probe, "seconds": time.monotonic() - started})
        sizes = [sum(len(c) for c, _ in lv) for lv in levels]
        largest = sorted(range(len(levels)), key=lambda i: -sizes[i])[:args.objects_per_update]
        global objects_for_update
        objects_for_update = lambda a, u, n: largest  # noqa: E731
        for update in range(1, 4):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.monotonic()
            row = train_update(model, optimizer, args, conditions, levels, update, device)
            torch.cuda.synchronize(device)
            ema.update()
            row.update(seconds=time.monotonic() - started,
                       peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)
            print({k: row[k] for k in ("loss", "grad_norm", "items", "rows", "chunks", "padded_tokens",
                                        "seconds", "peak_gib")}, flush=True)
            append_jsonl(args.output / "preflight.jsonl", row)
        print("PREFLIGHT_OK (largest objects; typical updates are faster)", flush=True)
        return
    deadline = time.monotonic() + args.train_minutes * 60
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    probe_log = args.output / "probes.jsonl"
    run_probe(model, ema, conditions, leaves, args, 0, probe_log)
    next_probe = time.monotonic() + args.probe_minutes * 60
    next_save = time.monotonic() + args.save_minutes * 60
    update = 0
    try:
        while time.monotonic() < deadline and update < args.max_updates and not stop["flag"]:
            update += 1
            started = time.monotonic()
            row = train_update(model, optimizer, args, conditions, levels, update, device)
            ema.update()
            torch.cuda.synchronize(device)
            row.update(update=update, seconds=time.monotonic() - started,
                       peak_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                       wall_minutes=(time.time() - args.start_wall) / 60)
            append_jsonl(args.output / "train.jsonl", row)
            if update % 10 == 0 or update <= 3:
                print(f"u{update} loss {row['loss']:.5f} gn {row['grad_norm']:.3f} lr {row['lr']:.2e} "
                      f"{row['seconds']:.1f}s peak {row['peak_gib']:.1f}GiB "
                      f"depthMSE {[round(x, 4) for x in row['per_depth_mse']]}", flush=True)
            write_json(status_path, {"state": "training", "update": update, "last_loss": row["loss"],
                                     "minutes_left": (deadline - time.monotonic()) / 60})
            if time.monotonic() >= next_probe:
                run_probe(model, ema, conditions, leaves, args, update, probe_log)
                next_probe = time.monotonic() + args.probe_minutes * 60
            if time.monotonic() >= next_save and time.monotonic() < deadline - 10 * 60:
                save(args.output / "latest_weights.pt", checkpoint_payload(model, ema, optimizer, args, update, False))
                next_save = time.monotonic() + args.save_minutes * 60
        run_probe(model, ema, conditions, leaves, args, update, probe_log)
        write_json(status_path, {"state": "saving_final", "update": update})
        final = args.output / "final.pt"
        save(final, checkpoint_payload(model, ema, optimizer, args, update, True))
        stale = args.output / "latest_weights.pt"
        if stale.exists():
            stale.unlink()
        write_json(status_path, {"state": "training_complete", "update": update, "final": str(final),
                                 "stopped_by_signal": stop["flag"]})
        print(f"TRAINING_COMPLETE updates={update} final={final}", flush=True)
    except Exception as error:
        write_json(status_path, {"state": "failed", "update": update, "error": repr(error),
                                 "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
