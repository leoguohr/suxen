"""Dense 50-object Vertex overfit training (S0 architecture). One GPU per process.

Round 1 (2026-10-09): warm start from S0@34000 with a wall-clock deadline (--train-minutes).
Round 2: unattended jobs (--max-updates) that resume themselves after a restart, evaluate every
--eval-every updates, stop early at 100/100, and decay the LR over the last --decay-fraction.

Same model, same rectified-flow velocity-MSE objective, same data. Every update takes K objects
and ALL nine depths of each (VecSet runs once per object); depths 1..C get R stratified-t copies.
--loss-weighting item: each (object, depth) has equal total weight (round 1, original protocol).
--loss-weighting token: every token has equal weight (plain mean over all tokens of the update).
"""
from __future__ import annotations

import os
# nexus-algo pairs protobuf 4.x with system onnx 1.16; torch.optim lazily imports onnx.
# S0 ran with the same setting. Must be set before torch is imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import hashlib
import math
import os
from pathlib import Path
import signal
import time
import traceback

import torch

from common import (S0_SHA256, append_jsonl, build_model, cache_fixed_features, coarse_probe,
                    load_data, run_evaluation, sha, write_json)
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


SIX_NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def perturb_parents(codes, target, depth, drop_max, add_max, g, device):
    """Self-correction training (J6): corrupt one item's parent set the way generation errs.

    Drop a random fraction (U[0, drop_max]) of the ground-truth parents, and add up to a random
    fraction (U[0, add_max]) of spurious parents: 6-neighbours of kept parents that are NOT
    ground-truth cells. Kept parents keep their true child targets; spurious parents get the
    all-zero target ("this cell has no children"), so the model learns to let wrong cells die out.
    Depth 1 (the root) is never perturbed.
    """
    n = len(codes)
    if depth < 2 or n == 0:
        return codes, target
    keep = torch.rand((n,), generator=g, device=device) >= torch.rand((), generator=g, device=device) * drop_max
    if not bool(keep.any()):
        keep[int(torch.randint(n, (1,), generator=g, device=device))] = True
    kept_codes, kept_target = codes[keep], target[keep]
    count = int(round(float(torch.rand((), generator=g, device=device)) * add_max * n))
    if count == 0:
        return kept_codes, kept_target
    limit = 1 << (depth - 1)
    directions = torch.tensor(SIX_NEIGHBOURS, dtype=torch.long, device=device)
    picks = torch.randint(len(kept_codes), (count,), generator=g, device=device)
    candidates = kept_codes[picks] + directions[torch.randint(6, (count,), generator=g, device=device)]
    inside = ((candidates >= 0) & (candidates < limit)).all(dim=1)
    candidates = candidates[inside]
    key = lambda c: (c[:, 0] * limit + c[:, 1]) * limit + c[:, 2]  # noqa: E731
    candidate_keys = torch.unique(key(candidates))
    candidate_keys = candidate_keys[~torch.isin(candidate_keys, key(codes))]
    if len(candidate_keys) == 0:
        return kept_codes, kept_target
    spurious = torch.stack([candidate_keys // (limit * limit), (candidate_keys // limit) % limit,
                            candidate_keys % limit], dim=1)
    return (torch.cat([kept_codes, spurious]),
            torch.cat([kept_target, torch.zeros((len(spurious), 8), dtype=target.dtype, device=device)]))


def build_items(args, objects, levels, update, device) -> list[Item]:
    g = seeded(f"dense|{args.seed}|{update}", device)
    items = []
    unit_weight = 1.0 / (len(objects) * len(levels[0]))
    fine_copies = getattr(args, "fine_copies", 1)
    drop_max, add_max = getattr(args, "parent_drop", 0.0), getattr(args, "parent_add", 0.0)
    for k, obj in enumerate(objects):
        for depth0, (codes, target) in enumerate(levels[obj]):
            depth = depth0 + 1
            copies = args.coarse_copies if depth <= args.coarse_depths else fine_copies
            u = torch.rand((copies,), generator=g, device=device)
            times = ((torch.arange(copies, device=device) + u) / copies).tolist()
            for t in times:
                item_codes, item_target = codes, target
                if drop_max > 0 or add_max > 0:  # no extra random draws when off: J1/J2 streams unchanged
                    item_codes, item_target = perturb_parents(codes, target, depth, drop_max, add_max, g, device)
                noise = torch.randn(item_target.shape, generator=g, device=device)
                items.append(Item(obj=k, depth=depth, codes=item_codes, target=item_target, time=float(t),
                                  noise=noise, weight=unit_weight / copies))
    return items


def lr_at(args, update: int) -> float:
    """Linear warmup over --warmup updates, constant, then linear decay to --min-lr over the
    last --decay-fraction of --max-updates (only when --max-updates is set)."""
    lr = args.lr * min(1.0, update / max(args.warmup, 1))
    if getattr(args, "decay_fraction", 0) > 0 and args.max_updates < 10**9:
        start = int(round(args.max_updates * (1 - args.decay_fraction)))
        if update > start:
            frac = min(1.0, (update - start) / max(args.max_updates - start, 1))
            lr = args.min_lr + (args.lr - args.min_lr) * (1.0 - frac)
    return lr


def train_update(model, optimizer, args, conditions, levels, update, device):
    objects = objects_for_update(args, update, len(conditions))
    items = build_items(args, objects, levels, update, device)
    token_denominator = (8.0 * sum(len(item.codes) for item in items)
                         if getattr(args, "loss_weighting", "item") == "token" else None)
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
        loss, per_item = chunk_loss(prediction, chunk, token_denominator)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss; optimizer not stepped")
        loss.backward()
        total += float(loss.detach())
        depth_sum.index_add_(0, chunk.item_depths - 1, per_item)
        depth_n.index_add_(0, chunk.item_depths - 1, torch.ones_like(per_item))
        tokens += chunk.tokens
    condition_full.backward(condition_leaf.grad.to(condition_full.dtype))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
    lr = lr_at(args, update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # free gradient memory between updates and before evaluations
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


def checkpoint_payload(model, ema, optimizer, args, update, full, **extra):
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "ema": ema_state(model, ema), "update": update,
               "args": vars(args) | {"output": str(args.output), "init": str(args.init)},
               "architecture": "S0", "source_checkpoint_sha256": args.init_sha, **extra}
    if full:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def run_probe(model, ema, conditions, leaves, args, update, log):
    if args.probe_minutes <= 0:
        return
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


def load_ema_into(ema, model, state_dict, device):
    names = [n for n, _ in model.named_parameters()]
    with torch.no_grad():
        for i, name in enumerate(names):
            ema.shadow[i].copy_(state_dict[name].to(device))


def evals_done(args) -> set:
    path = args.output / "evals.jsonl"
    if not path.exists():
        return set()
    import json
    return {(row["update"], row["weights"]) for row in map(json.loads, path.read_text().splitlines()) if row}


def run_milestone_eval(model, ema, data, args, update, weights):
    """Full 100-tree acceptance evaluation in-process; resumable (incomplete dirs are set aside)."""
    manifest, conditions, leaves, raw_gt = data
    folder = args.output / "evals" / f"u{update:06d}_{weights}"
    if folder.exists():
        folder.rename(folder.with_name(folder.name + f".incomplete-{int(time.time())}"))
    if weights == "ema":
        ema_swap(ema)
    try:
        summary = run_evaluation(model, manifest, conditions, leaves, raw_gt, folder, steps=args.eval_steps,
                                 info={"update": update, "weights": weights, "run_output": str(args.output)})
    finally:
        if weights == "ema":
            ema_swap(ema)
    row = {"update": update, "weights": weights, "full_trees_exact": summary["full_trees_exact"],
           "per_depth_exact": [d["full_level_exact"] for d in summary["per_depth"]],
           "first_mismatch_depth_histogram": summary["first_mismatch_depth_histogram"],
           "exact_by_vertex_count": summary["exact_by_vertex_count"], "minutes": summary["minutes"],
           "wall_minutes": (time.time() - args.start_wall) / 60}
    append_jsonl(args.output / "evals.jsonl", row)
    print(f"[eval u{update} {weights}] full_trees_exact {row['full_trees_exact']}/100 per-depth "
          f"{row['per_depth_exact']} by-size {row['exact_by_vertex_count']} ({row['minutes']:.0f} min)", flush=True)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init", type=Path, required=True, help="S0 checkpoint-034000.pt or a dense checkpoint")
    p.add_argument("--init-sha", default=S0_SHA256, help="expected SHA256 of --init; 'skip' to skip the check")
    p.add_argument("--init-state", choices=("weights", "full"), default="weights",
                   help="full: also load AdamW and EMA from --init when present")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--decay-fraction", type=float, default=0.0, help="linear LR decay over the last fraction of --max-updates")
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--loss-weighting", choices=("item", "token"), default="item")
    p.add_argument("--objects-per-update", type=int, default=8)
    p.add_argument("--coarse-depths", type=int, default=5)
    p.add_argument("--coarse-copies", type=int, default=4)
    p.add_argument("--fine-copies", type=int, default=1, help="stratified-t copies for depths above --coarse-depths")
    p.add_argument("--parent-drop", type=float, default=0.0, help="self-correction: max fraction of GT parents dropped per item")
    p.add_argument("--parent-add", type=float, default=0.0, help="self-correction: max fraction of spurious neighbour parents added")
    p.add_argument("--row-tokens", type=int, default=4096)
    p.add_argument("--chunk-tokens", type=int, default=8192)
    p.add_argument("--checkpoint-activations", type=int, default=1)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--train-minutes", type=float, default=None, help="round-1 wall-clock budget; omit for unattended jobs")
    p.add_argument("--max-updates", type=int, default=10**9)
    p.add_argument("--probe-minutes", type=float, default=30.0, help="<=0 disables probes")
    p.add_argument("--probe-depth", type=int, default=5)
    p.add_argument("--save-minutes", type=float, default=60.0)
    p.add_argument("--save-optimizer", type=int, default=1,
                   help="round-1 final.pt: 1 model+EMA+AdamW (~37 GB); 0 model+EMA (~19 GB)")
    p.add_argument("--eval-every", type=int, default=0, help="full 100-tree evaluation every N updates (raw weights)")
    p.add_argument("--final-evals", default="", help="comma list of weights evaluated at the end, e.g. raw,ema")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--stop-at-pass", type=int, default=1, help="stop early when an evaluation reaches 100/100")
    p.add_argument("--preflight", action="store_true", help="3 updates on the largest objects, no saves")
    args = p.parse_args()
    args.start_wall = time.time()
    unattended = args.train_minutes is None and not args.preflight
    assert args.preflight or args.train_minutes or args.max_updates < 10**9, \
        "give --train-minutes (round 1) or --max-updates (unattended)"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to ONE GPU"
    device = torch.device("cuda:0")
    torch.set_num_threads(8)
    status_path = args.output / "status.json"
    latest = args.output / "latest.pt"
    if status_path.exists():
        import json
        previous = json.loads(status_path.read_text())
        if previous.get("state") == "complete":
            print(f"ALREADY_COMPLETE {args.output}; nothing to do", flush=True)
            return
    resume = unattended and latest.exists()
    if not resume and args.output.exists() and not args.preflight:
        if unattended and any(args.output.iterdir()):
            aside = args.output / f"restart-{int(time.time())}"
            aside.mkdir()
            for child in list(args.output.iterdir()):
                if child != aside and not child.name.startswith("restart-"):
                    child.rename(aside / child.name)
            print(f"no latest.pt: previous partial files moved to {aside}", flush=True)
        elif not unattended:
            assert not any(args.output.iterdir()), "output must be a new directory"
    args.output.mkdir(parents=True, exist_ok=True)
    source = latest if resume else args.init
    if not resume and args.init_sha != "skip":
        print("checking SHA256 of --init ...", flush=True)
        actual = sha(args.init)
        assert actual == args.init_sha, f"--init SHA256 mismatch: {actual}"
    manifest, conditions, leaves, levels, raw_gt = load_data(device)
    state = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    model = build_model(state["model"], device)
    model.train().requires_grad_(True)
    model.flow.use_checkpoint = False
    model.condition_encoder.use_checkpoint = False
    cache_fixed_features(model, conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.0, fused=True)
    ema = EMA(model, args.ema_decay)
    start_update, final_stage = 0, False
    if resume or args.init_state == "full":
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        if "ema" in state:
            load_ema_into(ema, model, state["ema"], device)
    if resume:
        start_update = int(state["update"])
        final_stage = bool(state.get("final_stage", False))
        print(f"RESUMED from {latest} at update {start_update} (final_stage={final_stage})", flush=True)
    del state
    write_json(args.output / ("config.json" if not resume else f"config.resume-u{start_update}.json"),
               vars(args) | {"output": str(args.output), "init": str(args.init), "resumed_from": str(source) if resume else None,
                             "start_update": start_update, "uids": manifest["train_uids"], "torch": torch.__version__,
                             "gpu": torch.cuda.get_device_name(device),
                             "parameters": sum(p.numel() for p in model.parameters())})
    if args.preflight:
        started = time.monotonic()
        probe = coarse_probe(model, conditions, leaves, max_depth=args.probe_depth)
        print(f"[preflight probe] exact-through-depth {probe['exact_through_depth']} "
              f"(S0@34000 reference for seed 97029000: [49, 43, 30, 16, 9]) in {time.monotonic() - started:.0f}s",
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
            print({k: row[k] for k in ("loss", "grad_norm", "lr", "items", "rows", "chunks", "padded_tokens",
                                        "seconds", "peak_gib")}, flush=True)
            append_jsonl(args.output / "preflight.jsonl", row)
        print("PREFLIGHT_OK (largest objects; typical updates are faster)", flush=True)
        return

    data = (manifest, conditions, leaves, raw_gt)
    deadline = time.monotonic() + args.train_minutes * 60 if args.train_minutes else float("inf")
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    probe_log = args.output / "probes.jsonl"
    final_evals = [w for w in args.final_evals.split(",") if w]
    passed = False

    def save_latest(update, final=False):
        write_json(status_path, {"state": "saving", "update": update})
        save(latest, checkpoint_payload(model, ema, optimizer, args, update, True, final_stage=final))
        write_json(status_path, {"state": "final_evaluation" if final else "training", "update": update,
                                 "latest": str(latest)})

    update = start_update
    try:
        if not final_stage:
            # A restart during a milestone evaluation resumes at that update: finish the evaluation first.
            if (resume and args.eval_every > 0 and update > 0 and update % args.eval_every == 0
                    and update < args.max_updates and (update, "raw") not in evals_done(args)):
                row = run_milestone_eval(model, ema, data, args, update, "raw")
                passed = bool(args.stop_at_pass) and row["full_trees_exact"] == 100
            run_probe(model, ema, conditions, leaves, args, update, probe_log)
            next_probe = time.monotonic() + args.probe_minutes * 60
            next_save = time.monotonic() + args.save_minutes * 60
            while time.monotonic() < deadline and update < args.max_updates and not stop["flag"] and not passed:
                update += 1
                started = time.monotonic()
                row = train_update(model, optimizer, args, conditions, levels, update, device)
                ema.update()
                torch.cuda.synchronize(device)
                row.update(update=update, seconds=time.monotonic() - started,
                           peak_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                           wall_minutes=(time.time() - args.start_wall) / 60)
                append_jsonl(args.output / "train.jsonl", row)
                if update % 10 == 0 or update <= start_update + 3:
                    print(f"u{update} loss {row['loss']:.5f} gn {row['grad_norm']:.3f} lr {row['lr']:.2e} "
                          f"{row['seconds']:.1f}s peak {row['peak_gib']:.1f}GiB "
                          f"depthMSE {[round(x, 4) for x in row['per_depth_mse']]}", flush=True)
                if update % 25 == 0:
                    write_json(status_path, {"state": "training", "update": update, "last_loss": row["loss"],
                                             "max_updates": args.max_updates})
                if time.monotonic() >= next_probe:
                    run_probe(model, ema, conditions, leaves, args, update, probe_log)
                    next_probe = time.monotonic() + args.probe_minutes * 60
                milestone = args.eval_every > 0 and update % args.eval_every == 0 and update < args.max_updates
                if unattended and (milestone or time.monotonic() >= next_save):
                    save_latest(update)
                    next_save = time.monotonic() + args.save_minutes * 60
                elif not unattended and time.monotonic() >= next_save and time.monotonic() < deadline - 10 * 60:
                    save(args.output / "latest_weights.pt", checkpoint_payload(model, ema, optimizer, args, update, False))
                    next_save = time.monotonic() + args.save_minutes * 60
                if milestone and (update, "raw") not in evals_done(args):
                    row = run_milestone_eval(model, ema, data, args, update, "raw")
                    passed = bool(args.stop_at_pass) and row["full_trees_exact"] == 100
            if stop["flag"]:
                if unattended:
                    save_latest(update)
                    write_json(status_path, {"state": "interrupted", "update": update, "latest": str(latest)})
                    print(f"INTERRUPTED at update {update}; rerun the same command to resume", flush=True)
                    raise SystemExit(143)
            run_probe(model, ema, conditions, leaves, args, update, probe_log)
            if not unattended:  # round-1 behaviour
                write_json(status_path, {"state": "saving_final", "update": update})
                final = args.output / "final.pt"
                save(final, checkpoint_payload(model, ema, optimizer, args, update, bool(args.save_optimizer)))
                stale = args.output / "latest_weights.pt"
                if stale.exists():
                    stale.unlink()
                write_json(status_path, {"state": "training_complete", "update": update, "final": str(final),
                                         "stopped_by_signal": stop["flag"]})
                print(f"TRAINING_COMPLETE updates={update} final={final}", flush=True)
                return
            save_latest(update, final=True)
        print(f"TRAINING_COMPLETE updates={update}; final evaluations {final_evals}", flush=True)
        done = evals_done(args)
        for weights in final_evals:
            if (update, weights) not in done:
                run_milestone_eval(model, ema, data, args, update, weights)
        final = args.output / "final.pt"
        latest.replace(final)
        write_json(status_path, {"state": "complete", "update": update, "final": str(final),
                                 "passed_early": passed})
        print(f"JOB_COMPLETE updates={update} final={final}", flush=True)
    except SystemExit:
        raise
    except Exception as error:
        write_json(status_path, {"state": "failed", "update": update, "error": repr(error),
                                 "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
