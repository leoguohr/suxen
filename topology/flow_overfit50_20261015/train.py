"""Topology flow overfit on the fixed 50 objects at paper scale (one GPU per process).

Every update trains on ALL 50 objects packed into one token axis (52,820 tokens). Each object gets one
flow time t, stratified across the 50 objects: t_i = (rank_i + u_i) / 50. Target: a fresh posterior
sample z = mu + sigma * eps of the frozen OwnAE-v2 VAE, normalized per channel with the cache
statistics. Path x_t = (1 - t) * noise + t * z (t = 0 noise), velocity target z - noise, loss = mean
squared error over every token and channel of the update. AdamW, warmup, constant LR, linear decay
over the last --decay-fraction, EMA. Chain evaluations (flow -> frozen decoder -> strict edge/face
metric) every --eval-every updates and at the end.

Unattended behaviour (same contract as the vertex chat's dense_train.py): rerunning the same command
resumes from OUTPUT/latest.pt; SIGTERM saves latest.pt and exits 143; completion renames it final.pt;
a completed job prints ALREADY_COMPLETE.
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import S0_SHA256, append_jsonl, generator, save_torch, sha256_file, write_json  # noqa: E402
from data import load_cache  # noqa: E402
from metrics import aggregate, evaluate_latents, load_frozen_vae  # noqa: E402
from model import PAPER, TINY, TopologyFlowSystem, parameter_count, transfer_vertex_weights  # noqa: E402
from sampling import generate  # noqa: E402

T_BINS = 10


# ---------------------------------------------------------------- schedule, EMA, batches
def lr_at(args, update: int) -> float:
    """Linear warmup over --warmup updates, constant, then linear decay to --min-lr over the last
    --decay-fraction of --max-updates (updates are counted from 1)."""
    lr = args.lr * min(1.0, update / max(args.warmup, 1))
    if args.decay_fraction > 0:
        start = int(round(args.max_updates * (1 - args.decay_fraction)))
        if update > start:
            frac = min(1.0, (update - start) / max(args.max_updates - start, 1))
            lr = args.min_lr + (args.lr - args.min_lr) * (1.0 - frac)
    return lr


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.params = list(model.parameters())
        self.shadow = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def update(self):
        torch._foreach_lerp_(self.shadow, [p.detach() for p in self.params], 1.0 - self.decay)

    @torch.no_grad()
    def swap(self):
        """Exchange live and EMA weights in place; call twice to undo."""
        for i, p in enumerate(self.params):
            live = p.data
            p.data = self.shadow[i]
            self.shadow[i] = live

    def state(self, model) -> dict:
        names = [n for n, _ in model.named_parameters()]
        return {n: s.detach().cpu() for n, s in zip(names, self.shadow)}

    @torch.no_grad()
    def load(self, model, state: dict):
        for (name, _), shadow in zip(model.named_parameters(), self.shadow):
            shadow.copy_(state[name].to(shadow.device))


def make_batch(data, seed: int, update: int, target: str = "posterior"):
    """Deterministic in (seed, update): Run A and Run B see identical t, noise and posterior samples."""
    device = data.device
    g = generator(f"topology-overfit50/train/{seed}/{update}", device)
    rank = torch.randperm(data.count, generator=g, device=device)
    u = torch.rand(data.count, generator=g, device=device)
    t = (rank.float() + u) / data.count
    eps = torch.randn(data.mu.shape, generator=g, device=device)
    noise = torch.randn(data.mu.shape, generator=g, device=device)
    clean = data.mu + data.sigma * eps if target == "posterior" else data.mu
    z = (clean - data.mean) / data.std
    tt = t[data.seg.index].unsqueeze(-1)
    return t, (1 - tt) * noise + tt * z, z - noise


def object_groups(data, chunk_tokens: int) -> list:
    """Contiguous object groups of <= chunk_tokens tokens (0: one group with everything)."""
    if chunk_tokens <= 0 or data.seg.tokens <= chunk_tokens:
        return [dict(data=data, objects=list(range(data.count)), tokens=slice(0, data.seg.tokens))]
    groups, current, size = [], [], 0
    for i, n in enumerate(data.lengths):
        if current and size + n > chunk_tokens:
            groups.append(current)
            current, size = [], 0
        current.append(i)
        size += n
    groups.append(current)
    return [dict(data=data.select(g), objects=g,
                 tokens=slice(data.token_slice(g[0]).start, data.token_slice(g[-1]).stop)) for g in groups]


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")


def train_update(system, optimizer, data, groups, args, update: int) -> dict:
    device = data.device
    t, noisy, target = make_batch(data, args.seed, update, args.target)
    denominator = float(noisy.numel())
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    sse = torch.zeros(data.count, device=device)
    for group in groups:
        sub, sl = group["data"], group["tokens"]
        objects = torch.tensor(group["objects"], device=device)
        with autocast(device):
            condition = system.encode_condition(sub.points, sub.point_valid, sub.fps_indices,
                                                use_checkpoint=bool(args.checkpoint_activations))
            prediction = system.flow.forward_packed(noisy[sl], t[objects], sub.vertices, sub.seg, condition,
                                                    args.attention, bool(args.checkpoint_activations))
        error = (prediction.float() - target[sl]).square()
        loss = error.sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss; optimizer not stepped")
        loss.backward()
        total += float(loss.detach())
        sse.index_add_(0, objects[sub.seg.index], error.detach().sum(-1))
    norm = torch.nn.utils.clip_grad_norm_(system.parameters(), args.clip, error_if_nonfinite=True)
    lr = lr_at(args, update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    lengths = torch.tensor(data.lengths, device=device, dtype=torch.float32)
    per_object = sse / (lengths * noisy.shape[1])
    bins = torch.clamp((t * T_BINS).long(), max=T_BINS - 1)
    bin_sse = torch.zeros(T_BINS, device=device).index_add_(0, bins, sse)
    bin_tokens = torch.zeros(T_BINS, device=device).index_add_(0, bins, lengths * noisy.shape[1])
    return dict(loss=total, grad_norm=float(norm), lr=lr,
                t_bin_mse=(bin_sse / bin_tokens.clamp_min(1)).tolist(),
                largest_object_mse=float(per_object[int(np.argmax(data.lengths))]),
                smallest_object_mse=float(per_object[int(np.argmin(data.lengths))]))


# ---------------------------------------------------------------- probes and evaluations
def probe_objects(data) -> list:
    order = np.argsort(data.lengths)
    picks = [order[0], order[len(order) // 5], order[2 * len(order) // 5], order[3 * len(order) // 5],
             order[4 * len(order) // 5], order[-1]]
    return sorted({int(i) for i in picks})


@torch.no_grad()
def run_probe(system, data, args) -> dict:
    """Cheap learning signal: Euler-20 latent RMS error vs normalized mu on 6 objects, seed 0."""
    sub = data.select(probe_objects(data))
    latents = generate(system, sub, seed=0, steps=20, sampler="euler", dtype="bf16", attention=args.attention)
    rmse = {uid: float((latents[sub.token_slice(i)] - sub.mu_normalized[sub.token_slice(i)]).square().mean().sqrt())
            for i, uid in enumerate(sub.uids)}
    return dict(latent_rmse=rmse, sizes=dict(zip(sub.uids, sub.lengths)))


def evaluate_chain(system, vae, data, seeds, steps, sampler, dtype, attention) -> dict:
    results = {}
    for seed in seeds:
        latents = generate(system, data, seed=seed, steps=steps, sampler=sampler, dtype=dtype, attention=attention)
        rows = evaluate_latents(vae, data, latents, normalized=True)
        results[seed] = dict(summary=aggregate(rows), rows=rows)
    return results


def run_milestone_eval(system, ema, vae, data, args, update: int, weights: str) -> dict:
    folder = args.output / "evals" / f"u{update:06d}_{weights}"
    if weights == "ema":
        ema.swap()
    started = time.monotonic()
    try:
        results = evaluate_chain(system, vae, data, args.eval_seed_list, args.eval_steps, "euler", "bf16",
                                 args.attention)
    finally:
        if weights == "ema":
            ema.swap()
    for seed, result in results.items():
        write_json(folder / f"seed{seed}_summary.json", result["summary"])
        with (folder / f"seed{seed}_objects.jsonl").open("w") as stream:
            for row in result["rows"]:
                stream.write(json.dumps(row) + "\n")
    faces = {seed: r["summary"]["face"]["micro_f1"] for seed, r in results.items()}
    passed = all(f >= args.pass_f1 for f in faces.values()) and not any(
        r["summary"]["lower_bound"] for r in results.values())
    row = dict(update=update, weights=weights, steps=args.eval_steps,
               face_micro_f1=faces,
               edge_micro_f1={s: r["summary"]["edge"]["micro_f1"] for s, r in results.items()},
               joint_strict={s: r["summary"]["joint_strict"] for s, r in results.items()},
               latent_rmse={s: r["summary"]["latent_rmse_token_weighted"] for s, r in results.items()},
               by_size={s: r["summary"]["by_size"] for s, r in results.items()},
               overflow={s: r["summary"]["overflow_objects"] for s, r in results.items()},
               passed=passed, minutes=(time.monotonic() - started) / 60)
    append_jsonl(args.output / "evals.jsonl", row)
    print(f"[eval u{update} {weights}] face F1 {faces} strict {row['joint_strict']} "
          f"latent RMSE {row['latent_rmse']} ({row['minutes']:.1f} min)", flush=True)
    return row


def evals_done(output: Path) -> set:
    path = output / "evals.jsonl"
    if not path.exists():
        return set()
    return {(r["update"], r["weights"]) for r in map(json.loads, path.read_text().splitlines()) if r}


# ---------------------------------------------------------------- setup
def build_system(args, device):
    cfg = TINY if args.tiny else PAPER
    torch.manual_seed(args.init_seed)
    with torch.device(device):
        system = TopologyFlowSystem(cfg)
    return system.float()


def checkpoint_payload(system, ema, optimizer, args, update, final_stage) -> dict:
    return dict(model={k: v.detach().cpu() for k, v in system.state_dict().items()},
                ema=ema.state(system), optimizer=optimizer.state_dict(), update=update,
                final_stage=final_stage, config=system.cfg.to_dict(),
                args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})


def compare_attention_backends(system, data, args) -> dict:
    """bf16 forward of the probe objects with flash varlen vs per-object SDPA (same weights, same inputs)."""
    from model import flash_attn_varlen_func
    if flash_attn_varlen_func is None:
        return dict(flash_available=False)
    sub = data.select(probe_objects(data))
    noise = torch.randn(sub.mu.shape, generator=generator("backend-check", data.device), device=data.device)
    time_ = torch.full((sub.count,), 0.5, device=data.device)
    outputs = {}
    with torch.no_grad(), autocast(data.device):
        condition = system.encode_condition(sub.points, sub.point_valid, sub.fps_indices)
        for backend in ("flash", "sdpa"):
            outputs[backend] = system.velocity(noise, time_, sub, condition, backend).float()
    diff = (outputs["flash"] - outputs["sdpa"]).abs()
    return dict(flash_available=True, max_abs_diff=float(diff.max()),
                rel_rms_diff=float(diff.square().mean().sqrt() / outputs["sdpa"].square().mean().sqrt().clamp_min(1e-12)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True, help="fixed-50 latent cache directory (manifest f14f796c...)")
    p.add_argument("--vae", type=Path, required=True, help="OwnAE-v2 vae-1000.pt (e89b8078...)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--init-vertex", type=Path, default=None, help="Run B: S0 checkpoint-034000.pt for the warm start")
    p.add_argument("--init-vertex-sha", default=S0_SHA256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--max-updates", type=int, default=10000)
    p.add_argument("--decay-fraction", type=float, default=0.2)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--target", choices=("posterior", "mu"), default="posterior")
    p.add_argument("--seed", type=int, default=20261015, help="data stream (t, noise, posterior samples)")
    p.add_argument("--init-seed", type=int, default=20261015, help="parameter initialization")
    p.add_argument("--chunk-tokens", type=int, default=0, help="split each update into object groups of <= N tokens (0: all at once)")
    p.add_argument("--checkpoint-activations", type=int, default=1)
    p.add_argument("--attention", choices=("auto", "flash", "sdpa"), default="auto")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-seeds", default="0,1")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--milestone-weights", default="raw,ema")
    p.add_argument("--final-weights", default="raw,ema")
    p.add_argument("--pass-f1", type=float, default=0.99)
    p.add_argument("--stop-at-pass", type=int, default=1)
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--save-minutes", type=float, default=60.0)
    p.add_argument("--speed-check", type=int, default=0, help="run N updates, report s/update and memory, save nothing")
    p.add_argument("--tiny", action="store_true", help="tiny model, for plumbing checks only")
    args = p.parse_args()
    args.eval_seed_list = [int(s) for s in args.eval_seeds.split(",") if s != ""]
    args.start_wall = time.time()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, "set CUDA_VISIBLE_DEVICES to ONE GPU"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False  # the frozen decoder metric is defined in strict FP32
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)

    status_path, latest = args.output / "status.json", args.output / "latest.pt"
    if status_path.exists() and json.loads(status_path.read_text()).get("state") == "complete":
        print(f"ALREADY_COMPLETE {args.output}", flush=True)
        return
    resume = latest.exists() and not args.speed_check
    if not resume and not args.speed_check and args.output.exists() and any(args.output.iterdir()):
        aside = args.output / f"restart-{int(time.time())}"
        aside.mkdir()
        for child in list(args.output.iterdir()):
            if child != aside and not child.name.startswith("restart-"):
                child.rename(aside / child.name)
        print(f"no latest.pt: previous partial files moved to {aside}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_cache(args.cache, device)
    system = build_system(args, device)
    data.compute_fps(system.cfg.condition_tokens)
    vae = load_frozen_vae(args.vae, device)
    transfer = None
    if args.init_vertex is not None and not resume:
        if args.init_vertex_sha != "skip":
            actual = sha256_file(args.init_vertex)
            assert actual == args.init_vertex_sha, f"--init-vertex SHA256 mismatch: {actual}"
        s0 = torch.load(args.init_vertex, map_location="cpu", mmap=True, weights_only=False)
        transfer = transfer_vertex_weights(system, s0["model"])
        del s0
        print(f"warm start from S0: {transfer['copied']} tensors copied; fresh: {transfer['kept_fresh']}", flush=True)
    system.train()
    optimizer = torch.optim.AdamW(system.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=True)
    ema = EMA(system, args.ema_decay)
    start_update, final_stage = 0, False
    if resume:
        state = torch.load(latest, map_location="cpu", mmap=True, weights_only=False)
        system.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        ema.load(system, state["ema"])
        start_update, final_stage = int(state["update"]), bool(state["final_stage"])
        del state
        print(f"RESUMED from {latest} at update {start_update} (final_stage={final_stage})", flush=True)
    groups = object_groups(data, args.chunk_tokens)
    info = dict(vars(args), output=str(args.output), cache=str(args.cache), vae=str(args.vae),
                init_vertex=str(args.init_vertex) if args.init_vertex else None,
                start_update=start_update, resumed=resume, uids=data.uids, tokens=data.seg.tokens,
                parameters=dict(total=parameter_count(system), flow=parameter_count(system.flow),
                                condition_encoder=parameter_count(system.condition_encoder)),
                config=system.cfg.to_dict(), transfer=transfer, groups=len(groups),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(device))
    write_json(args.output / ("config.json" if not resume else f"config.resume-u{start_update}.json"), info)

    if args.speed_check:
        backends = compare_attention_backends(system, data, args)
        print(f"[attention backends] {backends}", flush=True)
        rows = []
        for update in range(1, args.speed_check + 1):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            row = train_update(system, optimizer, data, groups, args, update)
            ema.update()
            torch.cuda.synchronize(device)
            row.update(update=update, seconds=time.monotonic() - started,
                       peak_gib=torch.cuda.max_memory_allocated(device) / 2**30)
            rows.append(row)
            print(f"u{update} loss {row['loss']:.4f} gn {row['grad_norm']:.3f} {row['seconds']:.1f}s "
                  f"peak {row['peak_gib']:.1f}GiB", flush=True)
        steady = [r["seconds"] for r in rows[3:]] or [r["seconds"] for r in rows]
        report = dict(updates=len(rows), seconds_per_update_median=float(np.median(steady)),
                      peak_gib_max=max(r["peak_gib"] for r in rows), attention_backends=backends,
                      tokens_per_update=data.seg.tokens, parameters=info["parameters"],
                      hours_for_max_updates=float(np.median(steady)) * args.max_updates / 3600, rows=rows)
        write_json(args.output / "speed_check.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
        print("SPEED_CHECK_OK", flush=True)
        return

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    milestone_weights = [w for w in args.milestone_weights.split(",") if w]
    final_weights = [w for w in args.final_weights.split(",") if w]

    def save_latest(update, final=False):
        write_json(status_path, {"state": "saving", "update": update})
        save_torch(latest, checkpoint_payload(system, ema, optimizer, args, update, final))
        write_json(status_path, {"state": "final_evaluation" if final else "training", "update": update,
                                 "latest": str(latest)})

    update, passed = start_update, False
    try:
        if not final_stage:
            next_save = time.monotonic() + args.save_minutes * 60
            # A restart during a milestone evaluation finishes that evaluation first.
            if resume and args.eval_every and update and update % args.eval_every == 0 and update < args.max_updates:
                for weights in milestone_weights:
                    if (update, weights) not in evals_done(args.output):
                        row = run_milestone_eval(system, ema, vae, data, args, update, weights)
                        passed = passed or (bool(args.stop_at_pass) and row["passed"])
            while update < args.max_updates and not stop["flag"] and not passed:
                update += 1
                started = time.monotonic()
                row = train_update(system, optimizer, data, groups, args, update)
                ema.update()
                torch.cuda.synchronize(device)
                row.update(update=update, seconds=time.monotonic() - started,
                           peak_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                           wall_hours=(time.time() - args.start_wall) / 3600)
                append_jsonl(args.output / "train.jsonl", row)
                if update % 10 == 0 or update <= start_update + 3:
                    print(f"u{update} loss {row['loss']:.5f} gn {row['grad_norm']:.3f} lr {row['lr']:.2e} "
                          f"{row['seconds']:.1f}s peak {row['peak_gib']:.1f}GiB "
                          f"tbin {[round(x, 4) for x in row['t_bin_mse']]}", flush=True)
                if update % 25 == 0:
                    write_json(status_path, {"state": "training", "update": update, "last_loss": row["loss"],
                                             "max_updates": args.max_updates})
                if args.probe_every and update % args.probe_every == 0:
                    probe = run_probe(system, data, args)
                    append_jsonl(args.output / "probes.jsonl", dict(update=update, raw=probe))
                    print(f"[probe u{update}] latent RMSE {probe['latent_rmse']}", flush=True)
                milestone = args.eval_every and update % args.eval_every == 0 and update < args.max_updates
                if milestone or time.monotonic() >= next_save:
                    save_latest(update)
                    next_save = time.monotonic() + args.save_minutes * 60
                if milestone:
                    for weights in milestone_weights:
                        if (update, weights) not in evals_done(args.output):
                            row = run_milestone_eval(system, ema, vae, data, args, update, weights)
                            passed = passed or (bool(args.stop_at_pass) and row["passed"])
            if stop["flag"]:
                save_latest(update)
                write_json(status_path, {"state": "interrupted", "update": update, "latest": str(latest)})
                print(f"INTERRUPTED at update {update}; rerun the same command to resume", flush=True)
                raise SystemExit(143)
            save_latest(update, final=True)
        print(f"TRAINING_COMPLETE updates={update} passed_early={passed}; final evaluations {final_weights}", flush=True)
        done = evals_done(args.output)
        for weights in final_weights:
            if (update, weights) not in done:
                run_milestone_eval(system, ema, vae, data, args, update, weights)
        final = args.output / "final.pt"
        latest.replace(final)
        write_json(status_path, {"state": "complete", "update": update, "final": str(final), "passed_early": passed})
        print(f"JOB_COMPLETE updates={update} final={final}", flush=True)
    except SystemExit:
        raise
    except Exception as error:
        write_json(status_path, {"state": "failed", "update": update, "error": repr(error),
                                 "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
