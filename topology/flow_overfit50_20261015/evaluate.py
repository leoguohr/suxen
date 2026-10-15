"""Standalone evaluations on the fixed 50 objects (one GPU).

  --mode mu      decode the cached posterior mean; must reproduce the archived VAE result exactly
                 (edges 154,973/125/5, faces 102,878/870/12 TP/FP/FN). This checks the evaluator.
  --mode noise   decode normalized mu + alpha * N(0, I) for each --alphas value: how much latent error
                 the frozen decoder tolerates (sets the precision target for the flow).
  --mode flow    generate with a trained checkpoint (--weights raw|ema) for every combination of
                 --seeds, --steps and --samplers, then decode and score (the chain gate).

Outputs: OUTPUT/<config>/summary.json and objects.jsonl, plus OUTPUT/results.json (all configs).
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from common import generator, write_json  # noqa: E402
from data import load_cache  # noqa: E402
from metrics import aggregate, evaluate_latents, load_frozen_vae  # noqa: E402
from model import FlowConfig, TopologyFlowSystem  # noqa: E402
from sampling import generate  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vae_mu_baseline.json"


def save(output: Path, name: str, rows: list, extra: dict) -> dict:
    summary = dict(aggregate(rows), **extra)
    write_json(output / name / "summary.json", summary)
    with (output / name / "objects.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return summary


def compare_with_fixture(rows: list) -> dict:
    fixture = json.loads(FIXTURE.read_text())
    mismatches = []
    for row in rows:
        ref = fixture["per_uid"][row["uid"]]
        mine = dict(edge={k: row["edge"][k] for k in ("tp", "fp", "fn", "tn")},
                    face={k: row["face"][k] for k in ("tp", "fp", "fn", "tn")}, candidates=row["candidates"],
                    face_fn_missing=row["face_fn_missing"], face_fn_present=row["face_fn_present"])
        if mine != ref:
            mismatches.append(dict(uid=row["uid"], ours=mine, archived=ref))
    return dict(exact_match=not mismatches, mismatched_objects=len(mismatches), mismatches=mismatches[:10],
                archived_totals=fixture["totals"])


def load_system(path: Path, weights: str, device) -> TopologyFlowSystem:
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    with torch.device(device):
        system = TopologyFlowSystem(FlowConfig(**state["config"]))
    system.load_state_dict(state["model"], strict=True)
    if weights == "ema":
        own = system.state_dict()
        system.load_state_dict({**own, **{k: v for k, v in state["ema"].items()}}, strict=True)
    del state
    return system.float().eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("mu", "noise", "flow"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--vae", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--weights", choices=("raw", "ema"), default="ema")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--steps", default="20")
    p.add_argument("--samplers", default="euler")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--attention", choices=("auto", "flash", "sdpa"), default="auto")
    p.add_argument("--alphas", default="0,0.02,0.05,0.1,0.2,0.3")
    args = p.parse_args()
    assert torch.cuda.is_available(), "evaluation runs on one GPU"
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_cache(args.cache, device)
    vae = load_frozen_vae(args.vae, device)
    results, started = {}, time.monotonic()

    if args.mode == "mu":
        rows = evaluate_latents(vae, data, data.mu, normalized=False)
        check = compare_with_fixture(rows)
        results["mu"] = save(args.output, "mu", rows, dict(fixture_check=check))
        print(f"[mu] face F1 {results['mu']['face']['micro_f1']:.6f} edge F1 {results['mu']['edge']['micro_f1']:.6f} "
              f"fixture exact match: {check['exact_match']} ({check['mismatched_objects']} objects differ)", flush=True)

    elif args.mode == "noise":
        for alpha in [float(a) for a in args.alphas.split(",")]:
            for seed in [int(s) for s in args.seeds.split(",")]:
                noise = torch.cat([torch.randn(tuple(data.mu[data.token_slice(i)].shape),
                                               generator=generator(f"topology-overfit50/tolerance/{seed}/{uid}"))
                                   for i, uid in enumerate(data.uids)]).to(device)
                latents = data.mu_normalized + alpha * noise
                rows = evaluate_latents(vae, data, latents, normalized=True)
                name = f"alpha{alpha:g}_seed{seed}"
                results[name] = save(args.output, name, rows, dict(alpha=alpha, seed=seed))
                s = results[name]
                print(f"[noise alpha={alpha:g} seed={seed}] face F1 {s['face']['micro_f1']:.6f} edge F1 "
                      f"{s['edge']['micro_f1']:.6f} strict {s['joint_strict']}/50 lower_bound={s['lower_bound']}",
                      flush=True)

    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for --mode flow")
        system = load_system(args.checkpoint, args.weights, device)
        data.compute_fps(system.cfg.condition_tokens)
        for sampler in args.samplers.split(","):
            for steps in [int(s) for s in args.steps.split(",")]:
                for seed in [int(s) for s in args.seeds.split(",")]:
                    latents = generate(system, data, seed=seed, steps=steps, sampler=sampler, dtype=args.dtype,
                                       attention=args.attention)
                    rows = evaluate_latents(vae, data, latents, normalized=True)
                    name = f"{args.weights}_{sampler}{steps}_{args.dtype}_seed{seed}"
                    results[name] = save(args.output, name, rows, dict(
                        checkpoint=str(args.checkpoint), weights=args.weights, sampler=sampler, steps=steps,
                        dtype=args.dtype, seed=seed))
                    s = results[name]
                    print(f"[{name}] face F1 {s['face']['micro_f1']:.6f} edge F1 {s['edge']['micro_f1']:.6f} "
                          f"strict {s['joint_strict']}/50 latent RMSE {s['latent_rmse_token_weighted']:.4f}", flush=True)

    write_json(args.output / "results.json", dict(mode=args.mode, minutes=(time.monotonic() - started) / 60,
                                                   results={k: {kk: vv for kk, vv in v.items() if kk != "fixture_check"}
                                                            for k, v in results.items()}))
    print("EVALUATION_DONE", flush=True)


if __name__ == "__main__":
    main()
