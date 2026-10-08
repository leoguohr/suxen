#!/usr/bin/env python3
"""Supervised fixed-set Vertex overfit with deterministic diagnostics and resume."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_nexus.data_2k import Nexus2KManifestDataset, collate_nexus2k_samples
from mini_nexus.training import VertexStageSystem
from mini_nexus.vertex_evaluation import (
    cell_metrics, chamfer_mean_distance, denoise_level, generate_cells, trace_denoising,
)


MODEL_FIELDS = ("hidden_dim", "condition_dim", "condition_tokens", "num_layers",
                "num_heads", "condition_heads", "condition_layers", "max_depth")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--generation-steps", type=int, default=20)
    parser.add_argument("--max-generated-parents", type=int, default=10000)
    defaults = dict(hidden_dim=1536, condition_dim=2048, condition_tokens=1024,
                    num_layers=36, num_heads=12, condition_heads=16,
                    condition_layers=8, max_depth=9)
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    args = parser.parse_args()
    for name in ("steps", "eval_every", "expected_samples", "generation_steps",
                 "max_generated_parents", *MODEL_FIELDS):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not 1 <= args.max_depth <= 9 or args.warmup_steps < 0 or args.learning_rate <= 0:
        parser.error("invalid depth, warmup or learning rate")
    return args


def write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path: Path, payload):
    with path.open("a") as handle:
        handle.write(json.dumps(payload, allow_nan=False) + "\n")


def pair_at_step(step: int, samples: int, depths: int, seed: int) -> tuple[int, int]:
    """Each shuffled epoch covers every object/depth pair exactly once."""
    epoch, offset = divmod(step, samples * depths)
    order = torch.randperm(samples * depths, generator=torch.Generator().manual_seed(seed + epoch))
    pair = int(order[offset])
    return pair // depths, pair % depths + 1


def learning_rate_at_step(step: int, target: float, warmup: int) -> float:
    return target * min((step + 1) / max(warmup, 1), 1.0)


def amp(args):
    return torch.autocast(args.device, dtype=torch.bfloat16) if args.precision == "bf16" else contextlib.nullcontext()


def gradient_norm(module):
    values = [parameter.grad.norm() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def summarize(rows):
    result = {}
    for t in sorted({row["time"] for row in rows}):
        selected = [row for row in rows if row["time"] == t]
        result[str(t)] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("velocity_mse", "precision", "recall", "f1", "iou")
        }
    return result


@torch.no_grad()
def evaluate(args, model, samples, step: int, *, final: bool, status, should_stop=lambda: False):
    model.eval()
    times = (0.1, 0.5, 0.9) if final else (0.1, 0.5)
    probe_index = max(range(len(samples)), key=lambda i: len(samples[i].octree_levels[args.max_depth-1].parent_codes))
    activation_probe = None
    with amp(args):
        contexts = [model.condition_encoder(sample.condition[None].to(args.device)) for sample in samples]
        rows, swapped, generated = [], [], []
        for index, (sample, context) in enumerate(zip(samples, contexts)):
            if should_stop():
                model.train()
                return
            status("evaluating", step, uid=sample.uid, object_index=index, final=final)
            for level in sample.octree_levels[:args.max_depth]:
                if should_stop():
                    model.train()
                    return
                seed = args.seed + 100000 + index * 100 + level.depth
                for t in times:
                    row = denoise_level(model, level, context, time=t, seed=seed)
                    rows.append({"uid": sample.uid, **row})
                if final and len(samples) > 1:
                    row = denoise_level(model, level, contexts[(index + 1) % len(samples)], time=0.1, seed=seed)
                    swapped.append({"uid": sample.uid, "condition_uid": samples[(index + 1) % len(samples)].uid, **row})
                if index == probe_index and level.depth == args.max_depth:
                    activation_probe = {"uid": sample.uid,
                        **trace_denoising(model, level, context, time=0.1, seed=seed)}
                    if len(samples) > 1:
                        activation_probe["swapped_condition_uid"] = samples[(index + 1) % len(samples)].uid
                        activation_probe["swapped_denoising"] = denoise_level(
                            model, level, contexts[(index + 1) % len(samples)], time=0.1, seed=seed)
            if final:
                status("generating", step, uid=sample.uid, object_index=index)
                cells, detail = generate_cells(model, context, steps=args.generation_steps,
                                              seed=args.seed + 200000 + index,
                                              max_parents=args.max_generated_parents)
                target = torch.unique(sample.quantized_vertices >> (9 - args.max_depth), dim=0).to(args.device)
                metrics = {} if cells is None else {
                    **cell_metrics(cells, target),
                    "chamfer_mean_euclidean_normalized": chamfer_mean_distance(cells, target, args.max_depth),
                }
                generated.append({"uid": sample.uid, **detail, **metrics})
                directory = args.output / f"generation-{step:06d}"
                directory.mkdir(exist_ok=True)
                np.savez_compressed(directory / f"{sample.uid}.npz",
                    target=target.cpu().numpy(),
                    predicted=np.empty((0, 3), dtype=np.int64) if cells is None else cells.cpu().numpy(),
                    status=detail["status"], depth=args.max_depth)
                write_json(directory / "progress.json", generated)
    report = {"step": step, "final": final, "seed": args.seed,
              "denoising_is_not_generation": True,
              "sampler": "Euler engineering baseline; not paper DPM-Solver",
              "generation_steps": args.generation_steps,
              "max_generated_parents": args.max_generated_parents,
              "summary_by_time": summarize(rows), "per_object_depth_time": rows,
              "activation_probe": activation_probe,
              "swapped_summary_by_time": summarize(swapped),
              "swapped_condition": swapped, "generation": generated}
    write_json(args.output / f"evaluation-{step:06d}.json", report)
    append_json(args.output / "evaluations.jsonl", {"step": step, "final": final,
                "summary_by_time": report["summary_by_time"],
                "swapped_summary_by_time": report["swapped_summary_by_time"],
                "generation_statuses": [row["status"] for row in generated]})
    model.train()
    return report


def save_checkpoint(path, model, optimizer, args, step, manifest_hash):
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "manifest_sha256": manifest_hash,
                "architecture": model.architecture,
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}, temporary)
    temporary.replace(path)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume is None and (args.output / "train.jsonl").exists():
        raise ValueError("refusing to overwrite an existing run without --resume")
    stopped = False

    def stop_handler(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    def status(state, step, **extra):
        write_json(args.output / "status.json", {"state": state, "step": step,
                   "pid": os.getpid(), "time": time.time(), **extra})

    step = 0
    try:
        status("loading", step)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the selected device")
        dataset = Nexus2KManifestDataset(args.manifest, "train")
        all_rows = list(csv.DictReader(args.manifest.open()))
        if len(dataset) != args.expected_samples or len(all_rows) != args.expected_samples:
            raise ValueError("overfit manifest must contain exactly the expected training objects")
        samples = [dataset[index] for index in range(len(dataset))]
        model_args = {key: getattr(args, key) for key in MODEL_FIELDS}
        checkpoint = None
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
            if checkpoint.get("architecture") != VertexStageSystem.architecture or checkpoint.get("manifest_sha256") != dataset.manifest_sha256:
                raise ValueError("resume architecture or manifest mismatch")
            for key in (*MODEL_FIELDS, "seed", "learning_rate", "warmup_steps", "weight_decay", "precision"):
                if checkpoint["args"][key] != getattr(args, key):
                    raise ValueError(f"resume configuration mismatch: {key}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(8)
        model = VertexStageSystem(**model_args, use_checkpoint=True).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                     weight_decay=args.weight_decay, foreach=False)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = int(checkpoint["step"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if args.device == "cuda": torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            random.setstate(checkpoint["python_rng"])
            np.random.set_state(checkpoint["numpy_rng"])
            del checkpoint
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(architecture=model.architecture, manifest_sha256=dataset.manifest_sha256,
            parameter_count=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
            gpu=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            uids=list(dataset.uids), activation_checkpointing=True,
            sampling="Euler baseline, no forced children or top-k repair",
            source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted((ROOT / "mini_nexus").glob("*.py"))})
        config["source_sha256"]["scripts/train_vertex_overfit.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(args.output / ("resume_config.json" if args.resume else "config.json"), config)
        if step == 0:
            evaluate(args, model, samples, 0, final=False, status=status, should_stop=lambda: stopped)
        for index in range(step, args.steps):
            if stopped: break
            object_index, depth = pair_at_step(index, len(samples), args.max_depth, args.seed)
            sample = samples[object_index]
            batch = collate_nexus2k_samples([sample]).to(args.device)
            lr = learning_rate_at_step(index, args.learning_rate, args.warmup_steps)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            if args.device == "cuda": torch.cuda.reset_peak_memory_stats()
            status("training", index, uid=sample.uid, depth=depth)
            started = time.monotonic()
            with amp(args): loss = model(batch.condition, batch.octree_levels[depth - 1])
            if not torch.isfinite(loss): raise FloatingPointError("non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm): raise FloatingPointError("non-finite training gradient")
            record = {"step": index + 1, "uid": sample.uid, "depth": depth,
                      "loss": float(loss), "gradient_norm_before_clip": float(norm), "lr": lr}
            if index < 3 or (index + 1) % args.eval_every == 0:
                record["vecset_gradient_norm_after_clip"] = gradient_norm(model.condition_encoder)
                record["dit_gradient_norm_after_clip"] = gradient_norm(model.flow)
            optimizer.step()
            if args.device == "cuda": torch.cuda.synchronize()
            record.update(seconds=time.monotonic() - started,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None)
            append_json(args.output / "train.jsonl", record)
            print(json.dumps(record), flush=True)
            step = index + 1
            optimizer.zero_grad(set_to_none=True)
            if step % args.eval_every == 0 or step == args.steps or stopped:
                status("saving", step)
                save_checkpoint(args.output / "checkpoint-last.pt", model, optimizer, args, step, dataset.manifest_sha256)
                if not stopped:
                    evaluate(args, model, samples, step, final=step == args.steps, status=status, should_stop=lambda: stopped)
        if stopped:
            status("saving", step)
            save_checkpoint(args.output / "checkpoint-last.pt", model, optimizer, args, step, dataset.manifest_sha256)
        status("stopped" if stopped else "complete", step)
    except Exception as error:
        write_json(args.output / "failure.json", {"step": step, "error_type": type(error).__name__,
                   "error": str(error), "traceback": traceback.format_exc(), "time": time.time()})
        status("failed", step)
        raise


if __name__ == "__main__":
    main()
