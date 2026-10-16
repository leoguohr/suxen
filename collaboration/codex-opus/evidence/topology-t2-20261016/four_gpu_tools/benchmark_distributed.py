"""Disposable T1 full-update throughput probe; does not replace or edit train.py.

Example: torchrun --standalone --nproc_per_node=2 benchmark_distributed.py \
  --source /path/to/flow_overfit50_20261015 --cache /path/to/cache --output /path/to/probe

Every rank draws the original full batch with train.make_batch, selects disjoint complete
objects, divides its SSE by the GLOBAL token/channel count, then SUM-reduces gradients
before the original clipping, AdamW and EMA operations. No model or optimizer is saved.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist


def assignments(lengths, world):
    """LPT on token count plus fixed condition-encoder work; partition only whole meshes."""
    loads, result = [0] * world, [[] for _ in range(world)]
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        rank = min(range(world), key=lambda r: (loads[r], r))
        result[rank].append(index)
        loads[rank] += lengths[index] + 1024
    return [sorted(group) for group in result]


def token_indices(data, objects):
    return torch.cat([torch.arange(*data.seg.bounds[i], device=data.device) for i in objects])


def sum_gradients(model, bucket_bytes):
    """Bounded-size FP32 buffers. SUM is required because local losses use a global divisor."""
    if dist.get_world_size() == 1:
        return 0
    bucket, size, reduced = [], 0, 0

    def flush():
        nonlocal reduced
        if not bucket:
            return
        flat = torch.cat([grad.reshape(-1) for grad in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for grad in bucket:
            grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
            offset += grad.numel()
        reduced += flat.numel() * flat.element_size()

    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError("unexpected unused parameter; collective order would be ambiguous")
        grad = parameter.grad
        nbytes = grad.numel() * grad.element_size()
        if bucket and size + nbytes > bucket_bytes:
            flush()
            bucket, size = [], 0
        bucket.append(grad)
        size += nbytes
    flush()
    return reduced


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def distributed_update(system, optimizer, ema, data, shard, groups, indices, objects, args, update, train):
    device = data.device
    dist.barrier()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    t, noisy, target = train.make_batch(data, args.seed, update, args.target)
    denominator = float(noisy.numel())
    local_noisy, local_target, local_t = noisy[indices], target[indices], t[objects]
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    sse = torch.zeros(data.count, device=device)
    synchronize(device)
    batch_finished = time.perf_counter()
    for group in groups:
        sub, sl = group["data"], group["tokens"]
        ids = torch.tensor(group["objects"], device=device)
        with train.autocast(device):
            condition = system.encode_condition(sub.points, sub.point_valid, sub.fps_indices,
                                                use_checkpoint=bool(args.checkpoint_activations))
            prediction = system.flow.forward_packed(local_noisy[sl], local_t[ids], sub.vertices,
                                                    sub.seg, condition, args.attention,
                                                    bool(args.checkpoint_activations))
        error = (prediction.float() - local_target[sl]).square()
        loss = error.sum() / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss; benchmark not stepped")
        loss.backward()
        total += loss.detach()
        sse.index_add_(0, objects[ids[sub.seg.index]], error.detach().sum(-1))
    synchronize(device)
    backward_finished = time.perf_counter()
    gradient_bytes = sum_gradients(system, args.bucket_mib * 2**20)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(sse, op=dist.ReduceOp.SUM)
    synchronize(device)
    communication_finished = time.perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(system.parameters(), args.clip, error_if_nonfinite=True)
    lr = train.lr_at(args, update)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema.update()
    synchronize(device)
    finished = time.perf_counter()
    lengths = torch.tensor(data.lengths, device=device, dtype=torch.float32)
    per_object = sse / (lengths * noisy.shape[1])
    bins = torch.clamp((t * train.T_BINS).long(), max=train.T_BINS - 1)
    bin_sse = torch.zeros(train.T_BINS, device=device).index_add_(0, bins, sse)
    bin_tokens = torch.zeros(train.T_BINS, device=device).index_add_(0, bins, lengths * noisy.shape[1])
    return dict(update=update, loss=float(total), grad_norm=float(norm), lr=lr,
                t_bin_mse=(bin_sse / bin_tokens.clamp_min(1)).tolist(),
                largest_object_mse=float(per_object[int(np.argmax(data.lengths))]),
                smallest_object_mse=float(per_object[int(np.argmin(data.lengths))]),
                seconds=finished - started, batch_seconds=batch_finished - started,
                forward_backward_seconds=backward_finished - batch_finished,
                gradient_sync_seconds=communication_finished - backward_finished,
                clip_adam_ema_seconds=finished - communication_finished,
                gradient_payload_bytes=gradient_bytes, local_objects=shard.count,
                local_tokens=shard.seg.tokens,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None)


def synthetic_data(data_module, config, device):
    rng = np.random.default_rng(109)
    records = []
    for i in range(50):
        n = 4 + (i * 7) % 23
        points = rng.normal(size=(32, 6)).astype(np.float32)
        records.append(dict(uid=f"synthetic-{i:02d}", vertices=rng.uniform(-1, 1, (n, 3)).astype(np.float32),
                            mu=rng.normal(size=(n, config.latent_dim)).astype(np.float32),
                            logvar=np.full((n, config.latent_dim), -6, dtype=np.float32), points=points,
                            point_mask=np.ones(32, dtype=bool), edges=np.zeros((0, 2), dtype=np.int64),
                            faces=np.zeros((0, 3), dtype=np.int64)))
    result = data_module.TopologyData(records, torch.zeros(config.latent_dim),
                                     torch.ones(config.latent_dim), device)
    result.compute_fps(config.condition_tokens)
    return result


def tensor_differences(actual, expected, atol, rtol):
    """Report failures before raising, including near-zero-coordinate and update-relative errors."""
    assert actual.keys() == expected.keys()
    error_sq = expected_sq = actual_sq = dot = 0.0
    details = []
    for name, a in actual.items():
        b = expected[name]
        a, b = a.detach().double(), b.detach().double()
        difference = a - b
        absolute = difference.abs()
        failed = absolute > atol + rtol * b.abs()
        index = int(absolute.flatten().argmax())
        aa, bb, dd = float(a.square().sum()), float(b.square().sum()), float(difference.square().sum())
        actual_sq += aa
        expected_sq += bb
        error_sq += dd
        dot += float((a * b).sum())
        details.append(dict(name=name, max_abs=float(absolute.max()), max_abs_flat_index=index,
                            actual_at_max_abs=float(a.flatten()[index]), expected_at_max_abs=float(b.flatten()[index]),
                            failed_elements=int(failed.sum()), elements=a.numel(),
                            relative_l2=(dd / bb) ** .5 if bb else None))
    details.sort(key=lambda item: item["max_abs"], reverse=True)
    return dict(passed=all(item["failed_elements"] == 0 for item in details), atol=atol, rtol=rtol,
                max_abs=details[0]["max_abs"], failed_elements=sum(item["failed_elements"] for item in details),
                actual_l2=actual_sq ** .5, expected_l2=expected_sq ** .5, difference_l2=error_sq ** .5,
                relative_l2=(error_sq / expected_sq) ** .5 if expected_sq else None,
                cosine=dot / (actual_sq * expected_sq) ** .5 if actual_sq * expected_sq else None,
                worst_tensors=details[:10])


def gradient_capture(model, destination):
    def capture(_optimizer, _args, _kwargs):
        destination.clear()
        destination.update({name: p.grad.detach().clone() for name, p in model.named_parameters()})
    return capture


def validate_step(system, optimizer, ema, reference, ref_optimizer, ref_ema, data, args, update, train, row,
                  actual_before, reference_before, actual_gradients, reference_gradients):
    reference_row = train.train_update(reference, ref_optimizer, data, train.object_groups(data, 0), args, update)
    ref_ema.update()
    # Keep the original acceptance tolerances. Gradient/update diagnostics do not waive failures.
    actual, expected = dict(system.named_parameters()), dict(reference.named_parameters())
    parameters = tensor_differences(actual, expected, 2e-6, 5e-5)
    ema_values = tensor_differences(dict(zip(actual, ema.shadow)), dict(zip(expected, ref_ema.shadow)), 2e-6, 5e-5)
    gradients = tensor_differences(actual_gradients, reference_gradients, 2e-7, 1e-4)
    actual_coefficient = min(1., args.clip / (row["grad_norm"] + 1e-6))
    reference_coefficient = min(1., args.clip / (reference_row["grad_norm"] + 1e-6))
    reconstructed_preclip = tensor_differences(
        {n: g / actual_coefficient for n, g in actual_gradients.items()},
        {n: g / reference_coefficient for n, g in reference_gradients.items()}, 2e-7, 1e-4)
    deltas = tensor_differences({n: p.detach() - actual_before[n] for n, p in actual.items()},
                               {n: p.detach() - reference_before[n] for n, p in expected.items()}, 2e-6, 5e-5)
    adam = {}
    for key in ("exp_avg", "exp_avg_sq", "step"):
        adam[key] = tensor_differences({n: optimizer.state[p][key] for n, p in actual.items()},
                                      {n: ref_optimizer.state[p][key] for n, p in expected.items()}, 2e-7, 1e-4)
    coordinates = []
    for worst in parameters["worst_tensors"][:5]:
        name, index = worst["name"], worst["max_abs_flat_index"]
        pa, pb = actual[name], expected[name]
        sa, sb = optimizer.state[pa], ref_optimizer.state[pb]
        # Adam's bias-corrected denominator at the coordinate with the largest weight discrepancy.
        denom_a = (sa["exp_avg_sq"].flatten()[index] / (1 - .999 ** update)).sqrt() + 1e-8
        denom_b = (sb["exp_avg_sq"].flatten()[index] / (1 - .999 ** update)).sqrt() + 1e-8
        coordinates.append(dict(name=name, flat_index=index, parameter_error=worst["max_abs"],
                                actual_gradient=float(actual_gradients[name].flatten()[index]),
                                reference_gradient=float(reference_gradients[name].flatten()[index]),
                                actual_adam_denominator=float(denom_a), reference_adam_denominator=float(denom_b),
                                actual_update=float((pa.detach() - actual_before[name]).flatten()[index]),
                                reference_update=float((pb.detach() - reference_before[name]).flatten()[index])))
    loss_error = abs(row["loss"] - reference_row["loss"])
    norm_error = abs(row["grad_norm"] - reference_row["grad_norm"])
    passed = (parameters["passed"] and ema_values["passed"] and gradients["passed"]
              and all(part["passed"] for part in adam.values())
              and loss_error <= 2e-6 + 5e-5 * abs(reference_row["loss"])
              and norm_error <= 2e-6 + 5e-5 * abs(reference_row["grad_norm"]))
    return dict(passed=passed, reference="original train.train_update, one full 50-object group",
                parameters=parameters, ema=ema_values, adam=adam, postclip_gradients=gradients,
                preclip_gradients_reconstructed=reconstructed_preclip,
                reconstruction_note="optimizer pre-step hook records postclip gradients; divide by original clip coefficient",
                parameter_updates=deltas, largest_weight_error_coordinates=coordinates,
                loss_abs_error=loss_error, grad_norm_abs_error=norm_error,
                recipe=dict(lr=args.lr, effective_lr=row["lr"], warmup=args.warmup, ema_decay=args.ema_decay))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--vae", type=Path, help="optional residency parity with original speed check")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--validate", action="store_true", help="tiny CPU only, compare each step against original trainer")
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--discard", type=int, default=3)
    parser.add_argument("--chunk-tokens", type=int, default=0)
    parser.add_argument("--checkpoint-activations", type=int, default=1)
    parser.add_argument("--checkpoint-flow-layers", type=int, default=None,
                        help="first N Flow blocks recompute; omitted means all 36 paper / 2 tiny blocks")
    parser.add_argument("--attention", choices=("auto", "flash", "sdpa"), default="auto")
    parser.add_argument("--bucket-mib", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--init-seed", type=int, default=20261015)
    parser.add_argument("--target", choices=("posterior", "mu"), default="posterior")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=0.)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--max-updates", type=int, default=10000)
    parser.add_argument("--decay-fraction", type=float, default=.2)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--clip", type=float, default=1.)
    parser.add_argument("--ema-decay", type=float, default=.999)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.validate and not (args.tiny and args.device == "cpu"):
        parser.error("--validate requires --tiny --device cpu")
    if not args.tiny and args.cache is None:
        parser.error("real data requires --cache")
    sys.path.insert(0, str(args.source.resolve()))
    import data as data_module
    import model as model_module
    import train
    from common import append_jsonl, write_json

    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=8))
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = model_module.TINY if args.tiny else model_module.PAPER
    if args.checkpoint_flow_layers is None:
        args.checkpoint_flow_layers = cfg.num_layers
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / f"rank{rank}.jsonl").exists():
        raise FileExistsError("use a new benchmark output directory")
    data = (synthetic_data(data_module, cfg, device) if args.tiny
            else data_module.load_cache(args.cache, device))
    if not args.tiny:
        data.compute_fps(cfg.condition_tokens)
    system = train.build_system(args, device)
    from checkpoint_policy import bind_checkpoint_policy
    bind_checkpoint_policy(system, args.checkpoint_flow_layers)
    if args.validate:
        with torch.no_grad():
            for p in system.parameters():
                p.add_(.05 * torch.randn_like(p))  # exercise nonzero gates/heads and all gradients
    system.train()
    optimizer = torch.optim.AdamW(system.parameters(), lr=args.lr, betas=(.9, .999), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=device.type == "cuda")
    ema = train.EMA(system, args.ema_decay)
    vae = train.load_frozen_vae(args.vae, device) if args.vae else None
    partition = assignments(data.lengths, world)
    assert sorted(i for part in partition for i in part) == list(range(data.count))
    objects = torch.tensor(partition[rank], device=device)
    indices = token_indices(data, partition[rank])
    shard = data.select(partition[rank])
    groups = train.object_groups(shard, args.chunk_tokens)
    reference = ref_optimizer = ref_ema = None
    actual_gradients, reference_gradients = {}, {}
    if args.validate and rank == 0:
        reference = copy.deepcopy(system)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=args.lr, betas=(.9, .999), eps=1e-8,
                                          weight_decay=args.weight_decay, fused=False)
        ref_ema = train.EMA(reference, args.ema_decay)
        optimizer.register_step_pre_hook(gradient_capture(system, actual_gradients))
        ref_optimizer.register_step_pre_hook(gradient_capture(reference, reference_gradients))
    config = dict(vars(args), source=str(args.source), cache=str(args.cache), vae=str(args.vae),
                  output=str(args.output), rank=rank, world=world, partition=partition,
                  total_objects=data.count, total_tokens=data.seg.tokens,
                  parameters=model_module.parameter_count(system),
                  frozen_vae_resident=vae is not None, torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    write_json(args.output / f"config-rank{rank}.json", config)
    rows = []
    try:
        for update in range(1, args.updates + 1):
            if args.validate and rank == 0:
                actual_before = {n: p.detach().clone() for n, p in system.named_parameters()}
                reference_before = {n: p.detach().clone() for n, p in reference.named_parameters()}
            row = distributed_update(system, optimizer, ema, data, shard, groups, indices, objects,
                                     args, update, train)
            if args.validate and rank == 0:
                row["validation"] = validate_step(system, optimizer, ema, reference, ref_optimizer,
                                                  ref_ema, data, args, update, train, row, actual_before,
                                                  reference_before, actual_gradients, reference_gradients)
                write_json(args.output / f"validation-u{update:03d}.json", row["validation"])
            append_jsonl(args.output / f"rank{rank}.jsonl", row)
            rows.append(row)
            if rank == 0:
                printed = {k: v for k, v in row.items() if k != "validation"}
                if args.validate:
                    printed["validation_passed"] = row["validation"]["passed"]
                print(json.dumps(dict(rank=rank, world=world, **printed)), flush=True)
                if args.validate and not row["validation"]["passed"]:
                    raise AssertionError(f"validation failed; detailed evidence saved at update {update}")
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        if rank == 0:
            wall = [max(part[u]["seconds"] for part in gathered) for u in range(args.updates)]
            steady = wall[args.discard:] or wall
            report = dict(world_size=world, updates=args.updates, discarded_updates=args.discard,
                          global_objects_per_update=data.count, global_tokens_per_update=data.seg.tokens,
                          seconds_per_update_median=statistics.median(steady),
                          global_vertex_tokens_per_second=data.seg.tokens / statistics.median(steady),
                          projected_10000_update_wall_hours=statistics.median(steady) * 10000 / 3600,
                          projected_10000_update_gpu_hours=world * statistics.median(steady) * 10000 / 3600,
                          per_rank_rows=gathered, partition=partition,
                          note="benchmark only; no checkpoint, decoder evaluation or formal training run")
            write_json(args.output / "summary.json", report)
            print("DISTRIBUTED_BENCHMARK_OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
