"""T2 continuation adapter: whole-object data parallelism, original scientific implementation.

Launch with torchrun. Original train.py/model.py/data.py/evaluation remain unchanged. One
optimizer update still contains every fixed-50 object; SUM gradients precedes clipping.
Checkpoint generations commit through latest.json after fsync and SHA256 verification.
Only this output directory's superseded generation is removed after the pointer commits.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
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
import torch.distributed as dist

from benchmark_distributed import assignments, distributed_update, synthetic_data, token_indices


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--vae", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--init-vertex", type=Path)
    p.add_argument("--init-vertex-sha")
    p.add_argument("--init-checkpoint", type=Path,
                   help="provenance for an already-started single-GPU cooldown; migration.json required")
    p.add_argument("--init-checkpoint-sha")
    p.add_argument("--init-optimizer", type=int, default=1)
    p.add_argument("--ema-from-raw", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=0.)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--max-updates", type=int, default=10000)
    p.add_argument("--decay-fraction", type=float, default=.2)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--clip", type=float, default=1.)
    p.add_argument("--ema-decay", type=float, default=.999)
    p.add_argument("--target", choices=("posterior", "mu"), default="posterior")
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--init-seed", type=int, default=20261015)
    p.add_argument("--chunk-tokens", type=int, default=0)
    p.add_argument("--checkpoint-activations", type=int, default=1)
    p.add_argument("--checkpoint-flow-layers", type=int, default=None,
                   help="first N Flow blocks recompute; omitted means all Flow blocks")
    p.add_argument("--attention", choices=("auto", "flash", "sdpa"), default="auto")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-seeds", default="0,1")
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--milestone-weights", default="raw,ema")
    p.add_argument("--final-weights", default="raw,ema")
    p.add_argument("--pass-f1", type=float, default=.99)
    p.add_argument("--stop-at-pass", type=int, default=1)
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--save-minutes", type=float, default=60.)
    p.add_argument("--stop-after-updates", type=int, default=0,
                   help="bound this invocation, save, exit; does not change max-updates or LR")
    p.add_argument("--bucket-mib", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-step-seconds", type=float, default=30.,
                   help="save and stop if an effective update exceeds this bound (at most 30 s)")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--tiny", action="store_true", help="synthetic 50-object CPU smoke only; no evaluator")
    args = p.parse_args()
    if args.tiny:
        if args.device != "cpu" or args.eval_every or args.probe_every or args.final_weights:
            p.error("tiny requires --device cpu --eval-every 0 --probe-every 0 --final-weights ''")
    elif not args.cache or not args.vae or args.device != "cuda":
        p.error("real T1 requires --cache --vae and CUDA")
    if args.max_updates < 1 or args.stop_after_updates < 0:
        p.error("invalid update limit")
    if not 0 < args.max_step_seconds <= 30:
        p.error("--max-step-seconds must be positive and at most 30")
    return args


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes(source, sha256_file):
    files = {str(p.relative_to(source)): sha256_file(p) for p in sorted(source.rglob("*.py"))
             if not p.name.startswith("._")}
    for name in ("train_distributed.py", "benchmark_distributed.py", "checkpoint_policy.py", "single_gpu_migration.py"):
        files["execution_adapter/" + name] = sha256_file(Path(__file__).with_name(name))
    return files


def recipe(args, world, source, common):
    keys = ("lr", "min_lr", "warmup", "max_updates", "decay_fraction", "weight_decay", "clip",
            "ema_decay", "target", "seed", "init_seed", "chunk_tokens", "checkpoint_activations", "checkpoint_flow_layers",
            "attention", "eval_every", "eval_seeds", "eval_steps", "milestone_weights", "final_weights",
            "pass_f1", "stop_at_pass", "probe_every", "tiny", "device", "init_checkpoint_sha",
            "init_optimizer", "ema_from_raw")
    migration_path = args.output / "migration.json"
    migration_sha = canonical_sha(json.loads(migration_path.read_text())) if migration_path.exists() else None
    return dict(parameters={key: getattr(args, key) for key in keys}, world_size=world,
                parent_migration_sha256=migration_sha,
                source_sha256=source, torch=torch.__version__,
                init_vertex_sha256=args.init_vertex_sha if args.init_vertex else None,
                init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,
                cache_manifest_sha256=common.CACHE_MANIFEST_SHA256 if not args.tiny else "synthetic50",
                normalization_sha256=common.NORMALIZATION_SHA256 if not args.tiny else "synthetic",
                vae_sha256=common.VAE_SHA256 if not args.tiny else None)


def rank0_call(rank, control, callback):
    result = [None]
    if rank == 0:
        try:
            result[0] = dict(ok=True, value=callback())
        except BaseException as error:
            result[0] = dict(ok=False, error=repr(error), traceback=traceback.format_exc())
    dist.broadcast_object_list(result, src=0, group=control)
    if not result[0]["ok"]:
        raise RuntimeError(result[0]["error"] + "\n" + result[0]["traceback"])
    return result[0]["value"]


def rng_state(device):
    return dict(torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
                numpy=np.random.get_state(), python=random.getstate())


def restore_rng(state, device):
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def verify_pointer(output, name, sha256_file):
    pointer = json.loads((output / name).read_text())
    path = (output / pointer["file"]).resolve()
    if path.parent != output.resolve() or not path.name.startswith("checkpoint-u"):
        raise ValueError("checkpoint pointer escapes this experiment")
    if path.stat().st_size != pointer["bytes"] or sha256_file(path) != pointer["sha256"]:
        raise ValueError(f"checkpoint size/SHA256 mismatch: {path}")
    return pointer


def fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import common
    import data as data_module
    import model as model_module
    import train
    from single_gpu_migration import verify_migration, validate_legacy_checkpoint

    args.init_vertex_sha = args.init_vertex_sha or common.S0_SHA256
    cfg = model_module.TINY if args.tiny else model_module.PAPER
    if args.checkpoint_flow_layers is None:
        args.checkpoint_flow_layers = cfg.num_layers
    args.eval_seed_list = [int(s) for s in args.eval_seeds.split(",") if s]
    args.start_wall = time.time()
    rank, world, local_rank = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if args.device == "cuda" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=8))
    # A milestone may legitimately evaluate for hours; its waiting ranks have no pending NCCL work.
    control = dist.new_group(backend="gloo", timeout=timedelta(days=3))
    torch.set_num_threads(args.threads)
    random.seed(args.init_seed)
    np.random.seed(args.init_seed % 2**32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    immutable = recipe(args, world, source_hashes(args.source, common.sha256_file), common)
    immutable_sha = canonical_sha(immutable)
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    output_lock = None

    def inspect_output():
        nonlocal output_lock
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_lock = (args.output.parent / (args.output.name + ".lock")).open("a+")
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=True)
        if (args.output / "STOP").exists():
            return dict(complete=False, stopped=True, pointer=None)
        if (args.output / "final.json").exists():
            pointer = verify_pointer(args.output, "final.json", common.sha256_file)
            if pointer["recipe_sha256"] != immutable_sha:
                raise ValueError("completed output belongs to a different recipe")
            status_path = args.output / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("state") != "complete":
                status.update(state="complete", update=pointer["update"], final=pointer,
                              recovered_committed_final=True)
                common.write_json(status_path, status)
            return dict(complete=True, pointer=pointer)
        if (args.output / "latest.json").exists():
            pointer = verify_pointer(args.output, "latest.json", common.sha256_file)
            if pointer["recipe_sha256"] != immutable_sha:
                raise ValueError("resume recipe/source/world size differs")
            return dict(complete=False, pointer=pointer)
        if (args.output / "migration.json").exists():
            migration = verify_migration(args.output, common.sha256_file)
            if migration["target_world_size"] != world:
                raise ValueError("migration target world size differs")
            research_source = {k: v for k, v in immutable["source_sha256"].items()
                               if not k.startswith("execution_adapter/")}
            if migration["source_files"] != research_source:
                raise ValueError("migration research source differs")
            common.write_json(args.output / "recipe.json", dict(recipe=immutable, sha256=immutable_sha))
            return dict(complete=False, pointer=None, migration=migration)
        if any(args.output.iterdir()):
            raise FileExistsError("output is nonempty without a verified latest.json; nothing was moved or deleted")
        if args.init_checkpoint is not None:
            raise ValueError("T2 cooldown continuation requires a verified migration.json; fresh init is unsupported")
        common.write_json(args.output / "recipe.json", dict(recipe=immutable, sha256=immutable_sha))
        return dict(complete=False, pointer=None)

    try:
        startup = rank0_call(rank, control, inspect_output)
        if startup.get("stopped"):
            if rank == 0:
                print("STOP_FILE_PRESENT; remove STOP explicitly before resuming", flush=True)
            return
        if startup["complete"]:
            if rank == 0:
                print("ALREADY_COMPLETE " + str(args.output), flush=True)
            return
        cfg = model_module.TINY if args.tiny else model_module.PAPER
        data = (synthetic_data(data_module, cfg, device) if args.tiny
                else data_module.load_cache(args.cache, device))
        if not args.tiny:
            data.compute_fps(cfg.condition_tokens)
        system = train.build_system(args, device)
        from checkpoint_policy import bind_checkpoint_policy
        bind_checkpoint_policy(system, args.checkpoint_flow_layers)
        transfer = None
        resuming = bool(startup["pointer"] or startup.get("migration"))
        if args.init_vertex and not resuming:
            if common.sha256_file(args.init_vertex) != args.init_vertex_sha:
                raise ValueError("S0 SHA256 mismatch")
            s0 = torch.load(args.init_vertex, map_location="cpu", mmap=True, weights_only=False)
            transfer = model_module.transfer_vertex_weights(system, s0["model"])
            del s0
        system.train()
        optimizer = torch.optim.AdamW(system.parameters(), lr=args.lr, betas=(.9, .999), eps=1e-8,
                                      weight_decay=args.weight_decay, fused=device.type == "cuda")
        ema = train.EMA(system, args.ema_decay)
        update, final_stage, resume_rng, saved_passed = 0, False, None, False
        if startup["pointer"]:
            state = torch.load(args.output / startup["pointer"]["file"], map_location="cpu", mmap=True,
                               weights_only=False)
            if state["distributed"]["recipe_sha256"] != immutable_sha:
                raise ValueError("checkpoint payload recipe disagrees with commit pointer")
            system.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            ema.load(system, state["ema"])
            update, final_stage = int(state["update"]), bool(state["final_stage"])
            if update != startup["pointer"]["update"]:
                raise ValueError("checkpoint update disagrees with commit pointer")
            resume_rng = state["distributed"]["rng_by_rank"][rank]
            saved_passed = bool(state["distributed"].get("passed_early", False))
            del state
        elif startup.get("migration"):
            migration = startup["migration"]
            source_config = json.loads(Path(migration["source_config"]["file"]).read_text())
            if migration["data_cursor"]["uids"] != data.uids:
                raise ValueError("migration data UID order differs")
            state = torch.load(migration["parent"]["file"], map_location="cpu", mmap=True, weights_only=False)
            validate_legacy_checkpoint(state, args, cfg, migration, source_config, data=data)
            system.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            ema.load(system, state["ema"])
            update, final_stage = int(state["update"]), bool(state["final_stage"])
            del state
            if rank == 0:
                print(f"LEGACY_IMPORTED update={update} next_update={update + 1}; "
                      "model/EMA/Adam retained; input stream keyed by seed/update; "
                      "old global RNG unavailable, new rank RNG states will be checkpointed", flush=True)
        vae = train.load_frozen_vae(args.vae, device) if rank == 0 and not args.tiny else None
        partition = assignments(data.lengths, world)
        objects = torch.tensor(partition[rank], device=device)
        indices = token_indices(data, partition[rank])
        shard = data.select(partition[rank])
        groups = train.object_groups(shard, args.chunk_tokens)
        if resume_rng is not None:
            # Decoder construction must finish before restoring the exact saved random streams.
            restore_rng(resume_rng, device)
        start_update = update
        info = dict(world_size=world, partition=partition, tokens=data.seg.tokens, uids=data.uids,
                    parameters=model_module.parameter_count(system), transfer=transfer, start_update=update,
                    resume=resuming, recipe_sha256=immutable_sha,
                    args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        rank0_call(rank, control, lambda: common.write_json(args.output / f"config-u{update:06d}.json", info))
        if rank == 0:
            print(f"READY world={world} update={update} tokens={data.seg.tokens} recipe={immutable_sha}", flush=True)
        latest = startup["pointer"]

        def save_checkpoint(is_final_stage=False):
            nonlocal latest
            rngs = [None] * world
            dist.all_gather_object(rngs, rng_state(device), group=control)

            def write_checkpoint():
                name = f"checkpoint-u{update:06d}-{'finalstage' if is_final_stage else 'train'}.pt"
                destination = args.output / name
                common.write_json(args.output / "status.json", dict(state="saving", update=update))
                payload = train.checkpoint_payload(system, ema, optimizer, args, update, is_final_stage)
                payload["distributed"] = dict(recipe_sha256=immutable_sha, recipe=immutable,
                                              rng_by_rank=rngs, world_size=world, passed_early=passed,
                                              data_cursor=dict(next_update=update + 1, objects_per_update=data.count,
                                                               uids=data.uids, stateless_batch_seed=args.seed))
                if (args.output / "migration.json").exists():
                    payload["distributed"]["parent_migration"] = json.loads((args.output / "migration.json").read_text())
                common.save_torch(destination, payload)
                del payload
                digest = common.sha256_file(destination)
                pointer = dict(file=name, bytes=destination.stat().st_size, sha256=digest, update=update,
                               final_stage=is_final_stage, recipe_sha256=immutable_sha)
                common.write_json(args.output / "latest.json", pointer)
                fsync_directory(args.output)
                # Only this run's previously committed, superseded, non-final checkpoint is eligible.
                if latest and latest["file"] != name and not (args.output / "final.json").exists():
                    old = (args.output / latest["file"]).resolve()
                    if old.parent != args.output.resolve() or not old.name.startswith("checkpoint-u"):
                        raise ValueError("refusing checkpoint cleanup outside this run")
                    old.unlink()
                    fsync_directory(args.output)
                print(f"CHECKPOINT_COMMITTED update={update} file={name} sha256={digest}", flush=True)
                return pointer

            latest = rank0_call(rank, control, write_checkpoint)

        def evaluate(weights_list):
            def on_rank0():
                passed_here = False
                done = train.evals_done(args.output)
                for weights in weights_list:
                    if stop["flag"] or (args.output / "STOP").exists():
                        break
                    if (update, weights) not in done:
                        row = train.run_milestone_eval(system, ema, vae, data, args, update, weights)
                        passed_here = passed_here or (bool(args.stop_at_pass) and row["passed"])
                    else:
                        for line in (args.output / "evals.jsonl").read_text().splitlines():
                            row = json.loads(line)
                            if row["update"] == update and row["weights"] == weights:
                                passed_here = passed_here or (bool(args.stop_at_pass) and row["passed"])
                return dict(passed=passed_here, stopped=stop["flag"] or (args.output / "STOP").exists())
            result = rank0_call(rank, control, on_rank0)
            stop["flag"] = stop["flag"] or result["stopped"]
            return result["passed"]

        def stop_requested():
            request = torch.tensor(int(stop["flag"] or (rank == 0 and (args.output / "STOP").exists())), device=device)
            dist.all_reduce(request, op=dist.ReduceOp.MAX)
            return bool(request)

        def exit_interrupted():
            if latest is None or latest["update"] != update:
                save_checkpoint(is_final_stage=final_stage)
            rank0_call(rank, control, lambda: common.write_json(args.output / "status.json",
                       dict(state="interrupted", update=update, latest=latest)))
            if rank == 0:
                print(f"INTERRUPTED update={update}; resume after removing STOP", flush=True)
            raise SystemExit(143)

        def probe_if_due():
            if not args.probe_every or not update or update % args.probe_every:
                return
            def probe():
                path = args.output / "probes.jsonl"
                if path.exists() and any(json.loads(line)["update"] == update for line in path.read_text().splitlines()):
                    return None
                result = train.run_probe(system, data, args)
                common.append_jsonl(path, dict(update=update, raw=result))
                return result
            rank0_call(rank, control, probe)

        passed = saved_passed
        if startup.get("migration"):
            # Commit a real native resume point before the first post-import update.
            # Its RNG states belong to this distributed launch, never to the old run.
            save_checkpoint(is_final_stage=final_stage)
        milestone_weights = [w for w in args.milestone_weights.split(",") if w]
        final_weights = [w for w in args.final_weights.split(",") if w]
        next_save = time.monotonic() + args.save_minutes * 60
        if resuming and not final_stage:
            if stop_requested():
                exit_interrupted()
            probe_if_due()
        if (not final_stage and resuming and args.eval_every and update
                and update % args.eval_every == 0 and update < args.max_updates):
            passed = evaluate(milestone_weights)
        while update < args.max_updates and not final_stage and not passed:
            if stop_requested():
                exit_interrupted()
            update += 1
            row = distributed_update(system, optimizer, ema, data, shard, groups, indices, objects,
                                     args, update, train)
            rank_seconds = torch.tensor(row["seconds"], device=device)
            dist.all_reduce(rank_seconds, op=dist.ReduceOp.MAX)
            row["seconds"] = float(rank_seconds)
            row["wall_hours"] = (time.time() - args.start_wall) / 3600
            if rank == 0:
                common.append_jsonl(args.output / "train.jsonl", row)
                if update % 10 == 0 or update <= start_update + 3:
                    print(f"u{update} loss={row['loss']:.6f} lr={row['lr']:.3g} seconds={row['seconds']:.3f}", flush=True)
            requested_stop = torch.tensor(int(stop["flag"] or (rank == 0 and (args.output / "STOP").exists())),
                                          device=device)
            dist.all_reduce(requested_stop, op=dist.ReduceOp.MAX)
            bounded_stop = bool(args.stop_after_updates and update - start_update >= args.stop_after_updates)
            speed_stop = row["seconds"] > args.max_step_seconds
            milestone = bool(args.eval_every and update % args.eval_every == 0 and update < args.max_updates)
            save_due = rank0_call(rank, control, lambda: time.monotonic() >= next_save)
            if requested_stop or bounded_stop or speed_stop or milestone or save_due:
                save_checkpoint()
                next_save = time.monotonic() + args.save_minutes * 60
            if requested_stop or bounded_stop or speed_stop:
                state_name = ("speed_limit_exceeded" if speed_stop else
                              "interrupted" if requested_stop else "bounded_stopped")
                rank0_call(rank, control, lambda: common.write_json(args.output / "status.json",
                           dict(state=state_name, update=update, latest=latest,
                                update_seconds=row["seconds"], max_step_seconds=args.max_step_seconds)))
                if rank == 0:
                    print(f"{state_name.upper()} update={update}; resume the same recipe", flush=True)
                if requested_stop:
                    raise SystemExit(143)
                if speed_stop:
                    raise SystemExit(78)
                return
            probe_if_due()
            if milestone:
                passed = evaluate(milestone_weights)
        if not final_stage:
            if stop_requested():
                exit_interrupted()
            save_checkpoint(is_final_stage=True)
        evaluate(final_weights)
        if stop_requested():
            exit_interrupted()

        def finish():
            common.write_json(args.output / "final.json", latest)
            fsync_directory(args.output)
            common.write_json(args.output / "status.json", dict(state="complete", update=update,
                               final=latest, passed_early=passed))
            print(f"JOB_COMPLETE update={update} final={latest['file']} sha256={latest['sha256']}", flush=True)
        rank0_call(rank, control, finish)
    except SystemExit:
        raise
    except BaseException as error:
        if rank == 0 and args.output.is_dir():
            common.write_json(args.output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if output_lock is not None:
            output_lock.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
