"""Read-only validation of an explicitly identified T2 cooldown checkpoint.

The original checkpoint has no global RNG snapshot. This helper never invents one:
the source's input generator is keyed by seed/update. The adapter records newly
initialized per-rank RNG states when committing its first native checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


SOURCE_COMMIT = "205c1b7d24b7321debe08feb8fec127d52d94a2b"
SHARED_PARAMETERS = (
    "lr", "min_lr", "warmup", "max_updates", "decay_fraction", "weight_decay", "clip",
    "ema_decay", "target", "seed", "init_seed", "chunk_tokens", "checkpoint_activations",
    "attention", "eval_every", "eval_seeds", "eval_steps", "milestone_weights",
    "final_weights", "pass_f1", "stop_at_pass", "probe_every", "tiny", "init_vertex_sha",
    "init_checkpoint_sha", "init_optimizer", "ema_from_raw",
)
INPUT_PATHS = ("cache", "vae", "init_vertex", "init_checkpoint")


def require(condition, message):
    if not condition:
        raise ValueError("single-GPU migration: " + message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_migration(output, sha256_file):
    """Verify the manifest and immutable external parent/config without modifying either."""
    output = Path(output).resolve()
    migration = json.loads((output / "migration.json").read_text())
    require(migration.get("version") == 1, "unsupported manifest version")
    require(migration.get("kind") == "single_gpu_to_distributed", "wrong migration kind")
    require(migration.get("source_commit") == SOURCE_COMMIT, "unexpected research source commit")
    require(migration.get("target_world_size") == 4, "migration requires four ranks per experiment")
    require(migration.get("global_rng_state_saved") is False, "legacy global RNG must be marked unavailable")
    require(migration.get("input_randomness") == "seed/update keyed; global RNG unavailable",
            "input randomness declaration differs")
    require(type(migration.get("adam_step_offset")) is int and migration["adam_step_offset"] >= 0,
            "AdamW parent step offset is missing or invalid")
    for name in ("parent", "source_config"):
        entry = migration[name]
        path = Path(entry["file"])
        require(path.is_absolute(), name + " path must be absolute")
        path = path.resolve()
        require(not path.is_relative_to(output), name + " must be outside the new output")
        require(path.is_file(), name + " file is missing")
        require(type(entry["bytes"]) is int and entry["bytes"] > 0, name + " byte count is invalid")
        require(path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"],
                name + " size/SHA256 mismatch")
    update = migration["parent"]["update"]
    require(type(update) is int and update > 0, "parent update must be positive")
    cursor = migration["data_cursor"]
    require(cursor["next_update"] == update + 1, "next update disagrees with parent")
    require(cursor["objects_per_update"] == 50, "each update must contain all 50 objects")
    require(len(cursor["uids"]) == 50 and len(set(cursor["uids"])) == 50,
            "cursor must identify 50 unique UIDs")
    require(type(cursor["stateless_batch_seed"]) is int, "data stream seed is missing")
    files = migration.get("source_files", {})
    require(all(name in files for name in ("train.py", "data.py", "model.py", "common.py",
                                          "vendor/vertex_s0.py")), "research source hashes are incomplete")
    for name, sha in files.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py",
                "research source path escapes source directory")
        require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                "invalid research source SHA256")
    return migration


def normalized_path(value):
    return str(Path(value).resolve()) if value is not None else None


def validate_legacy_checkpoint(state, args, cfg, migration, source_config, *, data=None):
    """Reject recipe/state/data drift before any legacy state is used for an update.

    The caller must also load the cache through original data.load_cache (which
    verifies all arrays and frozen normalization) and load the model strictly.
    Passing the loaded data additionally binds its actual UID order and token count.
    """
    require(migration.get("target_world_size") == 4, "migration requires four ranks per experiment")
    require(set(state) == {"args", "config", "ema", "final_stage", "model", "optimizer", "update"},
            "unexpected legacy payload fields; native or RNG-bearing states need their own resume path")
    expected_config = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    require(state["config"] == expected_config == source_config["config"], "model configuration differs")
    update = state["update"]
    require(type(update) is int and update == migration["parent"]["update"], "parent update differs")
    require(type(state["final_stage"]) is bool, "final_stage must be boolean")
    legacy_args = state["args"]
    for key in SHARED_PARAMETERS:
        require(key in legacy_args and key in source_config and hasattr(args, key),
                "missing shared parameter " + key)
        require(legacy_args[key] == source_config[key] == getattr(args, key), "recipe differs at " + key)
    for key in INPUT_PATHS:
        require(key in legacy_args and key in source_config and hasattr(args, key), "missing input path " + key)
        require(normalized_path(legacy_args[key]) == normalized_path(source_config[key])
                == normalized_path(getattr(args, key)), "input identity/path differs at " + key)
    require(legacy_args.get("speed_check", 0) == 0 and source_config.get("speed_check", 0) == 0,
            "speed-only state is not a training continuation")
    require(0 < update <= args.max_updates, "update lies outside the unchanged stage budget")
    transfer = source_config.get("transfer") or {}
    if args.init_checkpoint is not None:
        require(transfer.get("init_checkpoint") == source_config["init_checkpoint"],
                "initial checkpoint transfer identity differs")
        require(transfer.get("optimizer_loaded") is bool(args.init_optimizer),
                "initial optimizer transfer declaration differs")
        require(transfer.get("ema_from_raw") is bool(args.ema_from_raw),
                "initial EMA transfer declaration differs")
        require(type(transfer.get("parent_update")) is int and transfer["parent_update"] > 0,
                "initial checkpoint parent update is missing")
        require(isinstance(args.init_checkpoint_sha, str) and len(args.init_checkpoint_sha) == 64,
                "initial checkpoint SHA256 is missing")
    offset = transfer.get("parent_update", 0) if transfer.get("optimizer_loaded", False) else 0
    require(migration.get("adam_step_offset") == offset, "AdamW parent step offset differs")
    cursor = migration["data_cursor"]
    require(cursor["stateless_batch_seed"] == args.seed, "cursor seed differs")
    require(cursor["next_update"] == update + 1, "cursor update differs")
    require(cursor["uids"] == source_config["uids"], "source UID order differs")
    require(cursor["objects_per_update"] == len(source_config["uids"]) == 50, "source cohort is not 50")
    if data is not None:
        require(data.uids == cursor["uids"] and data.count == 50, "loaded cache UID order differs")
        require(data.seg.tokens == source_config["tokens"], "loaded cache token count differs")
    source = Path(args.source).resolve()
    actual_files = {str(path.relative_to(source)): digest(path) for path in sorted(source.rglob("*.py"))
                    if not path.name.startswith("._")}
    require(actual_files == migration["source_files"], "research source files differ from identified parent")
    require(isinstance(state["model"], dict) and bool(state["model"]), "model state is empty")
    require(set(state["ema"]) == set(state["model"]), "EMA parameter mapping differs")
    for name, parameter in state["model"].items():
        shadow = state["ema"][name]
        require(isinstance(parameter, torch.Tensor) and isinstance(shadow, torch.Tensor),
                "non-tensor model/EMA entry " + name)
        require(parameter.shape == shadow.shape and parameter.dtype == shadow.dtype == torch.float32,
                "model/EMA tensor shape or FP32 master dtype differs at " + name)
    optimizer = state["optimizer"]
    require(len(optimizer["param_groups"]) == 1, "expected original single AdamW parameter group")
    group = optimizer["param_groups"][0]
    require(tuple(group["betas"]) == (.9, .999) and group["eps"] == 1e-8
            and group["weight_decay"] == args.weight_decay, "AdamW hyperparameters differ")
    require(not group.get("amsgrad", False) and not group.get("maximize", False), "AdamW algorithm differs")
    lr = args.lr * min(1., update / max(args.warmup, 1))
    if args.decay_fraction > 0:
        start = int(round(args.max_updates * (1 - args.decay_fraction)))
        if update > start:
            fraction = min(1., (update - start) / max(args.max_updates - start, 1))
            lr = args.min_lr + (args.lr - args.min_lr) * (1 - fraction)
    require(group["lr"] == lr, "saved AdamW learning rate disagrees with update schedule")
    ids = group["params"]
    require(len(ids) == len(set(ids)) == len(state["model"]), "AdamW parameter mapping is incomplete")
    require(set(optimizer["state"]) == set(ids), "AdamW states are incomplete")
    for pid, (name, parameter) in zip(ids, state["model"].items()):
        slot = optimizer["state"][pid]
        step = slot["step"]
        require(isinstance(step, torch.Tensor) and step.numel() == 1 and float(step) == offset + update,
                "AdamW step differs at " + name)
        for key in ("exp_avg", "exp_avg_sq"):
            require(slot[key].shape == parameter.shape and slot[key].dtype == parameter.dtype,
                    "AdamW moment mapping differs at " + name)
    return dict(parent=migration["parent"], source_config=migration["source_config"],
                source_commit=migration["source_commit"], source_files=migration["source_files"],
                source_world_size=1, target_world_size=4, global_rng_state_saved=False,
                global_rng_restored=False, input_randomness=migration["input_randomness"],
                adam_step_offset=offset,
                data_cursor=cursor)
