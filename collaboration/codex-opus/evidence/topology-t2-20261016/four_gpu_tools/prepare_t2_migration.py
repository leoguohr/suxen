"""Bind a stopped T2 cooldown to an isolated four-rank continuation; CPU only."""
import argparse
import json
from pathlib import Path
import sys

import torch

from single_gpu_migration import (INPUT_PATHS, SHARED_PARAMETERS, SOURCE_COMMIT,
                                  validate_legacy_checkpoint, verify_migration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("parent", "source-config", "source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    options = parser.parse_args()
    source = options.source.resolve()
    output = options.output.resolve()
    parent = options.parent.resolve()
    config_path = options.source_config.resolve()
    sys.path.insert(0, str(source))
    import common
    import data as data_module
    import model
    import train_distributed as adapter

    assert not torch.cuda.is_initialized()
    status = json.loads((parent.parent / "status.json").read_text())
    assert status["state"] == "interrupted", "stop and atomically save the single-GPU run first"
    source_config = json.loads(config_path.read_text())
    transfer = source_config["transfer"]
    assert transfer["optimizer_loaded"] is True and transfer["parent_update"] == 4000
    assert source_config["init_optimizer"] == 1 and source_config["ema_from_raw"] == 1
    assert source_config["init_checkpoint_sha"] == "abc330ba9328a79ac29c3eb0ad1989b18d2be6ebb0a941d8e3f773bb75ae9d59"
    assert source_config["max_updates"] == 1000 and source_config["warmup"] == 0
    assert source_config["decay_fraction"] == 1.0 and source_config["seed"] == 20261016

    command = [str(Path(__file__).with_name("train_distributed.py").resolve()),
               "--source", str(source), "--output", str(output)]
    for key in dict.fromkeys((*SHARED_PARAMETERS, *INPUT_PATHS)):
        value = source_config[key]
        if key == "tiny":
            assert value is False
        elif value is not None:
            command.extend(["--" + key.replace("_", "-"), str(value)])
    command.extend(["--checkpoint-flow-layers", "36"])
    previous = sys.argv
    try:
        sys.argv = command
        args = adapter.parse_args()
    finally:
        sys.argv = previous
    args.init_vertex_sha = args.init_vertex_sha or common.S0_SHA256
    assert args.init_vertex_sha == source_config["init_vertex_sha"]
    data = data_module.load_cache(args.cache, "cpu")
    state = torch.load(parent, map_location="cpu", mmap=True, weights_only=False)
    assert state["update"] == status["update"] and state["final_stage"] is False
    assert 0 < state["update"] < args.max_updates

    def bound(path):
        before = path.stat()
        value = dict(file=str(path), bytes=before.st_size, sha256=common.sha256_file(path))
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
        return value

    migration = dict(version=1, kind="single_gpu_to_distributed", target_world_size=4,
                     source_commit=SOURCE_COMMIT,
                     parent=dict(bound(parent), update=state["update"]), source_config=bound(config_path),
                     source_files={str(p.relative_to(source)): common.sha256_file(p)
                                   for p in sorted(source.rglob("*.py")) if not p.name.startswith("._")},
                     adam_step_offset=transfer["parent_update"], global_rng_state_saved=False,
                     input_randomness="seed/update keyed; global RNG unavailable",
                     data_cursor=dict(next_update=state["update"] + 1, objects_per_update=50,
                                      uids=data.uids, stateless_batch_seed=args.seed))
    validation = validate_legacy_checkpoint(state, args, model.PAPER, migration, source_config, data=data)
    with torch.device("meta"):
        template = model.TopologyFlowSystem(model.PAPER)
    template.load_state_dict(state["model"], strict=True, assign=True)
    assert model.parameter_count(template) == 2333965312
    assert not output.exists() or not any(output.iterdir()), "new continuation output must be empty"
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(output / "migration.json", migration)
    assert verify_migration(output, common.sha256_file) == migration
    common.write_json(output.parent / "four_gpu_launch_args.json", command)
    common.write_json(output.parent / "four_gpu_parent_validation.json",
                      dict(passed=True, cpu_only=True, model_strict_load=True,
                           optimizer_state_count=len(state["optimizer"]["state"]),
                           update=state["update"], adam_step=4000 + state["update"],
                           validation=validation, launch_args=command))
    assert not torch.cuda.is_initialized()
    print("T2_FOUR_GPU_PARENT_READY " + json.dumps(dict(update=state["update"],
          next_update=state["update"] + 1, adam_step=4000 + state["update"], output=str(output))), flush=True)


if __name__ == "__main__":
    main()
