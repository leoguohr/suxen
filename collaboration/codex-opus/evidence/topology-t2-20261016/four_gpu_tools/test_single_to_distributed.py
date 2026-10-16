"""CPU-only legacy T1 -> four-rank migration and native resume acceptance."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys

import numpy as np
import torch


def compare(actual, expected, path="root", *, exact=True, atol=0., rtol=0.):
    if isinstance(expected, torch.Tensor):
        if exact:
            assert torch.equal(actual, expected), path
        else:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=path)
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(actual, expected), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f"{path}/{key}", exact=exact, atol=atol, rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            compare(a, b, f"{path}/{index}", exact=exact, atol=atol, rtol=rtol)
    else:
        assert actual == expected, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source))
    import common
    import data as data_module
    import model
    import train
    import train_distributed as adapter
    from benchmark_distributed import assignments, synthetic_data, token_indices
    from single_gpu_migration import SOURCE_COMMIT, verify_migration

    torch.set_num_threads(1)
    legacy = args.output / "legacy"
    legacy.mkdir()
    cli = ["--source", str(args.source), "--output", str(legacy), "--tiny", "--device", "cpu",
           "--attention", "sdpa", "--eval-every", "0", "--probe-every", "0", "--final-weights", "",
           "--threads", "1", "--warmup", "0", "--max-updates", "1000",
           "--decay-fraction", "1", "--ema-decay", ".998", "--seed", "20261016"]
    original_argv = sys.argv
    try:
        sys.argv = ["train_distributed.py", *cli]
        recipe = adapter.parse_args()
    finally:
        sys.argv = original_argv
    recipe.init_vertex_sha = recipe.init_vertex_sha or common.S0_SHA256
    recipe.checkpoint_flow_layers = model.TINY.num_layers
    recipe.eval_seed_list = [0, 1]
    data = synthetic_data(data_module, model.TINY, torch.device("cpu"))
    groups = train.object_groups(data, recipe.chunk_tokens)
    partition = assignments(data.lengths, 4)
    assert len(partition) == 4 and all(partition)
    assert sorted(index for group in partition for index in group) == list(range(50))
    indices = torch.cat([token_indices(data, group) for group in partition])
    assert torch.equal(indices.sort().values, torch.arange(data.seg.tokens))

    def fresh():
        system = train.build_system(recipe, torch.device("cpu"))
        system.train()
        optimizer = torch.optim.AdamW(system.parameters(), lr=recipe.lr, betas=(.9, .999),
                                      eps=1e-8, weight_decay=recipe.weight_decay, fused=False)
        return system, optimizer, train.EMA(system, recipe.ema_decay)

    system, optimizer, ema = fresh()
    # A real two-update parent supplies nonzero Adam moments, then the cooldown
    # keeps Adam but resets its own update index and starts EMA from raw once.
    for update in (1, 2):
        train.train_update(system, optimizer, data, groups, recipe, update)
        ema.update()
    initial = legacy / "initial.pt"
    common.save_torch(initial, train.checkpoint_payload(system, ema, optimizer, recipe, 2, True))
    recipe.init_checkpoint = initial
    recipe.init_checkpoint_sha = common.sha256_file(initial)
    cli.extend(["--init-checkpoint", str(initial), "--init-checkpoint-sha", recipe.init_checkpoint_sha])
    ema = train.EMA(system, recipe.ema_decay)
    for update in (1, 2):
        train.train_update(system, optimizer, data, groups, recipe, update)
        ema.update()
    parent = legacy / "latest.pt"
    payload = train.checkpoint_payload(system, ema, optimizer, recipe, 2, False)
    assert "distributed" not in payload and "rng" not in payload
    common.save_torch(parent, payload)
    parent_sha = common.sha256_file(parent)
    source_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(recipe).items()}
    source_config.update(config=model.TINY.to_dict(), uids=data.uids, tokens=data.seg.tokens,
                         start_update=0, resumed=False, torch=torch.__version__, gpu="CPU synthetic test",
                         transfer=dict(init_checkpoint=str(initial), parent_update=2,
                                       optimizer_loaded=True, ema_from_raw=True))
    common.write_json(legacy / "config.json", source_config)

    # Production boundary is tested as a cursor/input case, never as fictitious training updates.
    for update in (3, 501):
        batch = train.make_batch(data, recipe.seed, update, recipe.target)
        torch.manual_seed(917)
        torch.randn(97)
        np.random.seed(918)
        np.random.randn(97)
        random.seed(919)
        [random.random() for _ in range(97)]
        state = adapter.rng_state(torch.device("cpu"))
        compare(train.make_batch(data, recipe.seed, update, recipe.target), batch, f"batch/{update}")
        compare(adapter.rng_state(torch.device("cpu")), state, f"global_rng_unchanged/{update}")
    assert train.lr_at(recipe, 500) == 5e-5
    assert train.lr_at(recipe, 1000) == 0.
    assert not torch.equal(train.make_batch(data, recipe.seed, 500)[1],
                           train.make_batch(data, recipe.seed, 501)[1])

    # Reference resumes the real two-update legacy payload, including nonzero Adam moments.
    reference, ref_opt, ref_ema = fresh()
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    reference.load_state_dict(saved["model"], strict=True)
    ref_opt.load_state_dict(saved["optimizer"])
    ref_ema.load(reference, saved["ema"])
    before_update_rng = adapter.rng_state(torch.device("cpu"))
    reference_row = train.train_update(reference, ref_opt, data, groups, recipe, 3)
    ref_ema.update()
    compare(adapter.rng_state(torch.device("cpu")), before_update_rng, "update_global_rng_unchanged")
    reference_state = dict(model=copy.deepcopy(reference.state_dict()),
                           optimizer=copy.deepcopy(ref_opt.state_dict()), ema=ref_ema.state(reference))

    def load(folder):
        pointer = json.loads((folder / "latest.json").read_text())
        checkpoint = folder / pointer["file"]
        assert checkpoint.stat().st_size == pointer["bytes"]
        assert common.sha256_file(checkpoint) == pointer["sha256"]
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

    def launch(folder, steps, suffix):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command = [sys.executable, "-m", "torch.distributed.run", "--master-addr", "127.0.0.1",
                   "--master-port", str(port), "--nproc_per_node", "4",
                   str(Path(__file__).with_name("train_distributed.py")), *cli]
        command.extend(["--output", str(folder), "--stop-after-updates", str(steps)])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=240)
        log = result.stdout + result.stderr
        (args.output / f"{folder.name}-{suffix}.log").write_text(log)
        if result.returncode:
            print(log, flush=True)
        assert result.returncode == 0, (folder, result.returncode)
        return log

    resumed, straight = args.output / "resumed", args.output / "straight"
    config_path = legacy / "config.json"
    migration = dict(version=1, kind="single_gpu_to_distributed",
                     parent=dict(file=str(parent), bytes=parent.stat().st_size, sha256=parent_sha, update=2),
                     source_config=dict(file=str(config_path), bytes=config_path.stat().st_size,
                                        sha256=common.sha256_file(config_path)),
                     target_world_size=4, source_commit=SOURCE_COMMIT, adam_step_offset=2,
                     source_files={str(path.relative_to(args.source)): common.sha256_file(path)
                                   for path in sorted(args.source.rglob("*.py"))
                                   if not path.name.startswith("._")},
                     global_rng_state_saved=False,
                     input_randomness="seed/update keyed; global RNG unavailable",
                     data_cursor=dict(next_update=3, objects_per_update=50, uids=data.uids,
                                      stateless_batch_seed=recipe.seed))
    for folder in (resumed, straight):
        common.write_json(folder / "migration.json", migration)
        verify_migration(folder, common.sha256_file)
    log = launch(resumed, 1, "u3")
    assert log.index("CHECKPOINT_COMMITTED update=2") < log.index("u3 loss=")
    imported_config = json.loads((resumed / "config-u000002.json").read_text())
    assert imported_config["world_size"] == 4 and imported_config["partition"] == partition
    assert imported_config["uids"] == data.uids and imported_config["tokens"] == data.seg.tokens
    migrated = load(resumed)
    assert migrated["update"] == 3 and migrated["distributed"]["data_cursor"]["next_update"] == 4
    assert all(float(slot["step"]) == 5 for slot in migrated["optimizer"]["state"].values())
    assert migrated["distributed"]["world_size"] == 4
    assert len(migrated["distributed"]["rng_by_rank"]) == 4
    provenance = migrated["distributed"]["parent_migration"]
    assert provenance["target_world_size"] == 4 and provenance["global_rng_state_saved"] is False
    for key in ("model", "ema"):
        compare(migrated[key], reference_state[key], key, exact=False, atol=2e-6, rtol=5e-5)
    compare(migrated["optimizer"], reference_state["optimizer"], "optimizer",
            exact=False, atol=2e-7, rtol=1e-4)
    rows = [json.loads(line) for line in (resumed / "train.jsonl").read_text().splitlines()]
    assert [row["update"] for row in rows] == [3]
    for key in ("loss", "grad_norm"):
        assert abs(rows[0][key] - reference_row[key]) <= 2e-6 + 5e-5 * abs(reference_row[key]), key
    assert rows[0]["lr"] == reference_row["lr"]
    launch(resumed, 1, "u4")
    straight_log = launch(straight, 2, "u3-u4")
    assert straight_log.index("CHECKPOINT_COMMITTED update=2") < straight_log.index("u3 loss=")
    a, b = load(resumed), load(straight)
    assert a["update"] == b["update"] == 4
    assert all(float(slot["step"]) == 6 for slot in a["optimizer"]["state"].values())
    for key in ("model", "ema", "optimizer"):
        compare(a[key], b[key], key)
    for key in ("rng_by_rank", "data_cursor"):
        compare(a["distributed"][key], b["distributed"][key], key)
    assert common.sha256_file(parent) == parent_sha
    report = dict(passed=True, synthetic_legacy_updates=2, migrated_update=3, resumed_update=4,
                  parent_sha256=parent_sha, parent_unchanged=True, target_world_size=4,
                  partition_covers_all_50_once=True, partition_covers_all_tokens_once=True,
                  single_to_distributed="within existing FP32 CPU tolerances",
                  native_resume="exact model/EMA/Adam/rank RNG/data cursor",
                  global_rng_independent_batch=True, update_preserves_global_rng=True,
                  production_cursor_test="500 -> 501 inputs and LR; update 1000 LR zero",
                  inherited_adam_offset=2, cooldown_updates=4, final_adam_steps=6,
                  historical_global_rng_recovered=False, gpu_work=False)
    common.write_json(args.output / "result.json", report)
    print("SINGLE_TO_DISTRIBUTED_MIGRATION_GATE_OK " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
