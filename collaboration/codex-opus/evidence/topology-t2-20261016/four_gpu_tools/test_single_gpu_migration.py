"""CPU rejection tests for legacy checkpoint identity and scientific continuity."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from single_gpu_migration import (SOURCE_COMMIT, SHARED_PARAMETERS, digest, verify_migration,
                                  validate_legacy_checkpoint)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.output = root / "distributed"
        self.output.mkdir()
        source = root / "source"
        source.mkdir()
        for name in ("train.py", "data.py", "model.py", "common.py", "vendor/vertex_s0.py"):
            path = source / name
            path.parent.mkdir(exist_ok=True)
            path.write_text("# synthetic fixture, not production training evidence\n")
        values = dict(lr=1e-4, min_lr=0., warmup=500, max_updates=10000, decay_fraction=.2,
                      weight_decay=.01, clip=1., ema_decay=.999, target="posterior", seed=20261015,
                      init_seed=20261015, chunk_tokens=0, checkpoint_activations=1, attention="sdpa",
                      eval_every=0, eval_seeds="0,1", eval_steps=20, milestone_weights="raw,ema",
                      final_weights="", pass_f1=.99, stop_at_pass=1, probe_every=0, tiny=True,
                      init_vertex_sha="fixture", init_vertex=None, cache=None, vae=None,
                      init_checkpoint=None, init_checkpoint_sha=None, init_optimizer=1, ema_from_raw=1)
        self.args = SimpleNamespace(**values, source=source, checkpoint_flow_layers=1, device="cpu")
        self.cfg = {"latent_dim": 2}
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=values["lr"], betas=(.9, .999),
                                      eps=1e-8, weight_decay=values["weight_decay"])
        for update in (1, 2):
            optimizer.zero_grad()
            model(torch.ones(2)).square().mean().backward()
            optimizer.param_groups[0]["lr"] = values["lr"] * update / values["warmup"]
            optimizer.step()
        self.state = dict(model=model.state_dict(), ema=copy.deepcopy(model.state_dict()),
                          optimizer=optimizer.state_dict(), update=2, final_stage=False,
                          config=self.cfg, args=values)
        parent = root / "latest.pt"
        torch.save(self.state, parent)
        self.source_config = dict(values, config=self.cfg, tokens=100, uids=[f"synthetic-{i}" for i in range(50)])
        config_file = root / "config.json"
        config_file.write_text(json.dumps(self.source_config))
        self.manifest = dict(version=1, kind="single_gpu_to_distributed", target_world_size=4,
                             adam_step_offset=0,
                             source_commit=SOURCE_COMMIT, global_rng_state_saved=False,
                             input_randomness="seed/update keyed; global RNG unavailable",
                             parent=dict(file=str(parent), bytes=parent.stat().st_size,
                                         sha256=digest(parent), update=2),
                             source_config=dict(file=str(config_file), bytes=config_file.stat().st_size,
                                                sha256=digest(config_file)),
                             source_files={str(p.relative_to(source)): digest(p) for p in source.rglob("*.py")},
                             data_cursor=dict(next_update=3, objects_per_update=50,
                                              uids=self.source_config["uids"], stateless_batch_seed=values["seed"]))
        self.write_manifest()
        self.data = SimpleNamespace(uids=self.source_config["uids"], count=50,
                                    seg=SimpleNamespace(tokens=100))

    def write_manifest(self):
        (self.output / "migration.json").write_text(json.dumps(self.manifest))

    def validate(self, state=None, args=None, config=None, data=None):
        return validate_legacy_checkpoint(self.state if state is None else state,
                                          self.args if args is None else args, self.cfg, self.manifest,
                                          self.source_config if config is None else config,
                                          data=self.data if data is None else data)

    def test_valid_parent_stays_unchanged_and_rng_is_not_invented(self):
        before = Path(self.manifest["parent"]["file"]).read_bytes()
        self.assertEqual(verify_migration(self.output, digest), self.manifest)
        provenance = self.validate()
        self.assertFalse(provenance["global_rng_restored"])
        self.assertEqual(provenance["target_world_size"], 4)
        self.assertEqual(provenance["data_cursor"]["next_update"], 3)
        self.assertEqual(Path(self.manifest["parent"]["file"]).read_bytes(), before)

    def test_two_rank_manifest_rejected(self):
        self.manifest["target_world_size"] = 2
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "four ranks"):
            verify_migration(self.output, digest)
        with self.assertRaisesRegex(ValueError, "four ranks"):
            self.validate()

    def test_cooldown_inherited_adam_step_and_ema_preserved(self):
        self.args.init_checkpoint = Path(self.temporary.name) / "initial-u4000.pt"
        self.args.init_checkpoint_sha = "a" * 64
        for values in (self.state["args"], self.source_config):
            values.update(init_checkpoint=str(self.args.init_checkpoint),
                          init_checkpoint_sha=self.args.init_checkpoint_sha)
        self.source_config["transfer"] = dict(init_checkpoint=str(self.args.init_checkpoint),
                                              optimizer_loaded=True, ema_from_raw=True, parent_update=4000)
        self.manifest["adam_step_offset"] = 4000
        for slot in self.state["optimizer"]["state"].values():
            slot["step"].add_(4000)
        ema_before = copy.deepcopy(self.state["ema"])
        self.assertEqual(self.validate()["adam_step_offset"], 4000)
        for key in ema_before:
            self.assertTrue(torch.equal(ema_before[key], self.state["ema"][key]))
        self.manifest["adam_step_offset"] = 0
        with self.assertRaisesRegex(ValueError, "offset differs"):
            self.validate()
        self.manifest["adam_step_offset"] = 4000
        next(iter(self.state["optimizer"]["state"].values()))["step"].sub_(4000)
        with self.assertRaisesRegex(ValueError, "AdamW step differs"):
            self.validate()

    def test_every_shared_parameter_rejects_drift(self):
        for key in SHARED_PARAMETERS:
            with self.subTest(key=key):
                args = copy.deepcopy(self.args)
                setattr(args, key, "altered")
                with self.assertRaisesRegex(ValueError, key):
                    self.validate(args=args)

    def test_optimizer_ema_and_update_corruption_rejected(self):
        for mutation in ("step", "lr", "betas", "mapping", "ema", "update", "final_stage", "global_rng"):
            with self.subTest(mutation=mutation):
                state = copy.deepcopy(self.state)
                if mutation == "step":
                    next(iter(state["optimizer"]["state"].values()))["step"].add_(1)
                elif mutation == "lr":
                    state["optimizer"]["param_groups"][0]["lr"] = .2
                elif mutation == "betas":
                    state["optimizer"]["param_groups"][0]["betas"] = (.8, .999)
                elif mutation == "mapping":
                    state["optimizer"]["param_groups"][0]["params"] = [0, 0]
                elif mutation == "ema":
                    state["ema"].pop("bias")
                elif mutation == "update":
                    state["update"] += 1
                elif mutation == "final_stage":
                    state["final_stage"] = "false"
                else:
                    state["rng"] = torch.get_rng_state()
                with self.assertRaises(ValueError):
                    self.validate(state=state)

    def test_source_changes_rejected(self):
        (self.args.source / "model.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "research source files"):
            self.validate()

    def test_actual_data_order_and_count_rejected(self):
        changed = copy.deepcopy(self.data)
        changed.uids.reverse()
        with self.assertRaisesRegex(ValueError, "UID order"):
            self.validate(data=changed)
        changed = copy.deepcopy(self.data)
        changed.seg.tokens += 1
        with self.assertRaisesRegex(ValueError, "token count"):
            self.validate(data=changed)

    def test_parent_sha_or_config_sha_corruption_rejected(self):
        original = copy.deepcopy(self.manifest)
        for key in ("parent", "source_config"):
            self.manifest = copy.deepcopy(original)
            self.manifest[key]["sha256"] = "0" * 64
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, "size/SHA256 mismatch"):
                verify_migration(self.output, digest)

    def test_parent_inside_output_rejected(self):
        inside = self.output / "parent.pt"
        inside.write_bytes(Path(self.manifest["parent"]["file"]).read_bytes())
        self.manifest["parent"]["file"] = str(inside)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "outside the new output"):
            verify_migration(self.output, digest)

    def test_cursor_and_rng_claim_rejected(self):
        self.manifest["data_cursor"]["next_update"] = 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "next update"):
            verify_migration(self.output, digest)
        self.manifest["data_cursor"]["next_update"] = 3
        self.manifest["global_rng_state_saved"] = True
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "global RNG"):
            verify_migration(self.output, digest)


if __name__ == "__main__":
    unittest.main()
