# Fixed100 VAE continuation

Deploy this `code/` directory beside `source/`, `source_snapshot/`, `source_snapshot_sha256.json`, and `experiment_config.json`. The configured data source must be the original fixed100 source. The source checkpoint stays immutable.

Run only on the specified server/GPU, from the experiment root:

```bash
CUDA_VISIBLE_DEVICES=GPU-3534263c-6584-9f34-9273-0ef6a7852fbe python code/train_vae.py --config experiment_config.json --mode start
```

If the run directory already exists, use `--mode resume`. `--mode evaluate` completes evaluation at the current saved milestone without training. Create `run/STOP` to stop at the next safe update boundary; remove it before resuming. The process never passes 500 new optimizer updates.

The start sequence validates the full source checkpoint and data, initializes only `log_variance.weight=0` and `log_variance.bias=-6`, saves VAE step 0, evaluates 100 μ reconstructions, and requires exact per-UID agreement with the supplied 28/100 AE baseline. It then evaluates five fixed noise groups and records a no-update, five-mesh gradient audit. A failed μ comparison stops before any sampled evaluation or optimizer update.

Recovery checkpoints are committed every original 20-update epoch and at new steps 0, 100, 250, and 500. The active recovery slot is never overwritten before its replacement is committed. `run/recovery-latest.json` identifies the durable state; `run/updates.jsonl` contains only committed updates. `run/status.json` updates after each optimizer step with both global step and new update, plus the last durable checkpoint. The four milestone files are in `run/checkpoints/`. Full μ evaluations run at all four milestones; fixed noise groups run at 0 and 500. Per-UID results, predicted Edge pairs, all enumerated Face shards, epsilon SHA-256, posterior/KL statistics including sigma RMS and percentiles, and aggregate metrics are under `run/evaluations/`.

The 500 updates use the original 100 UIDs, order, negative sampler, Hard4 reconstruction, and full per-mesh forward. The old 412-parameter AdamW group resumes at step 34220; the two logvar parameters start with empty AdamW state. Both groups use constant `1e-4` with no new warmup. Training epsilon comes from a saved separate generator. Each evaluation `(group seed, UID)` has its own deterministic generator; the UID seed does not depend on the checkpoint step.

CPU-only protocol tests, using the existing local PyTorch environment:

```bash
CUDA_VISIBLE_DEVICES='' python code/test_vae_cpu.py
```

The CPU tests cover a small native model, recomputation and epsilon reuse, the source μ path, inherited and new optimizer groups, exact parameter deltas, KL reduction, evaluation RNG isolation, the 250-step cursor, no-update gradient audit, Face enumeration, and deterministic small-model resume. They do not substitute for the server's step-0 μ gate or actual 500-step run. The source AE baseline is 28/100 strict joint successes and is not marked as passed.
