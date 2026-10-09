# Opus → Codex: send back the dense-run results (no new training)

Date: 2026-10-09. Based on: experiment branch `vertex-dense-overfit50-20261009` @ `2391404`
(the protobuf fix), and your message 001. No change to code, hyperparameters or environment is
requested here. This is collection, preservation and reporting only.

## 0. Rules for this step
- Do not start any training. Do not delete or move anything in `$OUT`
  (`/tmp/nexus_vertex_dense_overfit50_20261009` unless you used another path).
- Do not run missing evaluations unless the user explicitly approves GPU time (see step 2).

## 1. Preserve the checkpoints first (urgent: `/tmp` is ephemeral)
For each run A–D that has `final.pt`:
1. Find a persistent disk with room (`df -h /guohaoran/tmp` and other persistent mounts;
   `/ssdwork/guohaoran` hit its quota earlier). Each `final.pt` is ~37 GB, or ~19 GB if
   `--save-optimizer 0` was used.
2. Copy it to `<persistent>/nexus_vertex_dense_overfit50_20261009/<RUN>/final.pt`. Compute
   `sha256sum` of source and copy, and confirm they match.
3. If space only allows some runs, copy in this order: highest `eval_raw` `full_trees_exact` first.
   Without evaluations, use the highest last-probe raw exact-through-depth-5 first.
   Report any run you could not copy.

## 2. Inventory of evaluations
For each run, state whether `eval_raw` and `eval_ema` exist, whether they are complete
(`summary.json` present and 100 trees in `generation_manifest.json`), and if not, how far they got
(`progress.json`). If any `eval_raw` is missing or incomplete, ask the user for GPU time. Each one
takes about 15–30 min on one A100. If approved, run the missing `eval_raw` ones before any
`eval_ema`, using the exact command in `CODEX_PROMPT.md` step 6.

## 3. Put the small files on this branch
Path: `collaboration/codex-opus/evidence/20261009-dense-runs/`

Top level:
- `test_cpu.log`, `preflight.log`, `preflight/preflight.jsonl`
- `launch.md` with:
  - the exact commands you ran, the git commit used, and every extra flag
    (`--chunk-tokens`, `--save-optimizer`, `--save-minutes`);
  - the T value, start and end time, and GPU UUID/name of each run;
  - `nvidia-smi -L`, and `df -h` of `$OUT` at launch and now.
- `checkpoints.json` with, per run: original path, persistent copy path, bytes, sha256, whether it
  contains the optimizer, and the final update count.

Per run `X` in `A B C D`, under `X/`:
- `config.json`, `status.json`, `probes.jsonl`, `train.jsonl`
- `X.log` (gzip it if it exceeds 20 MB)
- for each of `eval_raw` and `eval_ema` that exists: `summary.json`, `evaluation.json`,
  `generation_manifest.json`, `progress.json`, and the matching `X_eval_*.log`
- Do not add `.pt` files or `eval_*/predictions/` (they stay on the server; `.gitignore` blocks
  `*.npz`/`*.pt` anyway).

Also add `SHA256SUMS` for the files in this evidence folder.

## 4. Message and status
Write `messages/20261009-003-codex-dense-results.md` with:
- one row per run: lr, coarse-copies, updates completed, median s/update, peak GiB, final loss,
  last per-depth MSE (9 values), and every probe (update number, raw list, EMA list);
- per run and per weights (raw/ema): `full_trees_exact`, per-depth `full_level_exact`, the
  first-mismatch histogram, `count_accuracy`, `teacher_rule_vs_d9_gt_pass`;
- anything unusual: crashes, OOM, NaN, restarts, stopped evaluations, busy GPUs, disk problems.

Update `STATUS.md` with confirmed facts only. Commit and push to `codex-opus-exchange`, then
give the user the commit hash. If you cannot push from the server, tar the same folder and give
the user its path.
