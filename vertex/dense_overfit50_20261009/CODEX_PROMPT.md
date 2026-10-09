You are running a time-boxed GPU experiment for a NEXUS Vertex Diffusion reproduction.
Run the commands exactly as specified. Do not modify any code or hyperparameters, with the one
exception listed in step 3.

## Context
- Repo: https://github.com/leoguohr/suxen, branch `vertex-dense-overfit50-20261009`.
  Folder: `vertex/dense_overfit50_20261009/`. Read `RUN_SHEET.md` in that folder first. This prompt
  follows it.
- Hardware: two servers, each with 2 × A100 80GB, so 4 GPUs. The GPU window ends at
  **19:27 Asia/Singapore** on 2026-10-09. All training and evaluation must finish by then.
- Starting checkpoint: S0@34000,
  `/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs/S0/checkpoint-034000.pt`
  (28 GB), SHA256 `3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935`.
  If the second server cannot read that path, copy the file there and verify the SHA256.
- Python env: `/guohaoran/envs/nexus-algo/bin/python` (torch, numpy, scipy). If it is missing
  on a server, report that rather than installing something different.
- Goal of this window: compare 4 training settings on the fixed 50-object overfit test. Acceptance
  is 100/100 exact depth-9 trees, measured by `evaluate.py`. Not reaching 100/100 is an expected
  outcome. Just report the numbers.

## Rules
- Write only under a new output directory `OUT` (below). Never modify, move or delete existing
  experiment folders or checkpoints.
- One process per GPU. Select GPUs by UUID (`CUDA_VISIBLE_DEVICES=GPU-...`). Use only GPUs that
  `nvidia-smi` shows as idle.
- If a gate below fails, stop at that point and report the full command, its output and your
  diagnosis. Do not patch the code to make it pass.
- Do not restart, resume or retune a run mid-way. If a run crashes, keep its log and
  `status.json` and report them. The other runs continue.

## Steps (on both servers unless stated)
0. `git fetch origin && git checkout vertex-dense-overfit50-20261009 && git pull`, then
   `cd vertex/dense_overfit50_20261009`. Set
   `PY=/guohaoran/envs/nexus-algo/bin/python`, `S0=<checkpoint path on this server>`,
   `OUT=<disk with >=120 GB free>/nexus_vertex_dense_overfit50_20261009`, then `mkdir -p $OUT`.
1. **Gate: CPU tests.** `CUDA_VISIBLE_DEVICES= $PY tests/test_cpu.py 2>&1 | tee $OUT/test_cpu.log`
   The last line must be `ALL CPU TESTS PASSED`.
2. **Gate: checkpoint identity.** `sha256sum $S0` must match the SHA256 above.
3. **Gate: preflight (server 1, one GPU).**
   `CUDA_VISIBLE_DEVICES=<uuid> $PY -u dense_train.py --init $S0 --init-sha skip --output $OUT/preflight --lr 3e-5 --preflight 2>&1 | tee $OUT/preflight.log`
   - The `[preflight probe, S0 weights]` list must be within ±2 of `[49, 43, 30, 16, 9]`
     at every entry.
   - It must end with `PREFLIGHT_OK`. Record `seconds` and `peak_gib` of the 3 updates.
   - Only exception to "don't change hyperparameters": if it fails with CUDA OOM or peak_gib > 76,
     re-run the preflight with `--chunk-tokens 4096`. If that passes, add `--chunk-tokens 4096` to
     every run in step 4.
4. **Launch 4 runs** as soon as step 3 passes. Compute `T` = whole minutes from now until
   (19:27 − 45 min = 18:42). Use the same T for all four runs. Two runs per server:
   ```bash
   # server 1
   CUDA_VISIBLE_DEVICES=<uuid1> nohup $PY -u dense_train.py --init $S0 --init-sha skip --output $OUT/A --lr 1e-5 --coarse-copies 4 --train-minutes $T > $OUT/A.log 2>&1 &
   CUDA_VISIBLE_DEVICES=<uuid2> nohup $PY -u dense_train.py --init $S0 --init-sha skip --output $OUT/B --lr 3e-5 --coarse-copies 4 --train-minutes $T > $OUT/B.log 2>&1 &
   # server 2
   CUDA_VISIBLE_DEVICES=<uuid3> nohup $PY -u dense_train.py --init $S0 --init-sha skip --output $OUT/C --lr 1e-4 --coarse-copies 4 --train-minutes $T > $OUT/C.log 2>&1 &
   CUDA_VISIBLE_DEVICES=<uuid4> nohup $PY -u dense_train.py --init $S0 --init-sha skip --output $OUT/D --lr 3e-5 --coarse-copies 8 --train-minutes $T > $OUT/D.log 2>&1 &
   ```
   Record the start time and the GPU UUID of each run.
5. **Monitor about every 15 min.** Use `tail -n 3 $OUT/X.log` and the latest
   `[probe uN] raw exact-through-depth [...] | ema [...]` line. A probe line appears at start and
   about every 30 min. Report one short status line per run each time: updates so far, s/update,
   last loss, latest probe.
6. **Evaluate** each run as soon as its log shows `TRAINING_COMPLETE`, on the same GPU:
   `CUDA_VISIBLE_DEVICES=<uuid> $PY -u evaluate.py --checkpoint $OUT/X/final.pt --weights raw --output $OUT/X/eval_raw 2>&1 | tee $OUT/X_eval_raw.log`
   After all four raw evaluations finish, run `--weights ema --output $OUT/X/eval_ema`. Run them in
   order D, B, A, C, only while they can finish before 19:27. An evaluation takes about 15–30 min;
   check `$OUT/X/eval_raw/progress.json`. Stop any evaluation that would overrun the window and
   say so.
7. **Package results (both servers).** Small files only, no `.pt` and no `predictions/`:
   ```bash
   cd $OUT && tar czf dense_results_$(hostname).tgz test_cpu.log preflight.log preflight/preflight.jsonl \
     */config.json */probes.jsonl */train.jsonl */status.json */eval_*/summary.json *.log 2>/dev/null
   ```
   Give the paths of the two `.tgz` files to the user.

## Final report (plain text, for the user to paste back to Claude)
- Gate results for steps 1–3: pass/fail, plus the preflight probe list, s/update and peak GiB.
- One row per run: run, lr, coarse-copies, updates completed, median s/update, final loss,
  probe history (raw and EMA lists at each probe), eval_raw `full_trees_exact`, per-depth exact
  list, first-mismatch histogram, and the same three for eval_ema if it was run.
- Anything unusual: crashes, OOM, NaN, GPUs that were busy, evaluations that were stopped,
  checkpoint copies.
