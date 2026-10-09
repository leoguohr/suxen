# Run sheet: dense 50-object Vertex overfit (2026-10-09, 4 × A100, ~4 h window)

Code: `vertex/dense_overfit50_20261009/` on branch `vertex-dense-overfit50-20261009`.
Data (the same 50 NEXUS2K objects, D9 labels, 8192-point XYZ+normal conditions) is in `data/`
with SHA256 checks. The S0 model code is vendored byte-for-byte in `vendor/`.

## What this tests

S0@34000 (best so far, 7/100) fails mostly at the cheap coarse depths. First mismatch over its
100 trees: D1 2, D2 10, D3 30, D4 26, D5 16, D6–D8 9, exact 7. Depths 1–5 of all 50 objects
together are only ~15k parent tokens, versus 154k for depths 6–9.

Same model, same velocity-MSE objective, same data. New: every update takes 8 objects × all
9 depths (VecSet runs once per object), depths 1–5 get extra stratified-t copies, the optimizer
is a fresh AdamW (wd 0) with a 50-update warmup, and an EMA of the weights is tracked.
Each (object, depth) has equal loss weight, as before.

## Acceptance criterion

**PASS = 100/100 exact trees.** 50 objects × 2 fixed seeds (97029000, 97029001). Generation runs
from noise at the root to depth 9, uses the model's own parents at every level, takes 20 Euler
steps per depth, and thresholds at 0.5. Exact means the vertex count is exact and every vertex is
in the correct D9 cell. This is the same protocol as S0's evaluation.
`evaluate.py` also reports teacher-style numbers: count accuracy, matched RMSE, and
max error / min spacing < 0.1.

## The four runs (one GPU each, no DDP)

| Run | Server/GPU | `--lr` | `--coarse-copies` | Question |
|---|---|---|---|---|
| A | server 1, GPU a | 1e-5 | 4 | dense batching alone (S0's learning rate) |
| B | server 1, GPU b | 3e-5 | 4 | dense + moderate LR |
| C | server 2, GPU a | 1e-4 | 4 | dense + the paper's LR |
| D | server 2, GPU b | 3e-5 | 8 | stronger depth-1–5 emphasis |

## Steps

Set once per server, then adjust paths:
```bash
cd <repo>/vertex/dense_overfit50_20261009
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8   # protobuf/onnx clash; S0 used this too
PY=/guohaoran/envs/nexus-algo/bin/python
S0=/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs/S0/checkpoint-034000.pt   # or the copied path
OUT=<a disk with >=120 GB free>/nexus_vertex_dense_overfit50_20261009
```
Each run writes `final.pt` (~37 GB: model + EMA + optimizer). During training it also keeps
`latest_weights.pt` (~19 GB, deleted at the end). If disk is short, `--save-optimizer 0` makes
`final.pt` ~19 GB.

1. **CPU tests (both servers, 1–3 min).** `CUDA_VISIBLE_DEVICES= $PY tests/test_cpu.py`
   It must end with `ALL CPU TESTS PASSED`. If any line says FAIL, stop and send the full output.
2. **Checkpoint identity (both servers).** `sha256sum $S0` must be
   `3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935`.
3. **Preflight (server 1, one GPU, ~5 min).**
   `CUDA_VISIBLE_DEVICES=<uuid> $PY dense_train.py --init $S0 --init-sha skip --output $OUT/preflight --lr 3e-5 --preflight`
   - The S0 probe line must be close to `[49, 43, 30, 16, 9]` (within ±2 per entry). Otherwise stop
     and report: it means the weights or sampling don't match S0.
   - Three updates on the 8 largest objects print `seconds` and `peak_gib`. If there is an OOM or
     peak > 76 GiB, add `--chunk-tokens 4096` to every run below.
   - Record seconds/update. These are the largest objects, so typical updates are faster.
4. **Launch the 4 runs.** Set `T` = minutes until (window end − 45 min). Example: launch 16:40,
   window ends 19:27, so T=120. Use the same T for all four. For each run:
   ```bash
   CUDA_VISIBLE_DEVICES=<uuid> nohup $PY -u dense_train.py --init $S0 --init-sha skip \
     --output $OUT/A --lr 1e-5 --coarse-copies 4 --train-minutes $T > $OUT/A.log 2>&1 &
   ```
   B: `--output $OUT/B --lr 3e-5 --coarse-copies 4`. C: `--output $OUT/C --lr 1e-4 --coarse-copies 4`.
   D: `--output $OUT/D --lr 3e-5 --coarse-copies 8`. Add `--chunk-tokens 4096` if step 3 required it.
5. **While training.** Each log prints a line every 10 updates (loss, per-depth MSE, seconds,
   peak memory) and a `[probe uN]` line at start and every 30 min. The probe shows exact-through-depth
   1..5 for 50 objects, seed 97029000, raw and EMA weights. Don't restart or change anything
   mid-run. If a run crashes, keep its log and `status.json` and report it.
6. **Evaluate (after `TRAINING_COMPLETE`, same GPU, ~15–30 min per evaluation).**
   ```bash
   CUDA_VISIBLE_DEVICES=<uuid> $PY -u evaluate.py --checkpoint $OUT/A/final.pt --weights raw --output $OUT/A/eval_raw
   ```
   Do raw for all four runs first. Then, if the window allows, `--weights ema --output $OUT/A/eval_ema`.
   Each prints `EVALUATION_COMPLETE full_trees_exact=N/100 ...`.
7. **Report back.** For each run, bundle `config.json probes.jsonl train.jsonl status.json`
   `eval_*/summary.json` plus the log (all small; skip `.pt` files and `eval_*/predictions`):
   `tar czf dense_A.tgz -C $OUT A/config.json A/probes.jsonl A/train.jsonl A/status.json A/eval_raw/summary.json A.log`
   Also include `$OUT/preflight/preflight.jsonl`.

## Not changed / not claimed

Model architecture and size, objective, data, and the evaluation protocol are unchanged. No ground
truth is used at inference, and there is no lookup prior. Training is warm-started from S0, so this
is not a from-scratch result. One seed per configuration, so small differences between runs are
not conclusive.
