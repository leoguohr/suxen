# Run sheet: topology flow overfit50 (2 GPUs)

Every command runs with `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8`.
Use two GPUs that no vertex job is using (check `nvidia-smi`; never touch vertex processes).
Nothing here modifies an earlier run folder or checkpoint: the cache, the VAE and the S0 checkpoint are
only read, and every output goes under `$OUT`.

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8
PY=/guohaoran/envs/nexus-algo/bin/python
CODE=<fresh checkout of branch topology-overfit50-20261015>/topology/flow_overfit50_20261015
CACHE=/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006/cache
#   manifest.json SHA256 must be f14f796c421cab6b9d5206047f927ea7c3490ff3c2435b2ebd38904a749c27e3
#   (the original is /ssdwork/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_20261002/cache)
VAE=/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/run/checkpoints/vae-1000.pt
#   SHA256 e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15
S0=/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs/S0/checkpoint-034000.pt
#   SHA256 3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935 (any byte-identical copy is fine)
OUT=/guohaoran/nexus_topology_flow_overfit50_20261015
mkdir -p $OUT && cd $CODE
```

## Step 1: CPU gate (no GPU)

```bash
CUDA_VISIBLE_DEVICES= $PY tests/test_cpu.py 2>&1 | tee $OUT/cpu_gate.log        # must end CPU_GATE_OK
CUDA_VISIBLE_DEVICES= $PY data.py --cache $CACHE 2>&1 | tee $OUT/cache_check.log # must end CACHE_OK
```
Stop and report if either fails. `cache_check.log` should show 50 objects, 52,820 tokens and a
`d9_center_max_offset` near 0, which confirms the RoPE unit.

## Step 2: GPU 2, evaluator self-check and decoder tolerance (about 1 hour)

```bash
CUDA_VISIBLE_DEVICES=<gpu2> $PY -u evaluate.py --mode mu --cache $CACHE --vae $VAE --output $OUT/eval_mu
CUDA_VISIBLE_DEVICES=<gpu2> $PY -u evaluate.py --mode noise --cache $CACHE --vae $VAE --output $OUT/tolerance \
    --alphas 0,0.02,0.05,0.1,0.2,0.3,0.5 --seeds 0
```
The mu run must print `fixture exact match: True`: edges 154,973 / 125 / 5 and faces 102,878 / 870 / 12.
If it does not, report the first mismatches in `eval_mu/mu/summary.json` and continue anyway.

## Step 3: GPU 1, speed check (about 10 minutes)

```bash
CUDA_VISIBLE_DEVICES=<gpu1> $PY -u train.py --cache $CACHE --vae $VAE --output $OUT/speed_check --speed-check 20
```
This prints seconds per update, peak memory and a flash-vs-SDPA comparison, and writes
`speed_check.json`. On out-of-memory, rerun with `--chunk-tokens 27000`; if that still fails, use
`--chunk-tokens 14000`. Pass the working value to Steps 4 and 5. If one update takes more than
30 s, stop and report.

## Step 4: GPU 1, Run A (paper-faithful, fresh initialization)

```bash
CUDA_VISIBLE_DEVICES=<gpu1> nohup $PY -u train.py --cache $CACHE --vae $VAE --output $OUT/A_fresh \
    [--chunk-tokens N] > $OUT/A_fresh.log 2>&1 &
```
Defaults: 10,000 updates, warmup 500, decay from update 8,000, EMA 0.999, chain evaluation (raw + EMA,
seeds 0 and 1, Euler 20) every 2,000 updates and at the end, probe every 250 updates, saving every
60 min. After any stop, rerun the identical command to resume.

## Step 5: GPU 2, Run B (same recipe, warm start from vertex S0)

```bash
CUDA_VISIBLE_DEVICES=<gpu2> nohup $PY -u train.py --cache $CACHE --vae $VAE --output $OUT/B_s0warm \
    --init-vertex $S0 [--chunk-tokens N] > $OUT/B_s0warm.log 2>&1 &
```
The command is identical to Run A apart from the S0 warm start. The data stream and noise seeds match Run A update for update.

## Step 6: after a run completes, sampler sweep on its final checkpoint

```bash
CUDA_VISIBLE_DEVICES=<gpu> $PY -u evaluate.py --mode flow --cache $CACHE --vae $VAE --checkpoint $OUT/A_fresh/final.pt \
    --weights ema --seeds 0,1 --steps 10,20,50 --samplers euler,dpm2m --output $OUT/A_fresh/sweep_ema
CUDA_VISIBLE_DEVICES=<gpu> $PY -u evaluate.py --mode flow --cache $CACHE --vae $VAE --checkpoint $OUT/A_fresh/final.pt \
    --weights raw --seeds 0,1 --steps 20 --samplers euler --output $OUT/A_fresh/sweep_raw
```
Also run the 20-step Euler EMA case with `--dtype fp32` once.

## Outputs to report (small files only, no .pt)

`cpu_gate.log`, `cache_check.log`, `eval_mu/results.json` and `eval_mu/mu/summary.json`,
`tolerance/results.json`, `speed_check/speed_check.json`, and for each run `config.json`, `train.jsonl`
(gzip), `probes.jsonl`, `evals.jsonl` and `evals/*/seed*_summary.json`.
