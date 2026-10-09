You are setting up round 2 of the NEXUS Vertex Diffusion 50-object overfit: two long, unattended
GPU jobs. Do not modify model/training code or hyperparameters. You own the server-specific
parts: storage, job scripts, copying checkpoints, submission and monitoring.

## Read first
- Repo `leoguohr/suxen`, branch `vertex-dense-overfit50-20261009`, folder
  `vertex/dense_overfit50_20261009/`. Read `ROUND2.md`: the plan, unattended behaviour and the
  exact commands.
- Round-1 records are on branch `codex-opus-exchange`. Round-1 artifacts are in
  `/guohaoran/tmp/nexus_vertex_dense_overfit50_20261009/`.

## Constraints
- Two GPUs: one GPU on each of two servers, via the long-running job system. There is no maximum
  runtime, and jobs can be stopped and restarted automatically.
- Store everything under `/guohaoran` or `/ssd/guohaoran`. Never modify or delete earlier
  experiment folders or checkpoints.
- Always set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `OMP_NUM_THREADS=8`.
- Keep coarse probes ON. Do not set `NEXUS_SKIP_COARSE_PROBES` and do not pass
  `--probe-minutes 0` in the real jobs.

## Steps
1. **CPU margin check (no GPU, do this first).** On any machine that sees `/guohaoran`:
   ```bash
   R1=/guohaoran/tmp/nexus_vertex_dense_overfit50_20261009
   $PY analyze_margins.py --eval-dir $R1/A/eval_raw --eval-dir $R1/D/eval_raw --eval-dir $R1/D/eval_ema \
     --eval-dir $R1/B/eval_raw --output <OUT>/round1_margins.json
   ```
   Keep the printed lines and the JSON.
2. **CPU tests.** `CUDA_VISIBLE_DEVICES= $PY tests/test_cpu.py`. The script forces the math
   attention kernel itself, so you don't need your earlier wrapper. It must end with
   `ALL CPU TESTS PASSED`. If not, stop and report.
3. **Init checkpoint.** Round-1 `A/final.pt`, SHA256
   `1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9`. Last time, mmap loading from
   the shared filesystem stalled. If that is a risk, copy the file to each server's fast local or
   SSD storage under `/ssd/guohaoran` and verify the SHA256.
4. **GPU smoke test (one GPU, ~15 min, before the long jobs).** This uses throwaway output and
   checks resume, milestone eval and the final stage:
   ```bash
   S=<OUT>/smoke && rm -rf $S   # throwaway smoke dir only
   $PY -u dense_train.py --init $INIT --init-sha skip --init-state full --lr 1e-5 --warmup 2 \
     --loss-weighting token --max-updates 4 --decay-fraction 0.5 --eval-every 2 --eval-steps 2 \
     --final-evals raw --probe-minutes 0 --save-minutes 1000 --seed 1 --output $S
   ```
   Interrupt it with SIGTERM (`kill -TERM <pid>`) after the log shows `u3`. It must print
   `INTERRUPTED at update N` and exit with code 143. N is 3, or 4 if the signal arrived during
   update 4. Then rerun the identical command. It must print `RESUMED ... at update N` and finish
   with `JOB_COMPLETE updates=4`. Run it once more: it must
   print `ALREADY_COMPLETE`. Check that `evals.jsonl` has rows for (2, raw) and (4, raw) and that
   `final.pt` exists. `--eval-steps 2` is only for this smoke test; the real jobs use the default
   of 20. Delete the smoke directory afterwards if space matters.
5. **Submit the two jobs** with exactly the commands in `ROUND2.md`: J1 (`--loss-weighting item`)
   on server 1 and J2 (`--loss-weighting token`) on server 2. The job script must rerun the
   identical command after a restart. Exit code 143 means interrupted: rerun it. Exit code 0 means
   complete or already complete. Anything else is a failure: read `status.json` and the log, and
   report.
6. **Report (push to `codex-opus-exchange`, as `messages/2026101x-NNN-codex-round2-*.md` plus
   evidence).**
   - **Right after submission:** the margin-check output, the CPU test log, the smoke-test log and
     result, the job IDs, servers/GPUs, storage paths, the exact job scripts, and a SHA256 for each
     copy of the init checkpoint.
   - **After each milestone evaluation (~every 4 h), and at the end:** per job, `evals.jsonl`,
     `probes.jsonl`, `status.json`, the new `evals/*/summary.json` and `evaluation.json`, and
     `train.jsonl`. Compress if large; keep the last row per update. No `.pt` and no `predictions/`.
   - **Anything unusual:** restarts, OOM, NaN, slow storage, or evaluations that were repeated after
     a restart.
