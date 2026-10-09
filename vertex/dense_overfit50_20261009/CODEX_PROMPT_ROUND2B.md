Two more GPUs are available. Add jobs J3 and J4 next to J1/J2. Do not modify code or hyperparameters.

1. `git pull` branch `vertex-dense-overfit50-20261009` and use the newest commit. If J1/J2 run from the
   same checkout, that is safe: the new flags default to off and leave their behaviour bit-identical.
2. CPU tests: `CUDA_VISIBLE_DEVICES= $PY tests/test_cpu.py` must end with `ALL CPU TESTS PASSED`. There are
   new checks for the J3/J4 options. If anything fails, stop and report.
3. GPU preflight for each new flag set (one GPU, ~5 min each, throwaway outputs; same $INIT and env as
   ROUND2.md):
   `$PY -u dense_train.py --init $INIT --init-sha skip --init-state full --lr 1e-5 --loss-weighting token --fine-copies 2 --preflight --output <OUT>/preflight_J3`
   `$PY -u dense_train.py --init $INIT --init-sha skip --init-state full --lr 1e-5 --loss-weighting token --parent-drop 0.1 --parent-add 0.1 --preflight --output <OUT>/preflight_J4`
   Each must end with `PREFLIGHT_OK`. Record seconds and peak GiB.
4. Submit J3 and J4 with the exact commands in `ROUND2.md`, section "Round 2b", using the same `$COMMON`
   as J1/J2 and the same job-script pattern (rerun the identical command after a restart; exit code 143 =
   interrupted, 0 = done).
5. Report on `codex-opus-exchange` as for J1/J2: right after submission (test log, preflight numbers, job
   IDs, GPUs, paths), then after each milestone evaluation. For J4, also run `analyze_cascade.py` on each new
   `evals/u*_raw` directory and include its output.
