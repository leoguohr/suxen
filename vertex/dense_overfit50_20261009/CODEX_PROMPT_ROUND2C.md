You are running two small add-on checks for Round 2 of the NEXUS Vertex Diffusion 50-object overfit.
Do not change the six running jobs, their commands, code checkouts or outputs. Same rules as before:
store everything under `/guohaoran` or `/ssd/guohaoran`, never modify or delete earlier experiment
folders or checkpoints, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8`.

Pull branch `vertex-dense-overfit50-20261009` into a NEW checkout (not the one the jobs run from).
The only code change is in `analyze_cascade.py`: a new table, "local FN with a wrong cell next to it",
which tells whether a missed vertex was lost or only placed one cell off. `dense_train.py`,
`common.py` and `evaluate.py` are unchanged.

## Part A: now, CPU only
Run the cascade analysis on these existing raw evaluations (all under `.../runs/<job>/evals/`):
J1 u3600, J2 u3600, J3 u1800, J3 u3600, J5 u1800, J6 u3600.
```bash
$PY analyze_cascade.py --eval-dir <J1>/evals/u003600_raw --eval-dir <J2>/evals/u003600_raw \
  --eval-dir <J3>/evals/u001800_raw --eval-dir <J3>/evals/u003600_raw --eval-dir <J5>/evals/u001800_raw \
  --eval-dir <J6>/evals/u003600_raw --output <NEW_DIR>/round2_cascade_shift.json | tee <NEW_DIR>/round2_cascade_shift.log
```
Push the log and JSON to `codex-opus-exchange` as message 011.

## Part B: decoding test, GPU, evaluation only
Question: are the remaining errors in the model, or in the 20-step sampler? With this rectified flow,
`--steps 1` returns noise + v(noise, t=0), which is the model's own estimate of each child bit's
occupancy probability, thresholded at 0.5. `--steps 50` is a finer ODE. The final evaluations of the
jobs already give `--steps 20`.

When a job finishes (`JOB_COMPLETE`, `final.pt` exists), use its freed GPU. Do not share a GPU with a
running job. Order: J3, then J1, then J2. J5 later, when it finishes. For each finished job:
```bash
D=<NEW_DIR>/decode_test
for W in ema raw; do
  CUDA_VISIBLE_DEVICES=<free uuid> $PY -u evaluate.py --checkpoint <job>/final.pt --weights $W --steps 1  --output $D/<job>_${W}_s1
done
for W in ema raw; do
  CUDA_VISIBLE_DEVICES=<free uuid> $PY -u evaluate.py --checkpoint <job>/final.pt --weights $W --steps 50 --output $D/<job>_${W}_s50
done
$PY analyze_cascade.py --eval-dir $D/<job>_ema_s1 --eval-dir $D/<job>_raw_s1 --output $D/<job>_s1_cascade.json
```
`--steps 1` should take a few minutes, `--steps 50` about 40 min. Push each `summary.json`, the
cascade log/JSON and the job's own final `evals/u009000_{raw,ema}/summary.json` side by side, as
messages 012+. No `.pt` and no `predictions/`.
