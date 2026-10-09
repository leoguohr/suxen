# Round 2: two unattended jobs from round-1 run A (loss weighting test)

## Where round 1 left off (2026-10-09, warm start from S0@34000, ~102 min per run)

| Run | LR | Exact trees /100 | <300 vertices | 300–700 | 700–1200 | 1200+ |
|---|---|---|---|---|---|---|
| S0@34000 | – | 7 | 7/20 | 0/22 | 0/14 | 0/44 |
| **A** | **1e-5** | **35** | 19/20 | 15/22 | 1/14 | **0/44** |
| B | 3e-5 | 18 | 15/20 | 2/22 | 1/14 | 0/44 |
| C | 1e-4 | 3 | 3/20 | 0/22 | 0/14 | 0/44 |
| D (8 coarse copies) | 3e-5 | 26 (EMA 28) | 18/20 | 8/22 | 0/14 | 0/44 |

- Depths 1–3 are now exact in 100/100 trees (S0: 98/88/58). Failures moved to depths 6–9.
- 1e-5 is the right LR for continuing this checkpoint. Higher LRs disrupted it (C: loss doubled
  after warmup, with 30% of updates clipped).
- A's loss was still falling at the end (0.023 → 0.0093).
- What fails now is the large objects. 33 of A's 65 failures had only 1–2 wrong cells at their
  first wrong depth, and the error then spread to the deeper levels.

## The loss-weighting question

Round 1 used `--loss-weighting item`: each (object, depth) item has equal weight, and errors are
averaged inside the item. A token's weight is therefore 1/(tokens in its item). At depth 9 a
token of the 2,528-parent object gets ~1/140 the weight of a token of an 18-parent object, yet
the large object needs every one of its thousands of bits right.

`--loss-weighting token` is the plain mean over all tokens of the update, the standard
flow-matching loss (the paper does not specify one). The ×4 stratified-t copies of depths 1–5
count as real tokens, so those levels keep about 28% of the loss and stay protected (56% under
`item`).

| Item in one 8-object update | `item` | `token` |
|---|---|---|
| depth 9 of the largest object | 1.4% | ~7% |
| depth 9 of an 18-vertex object | 1.4% | ~0.05% |
| depths 1–5 of all objects | 56% | ~28% |

## The two jobs (one GPU each, one per server)

Both start from round-1 A `final.pt` (SHA256 `1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9`)
with its AdamW and EMA state, and use the same seed. Every update therefore sees the same objects
and noise in both jobs. **The only difference is `--loss-weighting`.**

| Job | `--loss-weighting` | Question |
|---|---|---|
| J1 | `item` | Control: how far does more training with A's recipe go? |
| J2 | `token` | Does equal per-token weight fix the large objects? |

Common settings: `--lr 1e-5 --warmup 20 --max-updates 9000 --decay-fraction 0.2` (constant LR,
then linear decay to 0 from update 7,200), 8 objects per update, ×4 copies at depths 1–5,
EMA 0.995. Probes run every 30 min. A full 100-tree raw evaluation runs every 1,800 updates
(~4 h), and raw + EMA evaluations run at the end. The job stops early if any evaluation reaches
100/100.

Expected duration: about 8 s/update, so 9,000 updates ≈ 20 h. With evaluations (18 min each),
probes and saves, the total is about 23–24 h.

## Unattended behaviour of `dense_train.py` (when `--train-minutes` is omitted)

- Every `--save-minutes` and before every milestone evaluation it writes `latest.pt`
  (model + EMA + AdamW + update, ~37 GB, atomic).
- Rerunning the same command after any stop or restart resumes from `latest.pt`. If no
  `latest.pt` exists yet, earlier partial files are moved into `restart-<time>/` and training
  starts fresh.
- On SIGTERM it saves `latest.pt` and exits with code 143. On completion it exits 0 and renames
  `latest.pt` to `final.pt`. Rerunning a completed job prints `ALREADY_COMPLETE` and exits 0.
- Outputs:
  - `train.jsonl`. After a resume, updates since the last save appear twice; keep the last
    entry per update.
  - `probes.jsonl`
  - `evals.jsonl`, plus `evals/u<update>_<raw|ema>/summary.json` and `evaluation.json`
  - `status.json`
  - `config*.json`

## Commands

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8
PY=/guohaoran/envs/nexus-algo/bin/python
INIT=<local copy of round-1 A/final.pt>     # SHA256 1bd4ced4...b55f9
OUT=/guohaoran/<dir>/nexus_vertex_dense_round2   # or /ssd/guohaoran/...
COMMON="--init $INIT --init-sha 1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9 --init-state full \
  --lr 1e-5 --warmup 20 --max-updates 9000 --decay-fraction 0.2 --eval-every 1800 --final-evals raw,ema \
  --save-minutes 45 --probe-minutes 30 --seed 20261010"
# J1 (server 1):  $PY -u dense_train.py $COMMON --loss-weighting item  --output $OUT/J1_item
# J2 (server 2):  $PY -u dense_train.py $COMMON --loss-weighting token --output $OUT/J2_token
```

After a resume, the SHA check is skipped (state comes from `latest.pt`). On first start it hashes
`--init`, which takes about 1 min for 37 GB.

## CPU margin check (no GPU)

`python analyze_margins.py --eval-dir <round1>/A/eval_raw --eval-dir <round1>/D/eval_ema --output margins.json`

For each failed tree it reports the sampler's continuous values for the wrong bits at the first
wrong depth. Values near 0.5 point to sampling or near-miss problems; values near 0/1 mean the
model was confidently wrong.
