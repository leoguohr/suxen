# topology: Batch T2, gate 3 (full pipeline) plus an optional cooldown

Paste to Codex. Read all of it before running anything.

**Context.** T1 passed the topology gate on raw weights at update 4,000:
- Run B (`pair_resume_20261016/B_s0warm`): face micro-F1 0.99504 / 0.99505.
- Run A (`pair_resume_20261016/A_fresh`): 0.99425 / 0.99420.

T2 runs gate 3, the full pipeline. The vertex chat's generated D9 cells go through the topology flow and the frozen
decoder to edges and faces, which are scored against the GT mesh. No GT vertex, edge or face enters
generation.

**Code.** Branch `topology-overfit50-20261015`, head of branch (record the commit), folder
`topology/flow_overfit50_20261015/`. New in this head:
- `cascade.py`: gate 3;
- `train.py --init-checkpoint`: the cooldown;
- one new CPU test.

**Rules.** The same as T1:
- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8`;
- never touch vertex jobs or folders; reading the vertex evaluation's `predictions/` is fine;
- never modify earlier runs or checkpoints;
- outputs go under `/guohaoran/nexus_topology_flow_overfit50_20261015/`;
- no .pt files or secrets on GitHub.

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8
PY=/guohaoran/envs/nexus-algo/bin/python
CODE=<fresh checkout of the branch head>/topology/flow_overfit50_20261015
CACHE=/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006/cache
VAE=/guohaoran/nexus_topology_flow_overfit50_20261015/inputs/vae-1000.pt
ROOT=/guohaoran/nexus_topology_flow_overfit50_20261015
B=$ROOT/pair_resume_20261016/B_s0warm/checkpoint-u004000-finalstage.pt   # SHA256 abc330ba..., raw passed at u4000
A=$ROOT/pair_resume_20261016/A_fresh/checkpoint-u004000-finalstage.pt    # SHA256 92f09fab...
VEVAL=<the vertex chat's best official-seed evaluation folder: K2 final raw, 92/100; it contains predictions/>
VDATA=<checkout of branch vertex-dense-overfit50-20261009>/vertex/dense_overfit50_20261009/data/prepared_d9
cd $CODE
```

0. **Free GPUs.** Stop the separate four-GPU `quad_resume_20261016` A job; don't start quad B. The
   two-GPU pair already passed, and quad is a duplicate trajectory. Keep its folder and don't delete
   anything.
1. **CPU gate:** `CUDA_VISIBLE_DEVICES= $PY tests/test_cpu.py` must end with `CPU_GATE_OK`.
2. **Self-check, 1 GPU, about 5 min.** This re-runs gate 2 through the gate-3 code path:
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> $PY -u cascade.py --mode gt_reference --cache $CACHE --vae $VAE \
       --checkpoint $B --weights raw --flow-seeds 0 --vertex-data $VDATA --output $ROOT/T2/gt_reference_B
   ```
   - Expect face TP/FP/FN 102811/945/79 (Run B, raw, seed 0). A difference of a few faces can only
     come from GPU nondeterminism; anything larger is a bug, so stop and report.
   - The vertex GT check should list only `nexus_2k_001825`: the vertex GT has 1,279 cells, the
     approved topology mesh 1,277.
3. **Gate 3, 1 GPU, about 15 min per checkpoint:**
   ```bash
   for RUN in B A; do
     CUDA_VISIBLE_DEVICES=<gpu> $PY -u cascade.py --mode cascade --cache $CACHE --vae $VAE \
         --checkpoint ${!RUN} --weights raw --vertex-eval $VEVAL --vertex-seeds 97029000,97029001 \
         --flow-seeds 0,1 --vertex-data $VDATA --output $ROOT/T2/cascade_$RUN
   done
   ```
4. **Optional cooldown,** only if a GPU is free. Single GPU, about 2.5 h:
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> nohup $PY -u train.py --cache $CACHE --vae $VAE --output $ROOT/T2/B_cooldown \
       --init-checkpoint $B --init-checkpoint-sha abc330ba9328a79ac29c3eb0ad1989b18d2be6ebb0a941d8e3f773bb75ae9d59 --max-updates 1000 --warmup 0 --decay-fraction 1.0 --ema-decay 0.998 \
       --eval-every 500 --stop-at-pass 0 --seed 20261016 > $ROOT/T2/B_cooldown.log 2>&1 &
   ```
   This starts from Run B's u4000 weights and AdamW state, decays the LR from 1e-4 to 0 over 1,000
   updates, and starts the EMA from the raw weights. When it finishes, rerun Step 3 with
   `--checkpoint $ROOT/T2/B_cooldown/final.pt`, once with `--weights raw` and once with `--weights ema`.
   If `optimizer.load_state_dict` fails on the two-GPU checkpoint format, add `--init-optimizer 0`
   and say so.

**Report.** On `codex-opus-exchange`, add `messages/20261016-topology-003-codex-t2-gate3.md`, titled
"topology: T2 gate 3". It should contain:
- the commit, the vertex evaluation folder used and its `full_trees_exact`;
- the CPU gate result and the Step 2 totals with the vertex GT check;
- for each checkpoint and flow seed: strict and one-cell face and edge micro-F1, with the
  exact-vertex and inexact-vertex tree splits, joint strict counts, and the 10 trees with the most
  face errors;
- if Step 4 ran: the cooldown's u500 and u1000 raw/EMA evaluations and its gate-3 rerun.

Put `results.json`, each `summary.json` and `trees.jsonl` under
`evidence/topology-t2-20261016/`.
