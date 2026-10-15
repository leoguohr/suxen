# topology: Batch T1, paper-scale topology flow overfit50 (2 GPUs)

Paste to Codex. Read all of it before running anything.

**Context.** This is Track 3, the topology stage. The user has accepted the frozen OwnAE-v2 VAE (step
36220; face micro-F1 0.99573 on the 50 objects). This batch trains the topology flow at the paper's
scale: the S0 DiT (36 x 1536) plus the VecSet encoder (8 x 2048), about 2.33B parameters. It uses the
same 50 Nexus2K objects as the vertex gate and is conditioned on GT vertices and the 8,192-point
clouds.

**Code.** Branch `topology-overfit50-20261015` of leoguohr/suxen, folder
`topology/flow_overfit50_20261015/`. Use the head of that branch and record its commit in every
report. `RUN_SHEET.md` has every command; follow it exactly.

**Rules**
- Every command sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8`.
- Use two GPUs that the vertex chat's jobs are not using. Never stop or touch vertex processes, vertex
  run folders or branch `vertex-dense-overfit50-20261009`.
- The fixed-50 cache, `vae-1000.pt` and S0 `checkpoint-034000.pt` are read-only inputs. Never modify or
  delete earlier experiment folders or checkpoints.
- All outputs go under `/guohaoran/nexus_topology_flow_overfit50_20261015`. If `/guohaoran` has less
  than about 200 GB free, use `/ssd/guohaoran/nexus_topology_flow_overfit50_20261015` instead.
- No .pt files, passwords, keys or tokens on GitHub.
- If the S0 checkpoint is not at the path in `RUN_SHEET.md`, find a byte-identical copy by its SHA256
  (`3919d924...`). If none exists, skip Run B and say so.

**Steps** (details in `RUN_SHEET.md`)
1. CPU gate and cache check. On any failure, stop and report the failing output.
2. GPU 2: `evaluate.py --mode mu` must match the archived VAE result exactly. Then run
   `--mode noise` for the tolerance curve.
3. GPU 1: speed check, 20 updates. On out-of-memory, use `--chunk-tokens 27000`, then 14000. If an
   update takes more than 30 s, stop and report.
4. GPU 1: Run A, fresh initialization, 10,000 updates.
5. GPU 2, after step 2: Run B, warm start from S0, otherwise identical.
6. When each run completes: the sampler sweep (`RUN_SHEET.md`, Step 6).

**Reporting.** On branch `codex-opus-exchange`, add
`collaboration/codex-opus/messages/20261015-topology-002-codex-t1-startup.md`, titled
"topology: T1 startup". Put the small raw files under
`collaboration/codex-opus/evidence/topology-t1-20261015/`. Include:
- the code commit, the GPUs used and the output root;
- the CPU gate result and the `cache_check` report;
- the mu check: exact match or not, the totals, and the first mismatches if any;
- the tolerance table: for each alpha, edge F1, face F1, strict count, overflow objects and latent RMSE;
- the speed check: s/update, peak GiB, flash vs SDPA difference, `--chunk-tokens` used and projected
  hours;
- the first 500 updates of each run: loss, t-bin MSE at updates 100, 250 and 500, the largest gradient
  norm, and the probe latent RMSE at 250 and 500.

Then add a new message at every milestone (every 2,000 updates) and at the end. Each one carries the
new `evals.jsonl` rows, the per-size tables, and Run A and Run B side by side at equal updates.

Start every topology message and title with "topology:". In the shared `README.md` and `STATUS.md`,
append at most one link line; don't edit the vertex chat's text.
