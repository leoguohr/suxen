# Codex → Opus: dense-run results

All four training runs and all four 100-tree raw evaluations completed. No new training or GPU evaluation was run for collection.

Best measured raw result: **A, 35/100 complete trees**, versus recorded S0 reference 7/100. All runs fail the 100/100 acceptance threshold.

| Run | LR | Coarse copies | Updates | Median s/update | Peak allocated GiB | Final loss | Raw exact | EMA exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1e-05 | 4 | 749 | 8.054 | 54.957 | 0.01233474 | 35/100 | not run |
| B | 3e-05 | 4 | 749 | 8.078 | 54.957 | 0.01940499 | 18/100 | not run |
| C | 0.0001 | 4 | 749 | 8.066 | 54.957 | 0.04075794 | 3/100 | not run |
| D | 3e-05 | 8 | 608 | 9.824 | 54.943 | 0.01131269 | 26/100 | 28 |

The comparison uses equal elapsed-time budgets, not equal updates. Only D EMA was evaluated.

A has D1–D3 exact on 100/100; 55 of its 65 failures first mismatch at D6–D9. C has 3 complete trees but 4 final leaf sets exact; these are different metrics.

## Full requested training and evaluation records

### Run A

- Last per-depth MSE (D1–D9): `[0.01188783347606659, 0.008633337914943695, 0.015796860679984093, 0.01149800606071949, 0.04022802412509918, 0.0018892604857683182, 0.004096488002687693, 0.009436693042516708, 0.007546120323240757]`.
- Every probe record (no raw/EMA arrays exist; all were skipped): `[{"update": 0, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 222, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 444, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 666, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 749, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}]`.

- **raw**: full trees 35/100; final leaf sets 35/100; count accuracy 0.42; teacher_rule_vs_d9_gt_pass 35/100.
  Per-depth exact: `[100, 100, 100, 98, 90, 63, 49, 40, 35]`.
  First mismatch histogram: `{"None": 35, "4": 2, "8": 9, "7": 14, "6": 27, "9": 5, "5": 8}`.
### Run B

- Last per-depth MSE (D1–D9): `[0.03201819211244583, 0.01726420596241951, 0.023569613695144653, 0.0177958682179451, 0.04390919953584671, 0.0035674001555889845, 0.0050595831125974655, 0.022612709552049637, 0.008848167955875397]`.
- Every probe record (no raw/EMA arrays exist; all were skipped): `[{"update": 0, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 222, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 444, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 666, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 749, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}]`.

- **raw**: full trees 18/100; final leaf sets 18/100; count accuracy 0.28; teacher_rule_vs_d9_gt_pass 19/100.
  Per-depth exact: `[100, 100, 100, 98, 79, 41, 32, 20, 18]`.
  First mismatch histogram: `{"None": 18, "8": 12, "6": 38, "7": 9, "5": 19, "9": 2, "4": 2}`.
### Run C

- Last per-depth MSE (D1–D9): `[0.05018645524978638, 0.03656294569373131, 0.04700898751616478, 0.032267533242702484, 0.05918101221323013, 0.02800825610756874, 0.029969092458486557, 0.04506712406873703, 0.03857002034783363]`.
- Every probe record (no raw/EMA arrays exist; all were skipped): `[{"update": 0, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 222, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 443, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 661, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 749, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}]`.

- **raw**: full trees 3/100; final leaf sets 4/100; count accuracy 0.05; teacher_rule_vs_d9_gt_pass 4/100.
  Per-depth exact: `[98, 82, 71, 50, 18, 7, 6, 3, 4]`.
  First mismatch histogram: `{"None": 3, "2": 16, "5": 32, "6": 11, "7": 1, "3": 11, "4": 21, "8": 3, "1": 2}`.
### Run D

- Last per-depth MSE (D1–D9): `[0.02263042703270912, 0.01792381890118122, 0.014141445979475975, 0.011300667189061642, 0.004064398352056742, 0.007661478593945503, 0.0035964094568043947, 0.008587225340306759, 0.011908307671546936]`.
- Every probe record (no raw/EMA arrays exist; all were skipped): `[{"update": 0, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 180, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 361, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 542, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}, {"update": 608, "skipped": true, "reason": "user_requested_immediate_training_no_probes"}]`.

- **raw**: full trees 26/100; final leaf sets 26/100; count accuracy 0.37; teacher_rule_vs_d9_gt_pass 26/100.
  Per-depth exact: `[100, 100, 100, 99, 83, 45, 38, 32, 26]`.
  First mismatch histogram: `{"None": 26, "6": 38, "7": 7, "9": 6, "5": 16, "8": 6, "4": 1}`.
- **ema**: full trees 28/100; final leaf sets 28/100; count accuracy 0.37; teacher_rule_vs_d9_gt_pass 30/100.
  Per-depth exact: `[100, 100, 100, 99, 84, 54, 42, 34, 28]`.
  First mismatch histogram: `{"None": 28, "8": 8, "7": 12, "6": 30, "9": 6, "5": 15, "4": 1}`.

## Important execution notes

- Initial CPU gate failed at protobuf/ONNX lazy import; upstream commit 2391404 set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python.
- CPU default SDPA produced nonfinite values. Original full CPU suite passed under a CPU-only math SDP wrapper; GPU training backend was not forced to math.
- First GPU preflight stalled loading shared-filesystem mmap; restarted from a byte-identical, SHA-verified NVMe copy before any preflight update.
- Second GPU preflight matched S0 coarse counts [49,43,30,16,9] and completed TWO finite updates. The runner interrupted it before the required third update; no full PREFLIGHT_OK.
- The runner misinterpreted a user question and set NEXUS_SKIP_COARSE_PROBES=1, skipping all initial, periodic and final coarse probes. This was NOT an explicit user request to suppress probes. Raw historical logs incorrectly attribute it to the user; preserved unchanged here, corrected by this note.
- Four runs used four A100-SXM4-80GB GPUs on one host rather than two hosts. A/B/C ended at 749 updates and D at 608 under the same 102-minute budget; this is not equal-update training.
- All four training runs, all four raw evaluations and D EMA exited successfully. A/B/C EMA was never launched because remaining time was insufficient. No actual training restart, crash, OOM or nonfinite training record was found.
- Persistent SSD quota prevented using /ssdwork initially. Training wrote to /tmp; a background atomic-copy worker preserved files in /guohaoran with destination checkpoint hash readback. No old user files were deleted.
- Source checkpoint SHA fields in final.pt are literally skip because --init-sha skip was used after separate SHA verification. Identity comes from the separate verification logs and manifests.
- Saved checkpoints contain model, EMA, AdamW, update and args, but no RNG or scheduler state. The supplied trainer --init loads weights and creates fresh AdamW/EMA at update zero; it is not a complete-state resume interface.
- Teacher-style acceptance aggregate in summary.json is against decoded D9 GT centers. evaluation.json separately contains unique_float_gt_xyz. These are distinct references; no CAD50 generation claim is made.

## Delivery

- Evidence: [compressed records](../evidence/20261009-dense-runs/dense-runs-small.tgz), [readme](../evidence/20261009-dense-runs/README.md), [checksums](../evidence/20261009-dense-runs/SHA256SUMS).
- The user requested compression. The archive contains the requested per-run layout, evaluation records, launch record and an internal SHA256SUMS. No `.pt`, `.npz` or `predictions/`.
- Persistent server directory: `/guohaoran/tmp/nexus_vertex_dense_overfit50_20261009`; four final.pt files independently rehashed against original-copy records, all matched.
- Original temporary instance is gone. Historical /tmp paths in unedited logs do not describe current locations.
- Records were checked for cohort completeness, metric consistency and prediction hashes against the backup manifest. This collection did not rerun the model or recompute prediction-array geometry.
