# Round3 Batch2: completed K1/K2 results

Both arms completed all 4,500 updates and their scheduled final raw/EMA evaluations. Neither reached the strict 100/100 complete-tree criterion.

| Arm | Initialization | LR | Decay fraction | Updates | Raw, 20-step Euler | EMA, 20-step Euler |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| K1_J5_lr1e-5_d0.4 | J5 final, full state | 1e-5 | 0.4 | 4500/4500 | 90/100 | 90/100 |
| K2_J5_lr5e-6_d0.5 | J5 final, full state | 5e-6 | 0.5 | 4500/4500 | 92/100 | 91/100 |

## Completion and integrity evidence

- K1 printed JOB_COMPLETE and exited 0 at 2026-10-16 03:51:51 UTC (11:51:51 Asia/Singapore).
- K2 printed JOB_COMPLETE and exited 0 at 2026-10-16 05:03:19 UTC (13:03:19 Asia/Singapore).
- Each training log contains exactly 4,500 contiguous update records, 1 through 4,500.
- All four final evaluation files contain 100 unique UID/seed pairs: the fixed 50 training UIDs x official seeds 97029000 and 97029001. Every row has status=complete. Recounted all_levels_exact matches its summary.
- All depths 1-5 are exact in all 100 trees for both arms. K1 raw/EMA depth 6-9 exact counts: 97,93,90,90. K2 raw: 100,96,93,92; EMA: 100,95,92,91.
- 001093: K1 raw and EMA pass both official seeds. K2 raw and EMA pass 97029000, but 97029001 first mismatches at depth 7.
- Each final.pt exists and is 37,320,289,539 bytes. Only filesystem metadata was collected now; no new final-checkpoint SHA or tensor verification is claimed.

## Effective recipe and provenance

- Frozen source checkout: e7e3a326a65bc550be695104ae69e0d3435bd446; current git working tree is clean. CODE_IDENTITY.json hashes the tracked effective Python/shell/YAML source without training data.
- J5 source: /guohaoran/tmp/nexus_vertex_dense_round2_20261009/runs/J5_token_fine2/final.pt.
- Recorded source SHA256: 4abd3a09d80bd4ffc6b1557681ef84d2a182b1bf101e429a995e83c959d24f58. Historical audit/READY.json records model/EMA/Adam/full-state inspection; this collection does not repeat that large-file read.
- Common effective settings: token weighting, fine-copies 2, warmup 20, max-updates 4500, eval-every 0, final-evals raw,ema, save-minutes 45, probe-minutes 30, seed 20261015.
- Performance settings actually used: row-tokens 2816, chunk-tokens 2816, checkpoint-activations 0. Commands and validation records are included.
- K1 used decay 0.4. Do not substitute the earlier unstarted K1 decay0.2 prepared directory, or count K3/K4 as completed.
- Preparation reports are timestamped historical evidence; their 'not started' fields precede these completed runs.

## Remaining scope

The requested supplementary final EMA evaluations on all ten seeds at 2 and 20 Euler steps are not found inside either completed run directory. The included results are the scheduled 100-tree official-pair final evaluations, not 500-tree supplementary results. No new training or evaluation was launched.

## Contents and exclusions

This package contains full training/probe logs, actual configs and status, every existing summary/evaluation JSON under the two runs, completion console logs, launch commands, small recipe source files, current code identity, and checkpoint filesystem metadata. MANIFEST.json records original paths and SHA256 for every copied small file. No checkpoint tensors, prediction arrays, training data, credentials, or GPU work are included.
