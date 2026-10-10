# Codex → Opus: Round2C Part D dead-parent analysis

Ran the unmodified `analyze_cascade.py` at `427498e88912f351ff50b70aef5888999b2af26b` in the same independent checkout. The exact six Part C evaluations were reused: 600 saved trees, each evaluation with the original 50 UIDs × seeds 97029000/97029001. Prediction, input JSON and label hashes were verified against the previous Part C receipt. All nine levels have finite saved estimates.

CPU only. No model replay, GPU use, regeneration, threshold change, prediction repair, or training changes. The trainer, evaluator and common-code hashes still match Part C.

## Depth 9 summary

| Evaluation | All-depth tree exact | Dead real parents | FN under dead parents / local FN | Strongest bit hits GT / dead | Max estimate ≥0.25 / dead |
|---|---:|---:|---:|---:|---:|
| J1_item u5400 raw | 28/100 | 11 | 11/505 (2.2%) | 2/11 (18.2%) | 9/11 (81.8%) |
| J2_token u5400 raw | 44/100 | 6 | 6/66 (9.1%) | 1/6 (16.7%) | 0/6 (0.0%) |
| J3_token_lr5e-6 u5400 raw | 48/100 | 7 | 7/197 (3.6%) | 1/7 (14.3%) | 2/7 (28.6%) |
| J5_token_fine2 u3600 raw | 37/100 | 7 | 7/134 (5.2%) | 1/7 (14.3%) | 2/7 (28.6%) |
| J6_token_selfcorrect u5400 raw | 36/100 | 97 | 97/109 (89.0%) | 14/97 (14.4%) | 0/97 (0.0%) |
| J3_token_lr5e-6 u3600 raw | 44/100 | 4 | 4/166 (2.4%) | 0/4 (0.0%) | 1/4 (25.0%) |

## Interpretation boundaries

- A dead real parent was generated and belongs to GT, but all eight predicted occupancy bits are empty. It is different from an extra wrong parent disappearing.
- The FN share counts direct GT children at the reported depth, not final D9 vertices lost at all later depths. Counts aggregate parents across objects and seeds; they are not object-level success rates.
- A correct argmax only indicates ordering of the saved continuous estimates. It does not establish that changing a threshold or forcing a child would fix generation.
- The 0.25 field is descriptive. The original 0.5 occupancy threshold remains unchanged.
- Local plus inherited FN/FP were checked against the original per-depth evaluation totals. Dead-parent and sibling counts passed bounds checks. The full depth1–9 tables are in the log and JSON.

Analyzer and post-run verification time: 45.06 seconds. Exact command and SHA256 provenance are in `part_d_receipt.json`.

## Manual follow-up only

The user explicitly cancelled all scheduled tasks. No hourly follow-up, timer, background scanner or automatic GPU dispatch is enabled.
- When the final raw/EMA evaluations exist, rerun both boundary and dead-parent cascade analysis on those outputs.
- When each Part B 1-step/50-step raw/EMA evaluation completes, run the same two CPU analyses and publish the complete results.
- User-facing checkpoints: ask again on 2026-10-11 after 08:00 UTC+8 for J3/J1/J2, and on 2026-10-12 after 08:00 UTC+8 for J5. These are suggested contact times, not timers or guaranteed completion dates.
- GPU work remains separately gated on JOB_COMPLETE, final.pt, complete final raw/EMA results, and verified exclusive access to the corresponding released GPU. This Part D request did not launch Part B.

[Compressed JSON, log and provenance](../evidence/20261010-round2c-cascade-dead/README.md)
