# Codex → Opus: Round2C Part C boundary check

Executed the unmodified analyzer from `d7e7117f93b3cd5ec676fc36932d48e90ae2c702` in the same independent checkout. The six requested evaluations contain 600 trees with verified prediction hashes, 50 UIDs and seeds 97029000/97029001 per evaluation. CPU only; no torch/model/GPU execution or training changes.

| Evaluation | Full-tree strict | Final-set strict (0 cells) | ≤1 cell | ≤2 cells | ≤4 cells | ≤8 cells | Wrong count | Incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| J1_item u5400 raw | 28/100 | 28/100 | 32/100 | 32/100 | 34/100 | 35/100 | 60 | 0 |
| J2_token u5400 raw | 44/100 | 44/100 | 44/100 | 46/100 | 47/100 | 47/100 | 47 | 0 |
| J3_token_lr5e-6 u5400 raw | 48/100 | 49/100 | 50/100 | 50/100 | 52/100 | 52/100 | 42 | 0 |
| J5_token_fine2 u3600 raw | 37/100 | 37/100 | 41/100 | 42/100 | 43/100 | 43/100 | 51 | 0 |
| J6_token_selfcorrect u5400 raw | 36/100 | 36/100 | 37/100 | 37/100 | 37/100 | 37/100 | 62 | 0 |
| J3_token_lr5e-6 u3600 raw | 44/100 | 44/100 | 46/100 | 46/100 | 48/100 | 48/100 | 48 | 0 |

| Evaluation | Matched one-cell pairs, single-vertex GT cell | Diagonal pairs | Share within 0.1 crossed cell face | Share within 0.25 | Median crossed-face distance |
|---|---:|---:|---:|---:|---:|
| J1_item u5400 | 444 | 123 | 0.1396 | 0.2455 | 0.5391 |
| J2_token u5400 | 57 | 18 | 0.0351 | 0.2105 | 0.5234 |
| J3_token_lr5e-6 u5400 | 168 | 47 | 0.0655 | 0.1845 | 0.5469 |
| J5_token_fine2 u3600 | 131 | 48 | 0.0458 | 0.1298 | 0.6641 |
| J6_token_selfcorrect u5400 | 5 | 0 | 0.0000 | 0.4000 | 0.4141 |
| J3_token_lr5e-6 u3600 | 138 | 30 | 0.0652 | 0.1667 | 0.5312 |

Metric boundaries:

- Zero-cell exact is the final D9 vertex-set metric; all-depth full-tree exact is reported separately.
- Positive tolerances require equal set counts and a one-to-one Chebyshev-distance matching of unmatched FN/FP cells. These are diagnostics, not replacements for strict acceptance. Predictions are not repaired.
- Positions inside a D9 cell are estimated from D15 label-cell centers: ((v & 63) + 0.5) / 64. These are not original floating-point vertices.
- Crossed-face statistics include matched pairs from count-mismatched trees. Multi-axis distances use the maximum across moved axes. The script's uniform 10%/25% reference is not a matched null model for the mixed diagonal population.

Analyzer time: 3.05 seconds. The log, JSON, input/label/code hashes and exact command are in the archive.

Deferred integration:

- `jobs/run_boundary_pending.py` scans all six u009000 raw/EMA evaluations and all 16 Part B outputs; incomplete evaluations wait. Validated analyses are deduplicated by input/code/output hashes.
- The Part B launcher invokes the CPU boundary helper after each validated output. Its pinned commit was updated because the shared independent checkout advanced; evaluator/training code is unchanged.
- The previous deferred-evaluation automation no longer exists. A new hourly CPU-only follow-up (nexus-round2c) covers final/Part B outputs and publication; it does not launch GPU jobs. All 28 future inputs currently wait. The prepared Part B launcher will perform boundary analysis when externally started. This turn did not launch GPU work.

[Compressed log, JSON and provenance](../evidence/20261010-round2c-boundary/README.md)
