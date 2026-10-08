# Snapshot metrics report

All tables are rebuilt from snapshot metadata. Full50 aggregates require complete source summaries, exact per-UID SHA matches and recomputed TP/FP/FN agreement. Training MSE, VAE reconstruction and actual Euler50 Gaussian generation are separate.

| Variant | Coverage | Edge micro-F1 | Face micro-F1 | Edge strict | Joint strict |
|---|---:|---:|---:|---:|---:|
| vae_mu | 50/50 | 0.999580748 | 0.995731666 | 21/50 | 15/50 |
| vae_posterior_seed0 | 50/50 | 0.999583969 | 0.995726765 | 21/50 | 15/50 |
| vae_posterior_seed1 | 50/50 | 0.999574303 | 0.995722111 | 19/50 | 14/50 |
| c0_step500 | 50/50 | 0.033957970 | 0.000450216 | 0/50 | 0/50 |
| c0_step1000 | 50/50 | 0.098206540 | 0.003817798 | 0/50 | 0/50 |
| c1_step500 | 50/50 | 0.056811290 | 0.001302258 | 0/50 | 0/50 |

## Incomplete evaluations

- C1 update904: 41/50 committed metrics, source summary incomplete. C2 update500: 39/50 committed metrics, no source summary. Their subset counts live only in `observed_subset_aggregates.csv`; these are not full50 scores.
- C2 retained update1000 and latest1049 have no evaluated metrics. Their 100 per-UID cells remain explicitly not_evaluated with blank metric fields.
- Partial evaluation follows the fixed UID order, mostly omitting larger meshes; observed subsets are not unbiased estimates of full50 performance.

## Training exposure

| Group | Updates | Direct participations per mesh | Last10 velocity MSE |
|---|---:|---:|---:|
| C0 | 1000 | 100–100 | 0.367766 |
| C1 | 904 | 90–91 | 0.356902 |
| C2 | 1049 | 104–105 | 0.350178 |

## Reading the comparisons

- `common_uid_group_comparisons.csv` gives C0→C1 at500, C0 500→1000, C1 500→904 on the same observed41, and C2 at500 on its same observed39. Every row carries the exact denominator and UID list.
- Large-mesh groups use original GT N≥1500 and N≥2000; overlapping groups are labeled. VAE-worst10 uses only VAE errors, independent of Flow results.
- `flow_vs_vae_same_uid_gaps.csv` recomputes the VAE baseline on the identical observed UIDs for every comparison. Difference is not a causal error decomposition or a reconstruction ceiling.
- The branches share data, evaluation noise, and the logged training UID/time schedule over their common updates. Architecture and execution history still differ; no causal proof is asserted.
- Figures show recorded metadata only. Actual mesh contact sheets must use committed OBJ inputs listed in `visualization_inputs.json`; none are fabricated by this script.

Rebuild: `python tools/build_metrics_report.py --root .` (matplotlib required for PNG/SVG; `--no-plots` rebuilds tables only).
