# Codex → Opus: Round2b preparation

**J5/J6 are prepared and awaiting the later launch instruction. This workflow has submitted neither job.**

Approved commit: `2108acb08c6d5ca1e43c67da2159213f0d36080f`. All preparation gates passed from recorded evidence: `True`.

| Gate | Recorded state | Exit | Success marker | Finite updates | Peak allocated GiB |
|---|---|---:|---|---:|---:|
| CPU | passed | 0 | True | n/a | n/a |
| J5 | passed | 0 | True | 3 | 55.00607776641846 |
| J6 | passed | 0 | True | 3 | 55.00297784805298 |

Exact gate commands, environment/identity receipts, elapsed time, exit codes and preflight rows are in the compressed evidence. Failed or absent gates are not treated as passes.

| Job | Saved state | Last train-tail update | Evaluation directories observed |
|---|---|---:|---:|
| J1 | training | 331 | 0 |
| J2 | training | 298 | 0 |
| J3 | training | 81 | 0 |
| J4 | training | 59 | 0 |
| J5 | unavailable | unavailable | 0 |
| J6 | unavailable | unavailable | 0 |

## Boundaries and next action

- J5/J6 are prepared for a later user launch; this preparation workflow submits no formal jobs.
- J1-J4 keep their c00ebcc shared wrapper. J5/J6 use isolated repo_round2b pinned to 2108acb.
- Formal J5 adds only --fine-copies 2; J6 adds only --parent-drop 0.1 --parent-add 0.1 to J2 COMMON.
- Prescribed preflights omit formal schedule flags: defaults seed20261009, warmup50, max-updates1e9, decay0; formal uses seed20261010, warmup20, max-updates9000, decay0.2.
- Preflight synchronizes before EMA.update, not after it. Recorded seconds are preflight timings, not fully synchronized train+EMA speed estimates.
- Three largest-object updates do not guarantee 9000-update memory peaks, total runtime or completion ETA.
- J6 samples a per-item drop probability uniformly in [0,0.1], then Bernoulli-drops parents; realized dropped fraction can exceed 0.1, with at least one kept.
- J6 requested spurious-parent count is rounded U[0,0.1]*original-count; bounds, uniqueness and GT filtering can reduce the actual additions.
- J6 teaches extra parents to predict no children. Inference is unchanged and cannot recreate a true branch once its parent is absent.
- J6 consumes additional RNG draws and changes item shapes. The common seed pairs object order, not actual J2/J6 noise tensors.
- Job records are snapshots of saved files, possibly partial or duplicated after resume; they do not prove a current live pod process.
- Existing cascade outputs are packaged when present; this builder runs no analysis.
- Reporting is configured: active automation nexus-round2 discovers new completed milestones hourly from small saved files for all six jobs, publishes only new completed milestones and J6 CPU cascade analysis, and stays quiet when unchanged. It reports expired SSH access; it performs no continuous GPU monitoring or training changes.

After the later authorized launch, report submission identity. Hourly milestone discovery is configured as active automation `nexus-round2`: publish newly completed milestones and J6 CPU cascade results, staying quiet when saved evidence is unchanged.

No `.pt`, `.npz`, predictions or credentials are included. Server originals remain in `/guohaoran`; upload transit uses memory only.

[Compressed preparation evidence](../evidence/20261010-round2b-preparation/README.md)
