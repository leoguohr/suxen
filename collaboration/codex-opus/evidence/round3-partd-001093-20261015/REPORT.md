# Round3 Part D: 001093 follow-up

CPU-only analysis of 74 saved full-tree trajectories (110 NPZ files, including separately stored sweep layers), the original 8192-point training condition, and hash-bound labels. No training, GPU, model forward, checkpoint loading, repository edits, new generation or repair.

## Findings

- **J5's target bit regressed between saved raw updates 7200 and 9000, in both official seeds.** The four requested earlier milestones all retain it. No intermediate saved prediction identifies the exact transition update.
- **The target is not persistently absent from the common start.** Round1 A/final (update 749) retains it in both seeds. Its S0 ancestor misses it for one seed, then Round1 A recovers it. S1 is a comparison branch, not the J2/J5 initializer.
- **Local condition sampling is sparse, but the target cell is not unusually empty by the requested cell-count comparison:** 1 point, versus other-cell min/p05/median 0/0/1; 499/1151 other occupied cells have fewer points. This does not establish the cause of the miss.
- **J2's three all-10-seed failures have different error patterns.** 001792 changes error location across seeds and passes both 5-step trees. 000167 has a stable extra bit; 001232 has a stable missing bit plus a stable extra bit. Both also have seed-dependent errors.

## Density conventions

Depth-9 integer triples identify cells. Actual decoding uses their centers `q+0.5`; original float GT vertices retain subcell offsets. Counts at radius 4 differ between these centers, so all three interpretations—including literal integer coordinates—are reported separately below. Radius neighborhoods are Euclidean and inclusive. For per-cell comparisons, union counts count each condition point once across that cell's GT vertices; per-vertex min/mean/median and all individual counts are also in `density.json`.

## Stable direct J2 errors

| UID | Depth / cell | Type / training label | 20-step errors | 5-step errors | 20-step estimate range |
|---|---|---|---:|---:|---|
| nexus_2k_000167 | 6 / (59, 47, 36) | FP / 0 (absent) | 10/10 | 2/2 | 0.9961151481 to 1.002778053 |
| nexus_2k_001232 | 7 / (14, 30, 29) | FP / 0 (absent) | 10/10 | 2/2 | 0.9986773729 to 1.001618981 |
| nexus_2k_001232 | 7 / (31, 102, 23) | FN / 1 (present) | 10/10 | 2/2 | -0.0005535557866 to 0.002604186535 |

Training labels contain the missing GT cells, including 001232 depth7 (31,102,23). The two stable false positives are correctly absent from the labels; they must not be described as missing GT. The full 12 observations and all 8 sibling estimates for each candidate are in `recurrence.json`. `recurrence_cells.json/md/tsv` and `recurrence_siblings.tsv` list all 28 repeated global error cells, explicitly distinguishing direct errors from inherited descendants.

## Detailed historical observations



- Target: depth6 (18,31,35), parent (9,15,17), child bit3; GT=1; threshold=0.5.
- Actual shared start: Round1 A/final.pt (update749, SHA 1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9).
- S0@34000 is its ancestor. S1 is an independent sweep comparison.
- CPU array recomputation agrees with every available evaluation row; no model replay.

## When the target miss was observed

- J5 raw: target present in both seeds at updates1800,3600,5400,7200; missing in both at9000.
- Each official seed therefore brackets the observed regression between7200 and9000. Intermediate predictions are not available; the exact transition update is unknown.
- Round1 A, the direct common initialization, retains the target for both official seeds. Earlier S0@34000 misses it for97029000 but retains it for97029001. Thus this is not a continuously missing bit inherited unchanged from S0.
- J2 retains this bit in every inspected milestone and terminal prediction. A correct target bit does not imply a correct complete tree.
- Round1 A/B/C raw and D raw/EMA all retain the bit; all ten of these whole trees still fail elsewhere.

| Source | Update / cumulative | Weights | Seed | Estimate | Target | Whole tree | First mismatch |
|---|---:|---|---:|---:|---|---|---:|
| sweep/S0 | 10000 / 34000 | raw | 97029000 | 0.0225583762 | miss | fail | 3 |
| sweep/S0 | 10000 / 34000 | raw | 97029001 | 1.00412154 | occupied | fail | 3 |
| sweep/S1 | 10000 / 34000 | raw | 97029000 | parent absent | parent absent | fail | 1 |
| sweep/S1 | 10000 / 34000 | raw | 97029001 | 1.04613888 | occupied | fail | 4 |
| round1/A | 749 / None | raw | 97029000 | 1.00468564 | occupied | fail | 6 |
| round1/A | 749 / None | raw | 97029001 | 1.00310254 | occupied | fail | 6 |
| round1/B | 749 / None | raw | 97029000 | 1.01584566 | occupied | fail | 5 |
| round1/B | 749 / None | raw | 97029001 | 1.01989925 | occupied | fail | 6 |
| round1/C | 749 / None | raw | 97029000 | 0.971404076 | occupied | fail | 2 |
| round1/C | 749 / None | raw | 97029001 | 0.992580891 | occupied | fail | 4 |
| round1/D | 608 / None | ema | 97029000 | 1.01582551 | occupied | fail | 6 |
| round1/D | 608 / None | ema | 97029001 | 1.01228201 | occupied | fail | 6 |
| round1/D | 608 / None | raw | 97029000 | 1.02701902 | occupied | fail | 6 |
| round1/D | 608 / None | raw | 97029001 | 1.02556336 | occupied | fail | 6 |
| round2_milestones/J5_token_fine2 | 1800 / None | raw | 97029000 | 1.00064671 | occupied | fail | 6 |
| round2_milestones/J5_token_fine2 | 1800 / None | raw | 97029001 | 0.998414874 | occupied | fail | 6 |
| round2_milestones/J5_token_fine2 | 3600 / None | raw | 97029000 | 1.0004611 | occupied | fail | 6 |
| round2_milestones/J5_token_fine2 | 3600 / None | raw | 97029001 | 1.0038836 | occupied | fail | 6 |
| round2_milestones/J5_token_fine2 | 5400 / None | raw | 97029000 | 1.01181448 | occupied | fail | 7 |
| round2_milestones/J5_token_fine2 | 5400 / None | raw | 97029001 | 1.01306319 | occupied | fail | 7 |
| round2_milestones/J5_token_fine2 | 7200 / None | raw | 97029000 | 1.01388979 | occupied | fail | 6 |
| round2_milestones/J5_token_fine2 | 7200 / None | raw | 97029001 | 1.01423514 | occupied | fail | 6 |
| round2_terminal_supplement/J5_token_fine2 | 9000 / None | raw | 97029000 | 0.000550400466 | miss | fail | 6 |
| round2_terminal_supplement/J5_token_fine2 | 9000 / None | raw | 97029001 | 7.50161707e-05 | miss | fail | 6 |
| round2_terminal_supplement/J5_token_fine2 | 9000 / None | ema | 97029000 | 0.000355202705 | miss | fail | 6 |
| round2_terminal_supplement/J5_token_fine2 | 9000 / None | ema | 97029001 | 0.000172680244 | miss | fail | 6 |
| round2_milestones/J2_token | 1800 / None | raw | 97029000 | 1.00663853 | occupied | fail | 6 |
| round2_milestones/J2_token | 1800 / None | raw | 97029001 | 1.00603211 | occupied | fail | 6 |
| round2_milestones/J2_token | 3600 / None | raw | 97029000 | 1.00231647 | occupied | fail | 6 |
| round2_milestones/J2_token | 3600 / None | raw | 97029001 | 1.00388348 | occupied | fail | 6 |
| round2_milestones/J2_token | 5400 / None | raw | 97029000 | 1.00239301 | occupied | fail | 6 |
| round2_milestones/J2_token | 5400 / None | raw | 97029001 | 1.00290704 | occupied | fail | 7 |
| round2_milestones/J2_token | 7200 / None | raw | 97029000 | 1.00610232 | occupied | fail | 2 |
| round2_milestones/J2_token | 7200 / None | raw | 97029001 | 1.00564146 | occupied | fail | 5 |
| round2_terminal_supplement/J2_token | 9000 / None | raw | 97029000 | 1.00404012 | occupied | pass | None |
| round2_terminal_supplement/J2_token | 9000 / None | raw | 97029001 | 1.00349283 | occupied | fail | 8 |
| round2_terminal_supplement/J2_token | 9000 / None | ema | 97029000 | 1.0036633 | occupied | pass | None |
| round2_terminal_supplement/J2_token | 9000 / None | ema | 97029001 | 1.00193036 | occupied | fail | 8 |

## Limits

- Saved milestones are sparse; earliest observed failure does not identify the exact update when the miss began.
- Identical seeds do not establish identical actual noise when generated parent sets/orders differ.
- Round2 milestone arrays do not store actual noise; these are historical observations, not paired causal tests.
- Terminal raw/EMA rows are supplemental; EMA and raw histories are reported separately.

## Missing historical outputs

- /guohaoran/tmp/nexus_vertex_dense_overfit50_20261009/A/eval_ema: EMA evaluation not present; no claim it was run
- /guohaoran/tmp/nexus_vertex_dense_overfit50_20261009/B/eval_ema: EMA evaluation not present; no claim it was run
- /guohaoran/tmp/nexus_vertex_dense_overfit50_20261009/C/eval_ema: EMA evaluation not present; no claim it was run

## Input density results


- Fixed training condition: 8192 XYZ+normal slots; 8192 unique XYZ values.
- Occupied depth-6 cells: 1152; comparison cells excluding target: 1151.
- Coordinates: u=(XYZ+1)*256; depth-6 cells use [8c,8(c+1)); radii are Euclidean and inclusive.
- Main radius interpretation: true float GT vertices. D9 native decoder centers q+0.5 and literal integer q are separate checks.
- Counts do not deduplicate condition slots. No model, GPU, resampling or fitted alignment used.

| Statistic | Target | Other-cell min | p05 | median | Other cells strictly lower |
|---|---:|---:|---:|---:|---:|
| inside_cell | 1 | 0 | 0 | 1 | 499/1151 (43.354%) |
| d9_decoded_center/radius2/union_condition_slots | 0 | 0 | 0 | 0 | 0/1151 (0.000%) |
| d9_decoded_center/radius4/union_condition_slots | 2 | 0 | 0 | 1 | 848/1151 (73.675%) |
| source_float_gt/radius2/union_condition_slots | 0 | 0 | 0 | 0 | 0/1151 (0.000%) |
| source_float_gt/radius4/union_condition_slots | 2 | 0 | 0 | 1 | 852/1151 (74.023%) |

| Center convention | Vertex centers (D9 units) | Within 2 | Within 4 |
|---|---|---|---|
| source_float_gt | [[145.59100341796875, 250.17376375198364, 285.12769508361816], [148.35892486572266, 250.17376375198364, 282.05692291259766]] | [0, 0] | [2, 0] |
| d9_decoded_center | [[145.5, 250.5, 285.5], [148.5, 250.5, 282.5]] | [0, 0] | [2, 1] |
| literal_d9_integer | [[145.0, 250.0, 285.0], [148.0, 250.0, 282.0]] | [0, 0] | [0, 0] |

Every occupied cell and every vertex count is included in density.json.
The input, label and manifest hashes are verified against both J2 and J5's actual training checkouts.
Independent direct squared-distance calculations reproduce all KD-tree radius and union counts.
Radius-4 counts depend on the vertex convention: true float GT gives [2,0], decoded centers [2,1], literal integer corners [0,0].
These counts describe local sampling density; they do not establish the causal reason for the model miss.

## J2 per-object results


CPU-only recomputation: 36 saved predictions, 3 objects × (10 seeds at 20 steps + 2 official seeds at 5 steps).
No training, model forward, GPU use, repository modification, or new generation.

| Object | 20-step exact | 5-step exact | First-error depths (20 steps) | Identical first-error set across 10 seeds |
|---|---:|---:|---|---|
| nexus_2k_001792 | 0/10 | 2/2 | {'8': 6, '6': 2, '7': 2} | False |
| nexus_2k_000167 | 0/10 | 0/2 | {'6': 9, '5': 1} | False |
| nexus_2k_001232 | 0/10 | 0/2 | {'5': 3, '6': 5, '7': 2} | False |

A recurring missed GT cell does not mean each tree has only one wrong bit. New direct errors and descendants inherited from earlier errors are separated in recurrence.json.


## Evidence and execution

- `history.json`: 38 per-tree historical rows, 9-level verification, provenance and missing-output inventory.
- `density.json`: all 1152 occupied D6 cells and all per-vertex counts under each explicit coordinate convention.
- `recurrence.json`: 36 trees, complete per-depth FP/FN, direct/inherited classification, candidate estimates and siblings.
- `recurrence_cells.*`, `recurrence_siblings.tsv`: per-seed tables for recurring cells.
- Ad-hoc scripts, execution logs, input manifests, `VERIFICATION.json`, and `EXECUTION.txt` are included in the compressed evidence. No `.pt`, `.npz`, or `predictions/` folder is uploaded.
- Every source input hash was independently rechecked before packaging; the frozen repository remains clean at e7e3a326a65bc550be695104ae69e0d3435bd446.

Saved seeds alone do not prove identical actual noise when parent sets/order differ. Sparse historical observations bound the regression interval, not its precise cause. These outputs support follow-up analysis; no training intervention is prescribed or executed in this task.
