# NEXUS Round2 closeout

Scope: `finals`; update 9000; analysis code `e7c85bf2b82fdc6ee15eb59be20486ffc7853fa4`.

Saved predictions were validated and analyzed on CPU. This package runs no model forward or training.

| Job | Weights | Euler steps/layer | Teacher rule vs D9 GT | Full tree exact | Final integer set exact | Within one cell |
|---|---|---:|---:|---:|---:|---:|
| J1_item | raw | 20 | 64/100 | 64/100 | 64/100 | 66/100 |
| J1_item | ema | 20 | 65/100 | 65/100 | 65/100 | 67/100 |
| J2_token | raw | 20 | 85/100 | 85/100 | 85/100 | 85/100 |
| J2_token | ema | 20 | 85/100 | 85/100 | 85/100 | 85/100 |
| J3_token_lr5e-6 | raw | 20 | 76/100 | 76/100 | 76/100 | 78/100 |
| J3_token_lr5e-6 | ema | 20 | 77/100 | 77/100 | 77/100 | 79/100 |
| J5_token_fine2 | raw | 20 | 87/100 | 87/100 | 87/100 | 89/100 |
| J5_token_fine2 | ema | 20 | 91/100 | 91/100 | 91/100 | 93/100 |
| J6_token_selfcorrect | raw | 20 | 72/100 | 71/100 | 72/100 | 72/100 |
| J6_token_selfcorrect | ema | 20 | 73/100 | 72/100 | 73/100 | 73/100 |

| Job / weights / steps | depth 1–9 exact counts (each /100) | First mismatch depth |
|---|---|---|
| J1_item / raw / 20 | 100, 100, 100, 100, 98, 84, 76, 66, 64 | `{"5": 2, "6": 14, "7": 8, "8": 10, "9": 2, "None": 64}` |
| J1_item / ema / 20 | 100, 100, 100, 100, 98, 85, 75, 67, 65 | `{"5": 2, "6": 13, "7": 10, "8": 8, "9": 2, "None": 65}` |
| J2_token / raw / 20 | 100, 100, 100, 100, 98, 94, 90, 85, 85 | `{"5": 2, "6": 4, "7": 4, "8": 5, "None": 85}` |
| J2_token / ema / 20 | 100, 100, 100, 100, 98, 94, 90, 85, 85 | `{"5": 2, "6": 4, "7": 4, "8": 5, "None": 85}` |
| J3_token_lr5e-6 / raw / 20 | 100, 100, 100, 100, 100, 94, 89, 80, 76 | `{"6": 6, "7": 5, "8": 9, "9": 4, "None": 76}` |
| J3_token_lr5e-6 / ema / 20 | 100, 100, 100, 100, 100, 95, 90, 81, 77 | `{"6": 5, "7": 5, "8": 9, "9": 4, "None": 77}` |
| J5_token_fine2 / raw / 20 | 100, 100, 100, 100, 100, 96, 94, 90, 87 | `{"6": 4, "7": 2, "8": 4, "9": 3, "None": 87}` |
| J5_token_fine2 / ema / 20 | 100, 100, 100, 100, 100, 98, 97, 93, 91 | `{"6": 2, "7": 1, "8": 4, "9": 2, "None": 91}` |
| J6_token_selfcorrect / raw / 20 | 100, 100, 100, 100, 100, 96, 88, 78, 72 | `{"6": 4, "7": 8, "8": 10, "9": 7, "None": 71}` |
| J6_token_selfcorrect / ema / 20 | 100, 100, 100, 100, 100, 96, 88, 79, 73 | `{"6": 4, "7": 8, "8": 9, "9": 7, "None": 72}` |

| Job / weights / steps | Vertex count accuracy | Depth9 micro F1 | Full-tree exact by GT vertex-count bucket |
|---|---:|---:|---|
| J1_item / raw / 20 | 0.7200 | 0.999404 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 12, "trees": 44}, "300-700": {"exact": 18, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J1_item / ema / 20 | 0.7100 | 0.999441 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 13, "trees": 44}, "300-700": {"exact": 18, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J2_token / raw / 20 | 0.8900 | 0.999834 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 30, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J2_token / ema / 20 | 0.9000 | 0.999834 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 30, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J3_token_lr5e-6 / raw / 20 | 0.8000 | 0.999730 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 21, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J3_token_lr5e-6 / ema / 20 | 0.8000 | 0.999759 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 22, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J5_token_fine2 / raw / 20 | 0.9100 | 0.999882 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 32, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J5_token_fine2 / ema / 20 | 0.9400 | 0.999924 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 36, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J6_token_selfcorrect / raw / 20 | 0.7200 | 0.999806 | `{"0-300": {"exact": 19, "trees": 20}, "1200+": {"exact": 20, "trees": 44}, "300-700": {"exact": 19, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |
| J6_token_selfcorrect / ema / 20 | 0.7300 | 0.999820 | `{"0-300": {"exact": 19, "trees": 20}, "1200+": {"exact": 21, "trees": 44}, "300-700": {"exact": 19, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |

Distance metrics are vertex-set distances between D9 cell centres in [-1,1] units; they are not surface metrics or raw floating-point GT errors.
Hausdorff uses the maximum directional nearest-neighbour distance. Chamfer is half the sum of the two directional mean distances.
Missing-depth9 trees are excluded by the supplied distance evaluator. Empty predictions have infinite distance. Chamfer mean includes finite trees only; per-case denominators are recorded in RESULT.json and DISTANCE_DENOMINATORS.json.
Full-tree and integer-set success always retain all 100 scheduled trees as the denominator.

J4 is not counted as completed: its saved status/log/checkpoint inventory is provided without restarting it or inferring a final result.
Final training evaluations identify run/update, not a freshly replayed checkpoint hash. Checkpoint byte hashes and recovery-state audits are separate evidence; this builder never loads model or optimizer tensors.

J3@7075 eight-case decoding and 14 boundary analyses were already published at [commit d482f4c4](https://github.com/leoguohr/suxen/commit/d482f4c414fde9fe42078952e124785d3b2ab471); they are not rerun here.

Excluded: checkpoint tensors, prediction NPZ arrays, credentials. Those remain on the server. No causal attribution is made from these comparisons.

[Compressed evidence](../evidence/round2-closeout-finals-u009000-c772caec9891250b/evidence.tgz) · [Archive SHA256](../evidence/round2-closeout-finals-u009000-c772caec9891250b/PACKAGE.json) · [Structured results](../evidence/round2-closeout-finals-u009000-c772caec9891250b/RESULT.json)
