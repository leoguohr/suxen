# NEXUS Round2 closeout

Scope: `decode`; update 9000; analysis code `e7c85bf2b82fdc6ee15eb59be20486ffc7853fa4`.

Saved predictions were validated and analyzed on CPU. This package runs no model forward or training.

| Job | Weights | Euler steps/layer | Teacher rule vs D9 GT | Full tree exact | Final integer set exact | Within one cell |
|---|---|---:|---:|---:|---:|---:|
| J3_token_lr5e-6 | raw | 1 | 66/100 | 64/100 | 66/100 | 66/100 |
| J3_token_lr5e-6 | raw | 20 | 76/100 | 76/100 | 76/100 | 78/100 |
| J3_token_lr5e-6 | raw | 50 | 67/100 | 66/100 | 67/100 | 70/100 |
| J3_token_lr5e-6 | ema | 1 | 68/100 | 66/100 | 68/100 | 68/100 |
| J3_token_lr5e-6 | ema | 20 | 77/100 | 77/100 | 77/100 | 79/100 |
| J3_token_lr5e-6 | ema | 50 | 67/100 | 66/100 | 67/100 | 69/100 |

| Job / weights / steps | depth 1–9 exact counts (each /100) | First mismatch depth |
|---|---|---|
| J3_token_lr5e-6 / raw / 1 | 100, 100, 100, 100, 99, 95, 84, 67, 66 | `{"5": 1, "6": 4, "7": 12, "8": 16, "9": 3, "None": 64}` |
| J3_token_lr5e-6 / raw / 20 | 100, 100, 100, 100, 100, 94, 89, 80, 76 | `{"6": 6, "7": 5, "8": 9, "9": 4, "None": 76}` |
| J3_token_lr5e-6 / raw / 50 | 100, 100, 100, 100, 98, 86, 79, 70, 67 | `{"5": 2, "6": 13, "7": 7, "8": 9, "9": 3, "None": 66}` |
| J3_token_lr5e-6 / ema / 1 | 100, 100, 100, 100, 99, 94, 81, 68, 68 | `{"5": 1, "6": 5, "7": 13, "8": 14, "9": 1, "None": 66}` |
| J3_token_lr5e-6 / ema / 20 | 100, 100, 100, 100, 100, 95, 90, 81, 77 | `{"6": 5, "7": 5, "8": 9, "9": 4, "None": 77}` |
| J3_token_lr5e-6 / ema / 50 | 100, 100, 100, 100, 98, 86, 80, 69, 67 | `{"5": 2, "6": 13, "7": 6, "8": 11, "9": 2, "None": 66}` |

| Job / weights / steps | Vertex count accuracy | Depth9 micro F1 | Full-tree exact by GT vertex-count bucket |
|---|---:|---:|---|
| J3_token_lr5e-6 / raw / 1 | 0.7200 | 0.999636 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 14, "trees": 44}, "300-700": {"exact": 20, "trees": 22}, "700-1200": {"exact": 10, "trees": 14}}` |
| J3_token_lr5e-6 / raw / 20 | 0.8000 | 0.999730 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 21, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J3_token_lr5e-6 / raw / 50 | 0.7400 | 0.999565 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 13, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 12, "trees": 14}}` |
| J3_token_lr5e-6 / ema / 1 | 0.7400 | 0.999631 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 15, "trees": 44}, "300-700": {"exact": 20, "trees": 22}, "700-1200": {"exact": 11, "trees": 14}}` |
| J3_token_lr5e-6 / ema / 20 | 0.8000 | 0.999759 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 22, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J3_token_lr5e-6 / ema / 50 | 0.7100 | 0.999550 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 13, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 12, "trees": 14}}` |

Distance metrics are vertex-set distances between D9 cell centres in [-1,1] units; they are not surface metrics or raw floating-point GT errors.
Hausdorff uses the maximum directional nearest-neighbour distance. Chamfer is half the sum of the two directional mean distances.
Missing-depth9 trees are excluded by the supplied distance evaluator. Empty predictions have infinite distance. Chamfer mean includes finite trees only; per-case denominators are recorded in RESULT.json and DISTANCE_DENOMINATORS.json.
Full-tree and integer-set success always retain all 100 scheduled trees as the denominator.

The 20-step results are saved final training evaluations; 1/50-step cases are the completed frozen-checkpoint supplement.
Final training evaluations identify run/update, not a freshly replayed checkpoint hash. Checkpoint byte hashes and recovery-state audits are separate evidence; this builder never loads model or optimizer tensors.

J3@7075 eight-case decoding and 14 boundary analyses were already published at [commit d482f4c4](https://github.com/leoguohr/suxen/commit/d482f4c414fde9fe42078952e124785d3b2ab471); they are not rerun here.

Excluded: checkpoint tensors, prediction NPZ arrays, credentials. Those remain on the server. No causal attribution is made from these comparisons.
