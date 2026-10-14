# NEXUS Round2 closeout

Scope: `decode`; update 9000; analysis code `e7c85bf2b82fdc6ee15eb59be20486ffc7853fa4`.

Saved predictions were validated and analyzed on CPU. This package runs no model forward or training.

| Job | Weights | Euler steps/layer | Teacher rule vs D9 GT | Full tree exact | Final integer set exact | Within one cell |
|---|---|---:|---:|---:|---:|---:|
| J5_token_fine2 | raw | 1 | 85/100 | 85/100 | 85/100 | 87/100 |
| J5_token_fine2 | raw | 20 | 87/100 | 87/100 | 87/100 | 89/100 |
| J5_token_fine2 | raw | 50 | 77/100 | 76/100 | 77/100 | 79/100 |
| J5_token_fine2 | ema | 1 | 88/100 | 88/100 | 88/100 | 90/100 |
| J5_token_fine2 | ema | 20 | 91/100 | 91/100 | 91/100 | 93/100 |
| J5_token_fine2 | ema | 50 | 77/100 | 76/100 | 77/100 | 81/100 |

| Job / weights / steps | depth 1–9 exact counts (each /100) | First mismatch depth |
|---|---|---|
| J5_token_fine2 / raw / 1 | 100, 100, 100, 100, 99, 95, 91, 89, 85 | `{"5": 1, "6": 4, "7": 4, "8": 2, "9": 4, "None": 85}` |
| J5_token_fine2 / raw / 20 | 100, 100, 100, 100, 100, 96, 94, 90, 87 | `{"6": 4, "7": 2, "8": 4, "9": 3, "None": 87}` |
| J5_token_fine2 / raw / 50 | 100, 100, 100, 100, 98, 92, 87, 79, 77 | `{"5": 2, "6": 7, "7": 5, "8": 8, "9": 2, "None": 76}` |
| J5_token_fine2 / ema / 1 | 100, 100, 100, 100, 99, 94, 92, 92, 88 | `{"5": 1, "6": 5, "7": 2, "9": 4, "None": 88}` |
| J5_token_fine2 / ema / 20 | 100, 100, 100, 100, 100, 98, 97, 93, 91 | `{"6": 2, "7": 1, "8": 4, "9": 2, "None": 91}` |
| J5_token_fine2 / ema / 50 | 100, 100, 100, 100, 98, 92, 87, 79, 77 | `{"5": 2, "6": 7, "7": 5, "8": 8, "9": 2, "None": 76}` |

| Job / weights / steps | Vertex count accuracy | Depth9 micro F1 | Full-tree exact by GT vertex-count bucket |
|---|---:|---:|---|
| J5_token_fine2 / raw / 1 | 0.8800 | 0.999872 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 30, "trees": 44}, "300-700": {"exact": 22, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |
| J5_token_fine2 / raw / 20 | 0.9100 | 0.999882 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 32, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J5_token_fine2 / raw / 50 | 0.8200 | 0.999806 | `{"0-300": {"exact": 19, "trees": 20}, "1200+": {"exact": 24, "trees": 44}, "300-700": {"exact": 20, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |
| J5_token_fine2 / ema / 1 | 0.9000 | 0.999896 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 32, "trees": 44}, "300-700": {"exact": 22, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J5_token_fine2 / ema / 20 | 0.9400 | 0.999924 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 36, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J5_token_fine2 / ema / 50 | 0.8400 | 0.999801 | `{"0-300": {"exact": 19, "trees": 20}, "1200+": {"exact": 24, "trees": 44}, "300-700": {"exact": 20, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |

Distance metrics are vertex-set distances between D9 cell centres in [-1,1] units; they are not surface metrics or raw floating-point GT errors.
Hausdorff uses the maximum directional nearest-neighbour distance. Chamfer is half the sum of the two directional mean distances.
Missing-depth9 trees are excluded by the supplied distance evaluator. Empty predictions have infinite distance. Chamfer mean includes finite trees only; per-case denominators are recorded in RESULT.json and DISTANCE_DENOMINATORS.json.
Full-tree and integer-set success always retain all 100 scheduled trees as the denominator.

The 20-step results are saved final training evaluations; 1/50-step cases are the completed frozen-checkpoint supplement.
Final training evaluations identify run/update, not a freshly replayed checkpoint hash. Checkpoint byte hashes and recovery-state audits are separate evidence; this builder never loads model or optimizer tensors.

J3@7075 eight-case decoding and 14 boundary analyses were already published at [commit d482f4c4](https://github.com/leoguohr/suxen/commit/d482f4c414fde9fe42078952e124785d3b2ab471); they are not rerun here.

Excluded: checkpoint tensors, prediction NPZ arrays, credentials. Those remain on the server. No causal attribution is made from these comparisons.

[Compressed evidence](../evidence/round2-closeout-J5_token_fine2-u009000-cab7249edee65ed0/evidence.tgz) · [Archive SHA256](../evidence/round2-closeout-J5_token_fine2-u009000-cab7249edee65ed0/PACKAGE.json) · [Structured results](../evidence/round2-closeout-J5_token_fine2-u009000-cab7249edee65ed0/RESULT.json)
