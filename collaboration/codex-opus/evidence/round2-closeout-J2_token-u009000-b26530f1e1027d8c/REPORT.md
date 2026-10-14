# NEXUS Round2 closeout

Scope: `decode`; update 9000; analysis code `e7c85bf2b82fdc6ee15eb59be20486ffc7853fa4`.

Saved predictions were validated and analyzed on CPU. This package runs no model forward or training.

| Job | Weights | Euler steps/layer | Teacher rule vs D9 GT | Full tree exact | Final integer set exact | Within one cell |
|---|---|---:|---:|---:|---:|---:|
| J2_token | raw | 1 | 76/100 | 75/100 | 76/100 | 76/100 |
| J2_token | raw | 20 | 85/100 | 85/100 | 85/100 | 85/100 |
| J2_token | raw | 50 | 72/100 | 72/100 | 72/100 | 72/100 |
| J2_token | ema | 1 | 74/100 | 74/100 | 74/100 | 75/100 |
| J2_token | ema | 20 | 85/100 | 85/100 | 85/100 | 85/100 |
| J2_token | ema | 50 | 75/100 | 75/100 | 75/100 | 75/100 |

| Job / weights / steps | depth 1–9 exact counts (each /100) | First mismatch depth |
|---|---|---|
| J2_token / raw / 1 | 100, 100, 100, 100, 100, 96, 87, 82, 76 | `{"6": 4, "7": 9, "8": 5, "9": 7, "None": 75}` |
| J2_token / raw / 20 | 100, 100, 100, 100, 98, 94, 90, 85, 85 | `{"5": 2, "6": 4, "7": 4, "8": 5, "None": 85}` |
| J2_token / raw / 50 | 100, 100, 100, 100, 98, 90, 83, 74, 72 | `{"5": 2, "6": 8, "7": 7, "8": 9, "9": 2, "None": 72}` |
| J2_token / ema / 1 | 100, 100, 100, 100, 99, 93, 84, 79, 74 | `{"5": 1, "6": 6, "7": 9, "8": 5, "9": 5, "None": 74}` |
| J2_token / ema / 20 | 100, 100, 100, 100, 98, 94, 90, 85, 85 | `{"5": 2, "6": 4, "7": 4, "8": 5, "None": 85}` |
| J2_token / ema / 50 | 100, 100, 100, 100, 98, 90, 84, 76, 75 | `{"5": 2, "6": 8, "7": 6, "8": 8, "9": 1, "None": 75}` |

| Job / weights / steps | Vertex count accuracy | Depth9 micro F1 | Full-tree exact by GT vertex-count bucket |
|---|---:|---:|---|
| J2_token / raw / 1 | 0.8000 | 0.999778 | `{"0-300": {"exact": 19, "trees": 20}, "1200+": {"exact": 25, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 10, "trees": 14}}` |
| J2_token / raw / 20 | 0.8900 | 0.999834 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 30, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J2_token / raw / 50 | 0.7900 | 0.999673 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 18, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |
| J2_token / ema / 1 | 0.8000 | 0.999768 | `{"0-300": {"exact": 19, "trees": 20}, "1200+": {"exact": 25, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 9, "trees": 14}}` |
| J2_token / ema / 20 | 0.9000 | 0.999834 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 30, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 14, "trees": 14}}` |
| J2_token / ema / 50 | 0.7900 | 0.999702 | `{"0-300": {"exact": 20, "trees": 20}, "1200+": {"exact": 21, "trees": 44}, "300-700": {"exact": 21, "trees": 22}, "700-1200": {"exact": 13, "trees": 14}}` |

Distance metrics are vertex-set distances between D9 cell centres in [-1,1] units; they are not surface metrics or raw floating-point GT errors.
Hausdorff uses the maximum directional nearest-neighbour distance. Chamfer is half the sum of the two directional mean distances.
Missing-depth9 trees are excluded by the supplied distance evaluator. Empty predictions have infinite distance. Chamfer mean includes finite trees only; per-case denominators are recorded in RESULT.json and DISTANCE_DENOMINATORS.json.
Full-tree and integer-set success always retain all 100 scheduled trees as the denominator.

The 20-step results are saved final training evaluations; 1/50-step cases are the completed frozen-checkpoint supplement.
Final training evaluations identify run/update, not a freshly replayed checkpoint hash. Checkpoint byte hashes and recovery-state audits are separate evidence; this builder never loads model or optimizer tensors.

J3@7075 eight-case decoding and 14 boundary analyses were already published at [commit d482f4c4](https://github.com/leoguohr/suxen/commit/d482f4c414fde9fe42078952e124785d3b2ab471); they are not rerun here.

Excluded: checkpoint tensors, prediction NPZ arrays, credentials. Those remain on the server. No causal attribution is made from these comparisons.
