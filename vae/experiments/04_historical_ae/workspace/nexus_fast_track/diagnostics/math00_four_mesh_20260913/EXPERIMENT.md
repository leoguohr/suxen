# Four complete meshes: 1000-update observation from Hold

Starting point: `../math00_kl_hold_20260912/checkpoint-update0200.pt`; all five Adam groups and training RNG restored. Beta=1e-4 throughout. LR E/mu1e-8, decoder/heads1e-7, logvar1e-4; wd0, global clip1, math00, deterministic Graph, clamp[-20,10], fully-differentiable Edge+Face Soft4 unchanged. No posterior reinitialization, mu loss, subgraph sampling, optimizer reset, or LR adjustment.

Selected before inspecting Hold reconstruction:

| Role | UID | Vertices | Faces |
|---|---|---:|---:|
| Old Small | nexus_2k_000387 |386|768|
| Old Large | nexus_2k_001849 |2575|5146|
| New A | nexus_2k_001716 |534|1024|
| New B | nexus_2k_001333 |804|1604|

Selection is based on manageable full-graph size within the original20; no selection by successful reconstruction. Distinct vertex hashes verified. Old candidate files are copied byte-for-byte. New candidate pools use the exact archived prepare rule (historical checkpoint1000 mu/sample0, stable UID seed, GT false-cycle retention, hard-negative replacement and heldout split). Pool/source hashes and generation provenance are in selection.json and new_candidates/preparation.json. That historical mining checkpoint is only a data-preparation donor, never the training starting point. The mining stage's logged initial evaluations are not Hold step0.

Every update has one packed forward containing all four complete meshes. Edge loss covers every unordered vertex pair; Face training uses each mesh's fixed pool. Loss is the equal mean of each mesh's Edge+Face plus beta times the equal mean of each mesh's internal KL. Runtime checks verify both mesh averages each step. No cross-mesh pooling of vertex/candidate weights.

Memory preflight made one full packed sampled forward/backward, no optimizer update, then restored RNG and cleared gradients; peak allocated15.04GiB/reserved16.56GiB on79.14GiB device. Full four-mesh execution fits.

Budget: user-confirmed1000 actual updates, not a capacity-failure deadline. Mu/original fixed-noise diagnostics at0,1,10,20,50,100,200,400,600,800,1000. The monitoring50 noise conditions at0,200,400,600,800,1000; final another50 unseen conditions. Each noise condition covers all four meshes in the same packed forward. Existing two monitoring streams retained, two disjoint streams added. Final seeds10080000..10080199. Training uses a separate continuing RNG, fresh noise per mesh per update, reused within that update's backward/recompute.

Actual Face candidates are re-enumerated from each forward's predicted Edge graph; training pools never replace reconstruction candidates. Logs separate old pair, new pair, and all-four strict success. New mesh step0 errors are expected learning-task baseline, not revocation of the passed two-mesh VAE. Finite1000-step endpoint not being perfect does not prove capacity/VAE failure.

Checkpoints every200 updates include all five Adam groups, RNG and dataset/pool manifest. Per-step logs include four loss pairs, KL decomposition, sigma/raw/effective logvar/clamp, perturbation, actual group updates and preclip gradient/clip. Actual counts and signed margins are in full evaluation JSON/JSONL. No candidate VJP, line search, or architecture ablation.

Large checkpoints, actual epsilon tensors and NPZ stay on server. Code, provenance and full JSON metrics are copied locally. `verify.py` checks exact restoration, all4000 actual noise tensors, code/pool hashes and equal-mesh weighting without training. `analyze.py` produces per-mesh comparisons.
