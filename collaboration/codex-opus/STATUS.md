# Current confirmed status

- Training: A/B/C 749 updates; D 608 updates; all complete.
- Raw complete-tree exact: A 35/100, B 18/100, C 3/100, D 26/100. D EMA 28/100; A/B/C EMA not run (time budget).
- Acceptance 100/100: none passed. Baseline S0 reference 7/100.
- Four final checkpoints and predictions are preserved under `/guohaoran/tmp/nexus_vertex_dense_overfit50_20261009/`; current persisted hashes match original-copy records.
- Full CPU gate passed with a CPU math SDP wrapper. GPU preflight completed 2/3 updates; coarse training probes were skipped by a runner decision, not an explicit user instruction.
- No new training or GPU evaluation during collection. The archive contains no .pt or predictions.

[Latest results](messages/20261009-003-codex-dense-results.md) · [Compressed evidence](evidence/20261009-dense-runs/README.md)
