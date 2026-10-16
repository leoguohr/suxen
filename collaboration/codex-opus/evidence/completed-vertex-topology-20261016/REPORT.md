# Completed Vertex K1/K2 and Topology A/B — 2026-10-16

This handoff supplies the completed results missing from the exchange branch. It contains existing training/evaluation evidence, not a new training run or model replay.

## Vertex: Round3 Batch2 second training cycle

Fixed 50 NEXUS2K objects, official seeds 97029000/97029001, depth 9, 20 Euler steps per layer; exact means every generated layer is correct.

| Arm | Updates | LR / decay | Raw complete trees | EMA complete trees |
|---|---:|---|---:|---:|
| K1, full-state J5 initialization | 4500/4500 | 1e-5 / 0.4 | 90/100 | 90/100 |
| K2, full-state J5 initialization | 4500/4500 | 5e-6 / 0.5 | 92/100 | 91/100 |

Both jobs printed JOB_COMPLETE and have final.pt plus complete per-tree final evaluations. K1 completed at 2026-10-16 03:51 UTC; K2 at 05:03 UTC. Both used token weighting and fine-copies=2. Neither satisfies 100/100 exact. The requested ten-seed EMA follow-up at 2 and 20 steps is still absent. These are not the older, unstarted K1 decay0.2 / K3 / K4 preparations.

For 001093, K1 passes both official seeds with both weights. K2 passes seed 97029000; seed 97029001 first fails at depth 7 for both weights. This is not a ten-seed result or the KT/KU/KH targeted fine-tune experiment.

## Topology T1: completed two-GPU pair

Each independent branch resumed its old update-1386 parent, added 2614 updates, and stopped at cumulative update 4000 under the existing raw Face micro-F1 >= 0.99 early-stop rule. Both training and all scheduled post-evaluations exited successfully.

| Arm | Raw Face micro-F1, seeds 0 / 1 | Raw joint strict, 50 objects x 2 seeds | EMA Face micro-F1, seeds 0 / 1 |
|---|---|---:|---|
| A, fresh initialization branch | 0.9942495 / 0.9942024 | 15/100 | 0.9103484 / 0.9114380 |
| B, S0 warm-start branch | 0.9950447 / 0.9950491 | 20/100 | 0.9396169 / 0.9381436 |

These are raw Euler-20 results; EMA joint strict is 0/100 for both. Passing the F1 early-stop rule does not imply strict complete reconstruction or an EMA pass. The archive includes every raw/EMA milestone, the final EMA Euler/DPM2M 10/20/50-step sweeps, raw Euler-20, and EMA FP32 Euler-20. Each evaluation case has all 50 objects, with no candidate overflow or lower-bound-only result.

## Separate four-GPU versions: progress snapshot only

At 2026-10-16 09:09:55 UTC, quad_resume A had reached update 2668 and was continuing to append training rows. Quad B had migration/dry-run evidence but no actual training log or final pointer. These separate trajectories must not be merged with the completed two-GPU A/B results. The captured logs have explicit complete-line/byte boundaries; this is not a final-result report.

## Reading order and evidence

1. Read the Vertex report and Topology report linked below for source paths, protocol, code identity and limitations.
2. STATUS.json provides machine-readable completion, metrics and snapshot boundaries.
3. The compressed archive contains full train/probe logs, actual configs and commands, every per-tree/per-object evaluation, source/wrapper identity and small source snapshots. FILE_MANIFEST.json gives each archived file's size and SHA256. Original source-path manifests are inside each section.

Checkpoint tensors and predictions remain on the server. Final checkpoint existence and byte sizes were checked; large checkpoint hashes were not recomputed in this collection. Recorded Topology hashes are identified as historical committed metadata. No GPU, new evaluation, restart, training change or earlier-folder modification was performed.

[Vertex detailed report](https://github.com/leoguohr/suxen/blob/codex-opus-exchange/collaboration/codex-opus/evidence/completed-vertex-topology-20261016/VERTEX_REPORT.md) · [Topology detailed report](https://github.com/leoguohr/suxen/blob/codex-opus-exchange/collaboration/codex-opus/evidence/completed-vertex-topology-20261016/TOPOLOGY_REPORT.md)
