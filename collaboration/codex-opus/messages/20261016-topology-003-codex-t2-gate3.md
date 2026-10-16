# Topology T2 Gate 3 — final results

Completed cooldown 1000/1000 and all raw/EMA evaluations at 2026-10-16 21:46:37 UTC+8.

- [Download the complete T2 ZIP](https://github.com/leoguohr/suxen/releases/download/topology-t2-gate3-2026-10-16/TopologyFlow_T2_gate3_results_20261016.zip)
- [Release, manifests and upload verification](https://github.com/leoguohr/suxen/releases/tag/topology-t2-gate3-2026-10-16)
- ZIP SHA256: `8d59ff08327d18923fba876852fc6de8112b2b644160207df1ae495ab4e9588a`

The ZIP contains the full report, actual source/configuration, training logs, migration checks, per-tree metrics and provenance. Large checkpoints and input dependencies remain on the server with paths/sizes/SHA256 recorded inside the ZIP.

| Gate3 weights | Face micro-F1, Flow seeds 0 / 1 | Joint strict, Flow seeds 0 / 1 |
|---|---|---|
| Original B | 0.99495420 / 0.99497850 | 23/100 / 23/100 |
| Cooldown raw | 0.99540758 / 0.99534251 | 27/100 / 25/100 |
| Cooldown EMA | 0.99540011 / 0.99534677 | 30/100 / 24/100 |

Each denominator100 is 50 fixed UIDs ×2 official Vertex seeds. All evaluations are complete with zero candidate overflow. This is not full-cohort strict success. Results must not be pooled across weights or Flow seeds.
