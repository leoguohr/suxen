# Codex → Opus: Round 2 preparation gates

The user explicitly limited this session to steps 1–4. **J1/J2 have not been submitted.**

- Effective code: `c00ebcc984ed354609c7a9e539252402e5329ef3`; clean working tree: `True`.
- CPU gates: `cpu_gates_passed`.
- GPU smoke: `completed`.
- This instance exposes one A100-SXM4-80GB. No second server or long-job submission interface was supplied.
- Exact commands, environment, timing and exit codes are in the compressed receipts and logs.

| Round1 evaluation | Exact trees | Wrong bits at first mismatch | Within 0.05 of threshold |
|---|---:|---:|---:|
| A/eval_raw | 35/100 | 278 | 0 |
| D/eval_raw | 26/100 | 356 | 0 |
| D/eval_ema | 28/100 | 205 | 1 |
| B/eval_raw | 18/100 | 517 | 0 |

These are CPU reanalyses of saved Round1 predictions, not new model replays. Wrong values are mostly far from the 0.5 threshold; the records do not support treating most failures as tiny threshold near-misses. They do not establish a cause or calibrated confidence. The supplied 0–1 histogram excludes values outside that interval; its bins need not sum to all wrong bits.

## Checkpoint and storage

- Source: `/guohaoran/tmp/nexus_vertex_dense_overfit50_20261009/A/final.pt`.
- Required SHA256: `1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9`.
- Source A contains model, EMA and AdamW. Historical checkpoint metadata records 907 model tensors, 907 EMA tensors and 907 Adam state entries; no RNG or scheduler keys.
- `/ssd/guohaoran` on this host is an ephemeral overlay cache, not a persistent SSD mount. The source checkpoint and smoke outputs remain under persistent `/guohaoran`.
- `init_copy.json` records the copy hash and independent destination readback; `init_metadata.json`, when present, records the current CPU-only checkpoint inspection.

## Smoke boundaries

- State: `completed`; error: `none`.
- The smoke command explicitly uses `--probe-minutes 0` as prescribed in step 4. `NEXUS_SKIP_COARSE_PROBES` is unset. Formal J1/J2 retain `--probe-minutes 30`.
- Attempt 1: exit `143`, SIGTERM sent `True`.
- Attempt 2: exit `0`, SIGTERM sent `False`.
- Attempt 3: exit `0`, SIGTERM sent `False`.
- Interrupted at update 4; identical command resumed to update 4; third invocation was ALREADY_COMPLETE.
- Milestone u2/raw and final u4/raw each contain 100 trees. Both use the prescribed smoke-only two Euler steps.
- Full final checkpoint remains on the server; it is not a new experimental starting point.

## Setup events and next action

- Initial bundle unpacking was attempted before transfer finished (early EOF). After transfer, shallow Git history caused a second clone error. No checks or GPU jobs had run. Restoring the exact source shallow boundary allowed fetch and git fsck to pass; model/training code was unchanged.
- Any failed gate stops this workflow. No failed science gate is repaired or bypassed.
- Await the user’s formal-job launch instruction and long-running allocation details; keep the prescribed 9000-update J1/J2 commands and coarse probes enabled.

[Compressed evidence and checksum](../evidence/20261009-round2-gates/README.md)
