# Topology T1: completed two-GPU A/B and separate four-GPU snapshot

Captured: 2026-10-16T09:09:55.055324+00:00 (UTC). Source research commit: 72c993772a91482a3660faa7c83993bfd566cbfe.

## Completed experiments

pair_resume_20261016/A_fresh and B_s0warm each resumed from cumulative update 1386, performed 2614 additional updates, and completed at cumulative update **4000**. Each has JOB_COMPLETE, status=complete, a committed final checkpoint pointer, an existing checkpoint with matching recorded byte size, and a launcher that exited successfully after all post-training evaluation stages.

The configured upper limit was 10000. The existing stop_at_pass rule stopped both at 4000 because **raw weights** passed the predefined aggregate Face F1 >= 0.99 criterion for both seeds. This is not a 100/100 strict reconstruction claim, and EMA did not pass.

| Fixed 50 objects; Euler 20; seeds 0 / 1 | A fresh | B S0 warm |
|---|---:|---:|
| Raw Face micro F1 | 0.9942495103 / 0.9942024360 | 0.9950446658 / 0.9950491456 |
| Raw Edge micro F1 | 0.9989200411 / 0.9988943437 | 0.9993002370 / 0.9992679993 |
| Raw joint strict | 8/50 + 7/50 = 15/100 | 11/50 + 9/50 = 20/100 |
| EMA Face micro F1 | 0.9103483791 / 0.9114379942 | 0.9396169054 / 0.9381436110 |
| EMA joint strict | 0/100 | 0/100 |

A and B are separate initialization branches, not sequential training stages. The completed versions here each used two GPUs per update. Their original single-GPU update-1386 parent folders remain separate.

## Completed evaluation coverage

For **each** completed branch:
- Milestones u2000 and u4000: raw and EMA, both seeds, 50 objects per seed.
- Final EMA: Euler and DPM2M, each at 10/20/50 steps, both seeds = 12 cases.
- Final raw: Euler 20, both seeds = 2 cases.
- Final EMA FP32: Euler 20, both seeds = 2 cases.
- Every final case covers 50 objects. No candidate overflow and no lower-bound-only summaries.
- Full per-object objects.jsonl / seed*_objects.jsonl and all summaries are included; this evaluator does not use an evaluation.json filename.
- Training and all three post-training evaluator processes exited 0.

The complete train/probe logs, raw/EMA milestone summaries, exact launch commands, execution wrappers, recipe/config/source hashes, final pointer records and per-object evaluations are included. Full checkpoints and prediction arrays remain on the server.

## Separate four-GPU versions: snapshot, not completion

quad_resume_20261016 is an independently launched version from the old 1386 parents. Do not merge its updates with the completed two-GPU trajectories.
- A: last complete captured training row is update **2668**; logs progressed during the read-only inspection. Its u2000 raw Face F1 is 0.9871294400 / 0.9861867779; joint strict 4/50 + 4/50. No final result is claimed.
- B: at capture, only migration and dry-run preparation evidence exists; no train log, final pointer or actual START is observed. We cannot infer the scheduler queue state.
- Quad logs are bounded snapshots. FILE_MANIFEST.json records source size, complete-line truncation boundary, capture time, source mtime, full bounded-source SHA, and copied-tail SHA. No claim of a live PID on this separate maintenance container is made.

## Provenance and limits

Evidence source root: /guohaoran/nexus_topology_flow_overfit50_20261015.
The scientific source snapshot corresponds to the commit above. Additional distributed execution and migration wrappers are archived separately; their hashes identify the effective execution code.

Recorded checkpoint SHAs are copied from the completed atomic-commit metadata. This handoff checks file existence and size only; it does not reload or rehash 37 GB checkpoint files.

The former root-level four-GPU attempt exited without new valid updates because legacy latest.pt lacked the required verified latest.json pointer. The repair and successful pair_resume_20261016 runs are distinct from that failed attempt.

No GPU work, new training, restarts, deletions, source edits, or result recomputation was performed to collect this handoff.
