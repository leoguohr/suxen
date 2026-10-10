# Codex → Opus: Round2 five missing milestones and six-job status

5 new raw evaluations passed the fixed50 × two-seed completeness contract and all saved prediction SHA256 checks. Earlier reports remain unchanged. This backfills J1/J2/J3/J6 u5400 and J5 u3600; J4 has no newly completed milestone.

| Job | Saved train-tail update | Latest completed receipt | Newly validated | <300 | 300–700 | 700–1200 | 1200+ |
|---|---:|---|---|---:|---:|---:|---:|
| J1 | 6532 | u5400/raw: 28/100 | u5400/raw: 28/100 | 18/20 | 9/22 | 1/14 | 0/44 |
| J2 | 6452 | u5400/raw: 44/100 | u5400/raw: 44/100 | 20/20 | 17/22 | 6/14 | 1/44 |
| J3 | 6322 | u5400/raw: 48/100 | u5400/raw: 48/100 | 18/20 | 17/22 | 9/14 | 4/44 |
| J4 | 5160 | u3600/raw: 33/100 | no new target | not collected | not collected | not collected | not collected |
| J5 | 3773 | u3600/raw: 37/100 | u3600/raw: 37/100 | 18/20 | 12/22 | 6/14 | 1/44 |
| J6 | 5972 | u5400/raw: 36/100 | u5400/raw: 36/100 | 18/20 | 11/22 | 7/14 | 0/44 |

Latest new full-tree exact results: J1 28/100, J2 44/100, J3 48/100, J5 37/100, J6 36/100. J1/J2/J3/J6 are at u5400; J5 is at u3600. J3 final-set exact is 49/100 but its all-depth full-tree exact is 48/100. All runs are interim; none has completed the final 9000-update gate.

- All canonical receipts, JSON cohort counts, UID/seed sets, per-depth aggregates, first-error histograms, size buckets and prediction hashes agree.
- Six-job configs, saved statuses, all complete train/probe JSONL records plus console tails, evaluation receipts, code/data hashes are included. Saved files do not prove each formal pod live-process state; ETA is an estimate.
- J6 CPU cascade analysis ran on the new u5400 cohort after validating all100 prediction hashes; command, elapsed time, exit status and raw analysis are included.
- Snapshot collection is read-only for running jobs. Complete JSONL lines are captured up to collection time; active logs can advance afterward.
- J5 changed from 49/100 at u1800 to 37/100 at u3600. J6 changed from 27/100 at u3600 to 36/100 at u5400. No causal attribution or final acceptance is claimed.
- Round2C Part B 1/50-step evaluation is not included or claimed here; these are original 20-step raw milestones.
- No checkpoint reads, model/GPU work, training changes or Mac result artifacts. NPZ predictions remain on the server.

[Compressed evidence](../evidence/20261010-round2-milestones-2100/README.md)
