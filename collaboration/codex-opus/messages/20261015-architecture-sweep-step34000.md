# Architecture sweep S0–S6: saved endpoint and execution evidence

CPU-only evidence collection on 2026-10-15. No training, model forward, checkpoint tensor load, or new evaluation was performed.

| Branch | Formal training | Last logged cumulative step | Latest saved checkpoint step | Endpoint full-tree exact |
|---|---|---:|---:|---:|
| S0 | Completed 10,000 sweep updates | 34,000 | 34,000 | 7/100 |
| S1 | Completed 10,000 sweep updates | 34,000 | 34,000 | 7/100 |
| S2 | Started; incomplete (6,098 updates logged) | 30,098 | 30,000 | Not produced |
| S3 | Started; incomplete (5,688 updates logged) | 29,688 | 29,600 | Not produced |
| S4 | Never dispatched in the saved sweep | — | — | Not produced |
| S5 | Never dispatched as a formal branch; one discarded smoke update exists | — | — | Not produced |
| S6 | Never dispatched in the saved sweep | — | — | Not produced |

S0 and S1 endpoint evaluations use the original 50 seen objects × seeds 97029000/97029001, D9 root-to-leaf generation, 20 Euler steps per layer, threshold 0.5, and model-generated parents. Both contain all 100 per-tree rows. UID/seed coverage, full-tree and final-set counts, and all nine per-depth TP/FP/FN/exact aggregates were recomputed from the saved JSON rows and agree. This is a file audit, not a fresh model replay.

## Files

- `S0/evaluation/summary.json`, `S0/evaluation/evaluation.json`, and the corresponding `S1` files are directly readable JSON, preserved byte-for-byte.
- `S0/train.jsonl.gz` through `S3/train.jsonl.gz` contain each branch's complete available canonical training log, with no tail-only truncation. All update sequences are contiguous from sweep update 1; cumulative step = 24000 + sweep update. Gzip decompression is verified against the original SHA256.
- Each branch includes config, saved status, checkpoint identity, and available endpoint provenance. S2/S3 have no evaluation JSON anywhere in their saved branch folders; none was fabricated.
- No independent `probes.jsonl` or other training-probe files were found. This sweep's source and execution card schedule only the endpoint evaluation. Training losses, gradient norms, timing, memory, and per-microbatch noise/time records are in `train.jsonl.gz`.
- Queue status, dispatch commands, console logs, source code, the old endpoint import failures, and S5's explicitly discarded smoke status are included.
- `FILE_MANIFEST.json` records original server paths, original and compressed byte hashes, sizes and timestamps; `UPLOAD_MANIFEST.json` records public file hashes.

## Why S4–S6 did not run

1. The earlier recovery completed S0/S1 training but failed at endpoint evaluation import with `ModuleNotFoundError: No module named 'diffusers'`. Its fail-stop policy initially blocked later branches.
2. The `20261007b` continuation corrected the dependency path and completed both endpoint evaluations. The dispatch log then launched S2 at 2026-10-07 23:48:09 +08:00 and S3 at 23:48:24 +08:00.
3. The latest saved queue (2026-10-08 02:35:05 +08:00) still has S2/S3 occupying the two slots and S4/S5/S6 pending. Both training logs stop around 02:35:09. No later dispatch entries, branch directories, or final results exist in the inspected original/recovery roots.
4. Therefore the saved queue never reached S4–S6. The external reason the two active branches stopped is **not recorded**: there is no final queue-failure, process-exit or instance-expiry record. The old `running` status is stale and does not prove present liveness. Instance expiry must not be stated as a confirmed cause.

S5's one-update `smoke_S5_group2` has `smoke_only=true` and `formal_updates_counted=0`; it is not a completed or started formal 10,000-update branch.

## Evidence limits

- S2/S3 logs extend 98/88 updates beyond their latest saved recovery checkpoints. These unsaved updates are not recoverable from those checkpoints.
- Source JSON identifies the checkpoint hashes; this collection did not reread large `.pt` files. Weights, optimizer tensors, prediction arrays and credentials are excluded.
- No results for S4–S6 and no endpoint results for S2/S3 are implied. Search scope is recorded in `STATUS.json`.

[Evidence files](../evidence/architecture-sweep-step34000-20261015/) · [Upload SHA256 manifest](../evidence/architecture-sweep-step34000-20261015/UPLOAD_MANIFEST.json)
