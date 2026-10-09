# Codex → Opus: Round 2 J3/J4 performance tests

The benchmarks run independently on two idle A100 allocations, one recipe per host. Formal J1/J2 were submitted separately by the user.

Approved trainer commit: `c00ebcc984ed354609c7a9e539252402e5329ef3`. Full A init SHA256: `1bd4ced48296b45c80d54386171652a6f9eb9c09e8fc976965caad62899b55f9`.

| Recipe | Suite state | Case | Median u2–5 seconds | Peak allocated GiB | Peak reserved GiB | Exit |
|---|---|---|---:|---:|---:|---:|
| J3 | completed | J3_baseline | 7.975142465904355 | 54.974082946777344 | 57.072265625 | 0 |
| J3 | completed | J3_chunk12288 | 8.0267318286933 | 56.08516502380371 | 57.80859375 | 0 |
| J3 | completed | J3_no_recompute4096 | 6.60657284874469 | 76.60822820663452 | 78.625 | 0 |

J3: fastest completed measurement is `J3_no_recompute4096`.
| J4 | failed_stopped | J4_baseline | 10.064905076287687 | 54.982951164245605 | 56.990234375 | 0 |
| J4 | failed_stopped | J4_chunk12288 | 10.055694594047964 | 56.08989858627319 | 57.939453125 | 0 |
| J4 | failed_stopped | J4_no_recompute4096 | unavailable | unavailable | unavailable | 1 |

J4: fastest completed measurement is `J4_chunk12288`.
Suite error: `AssertionError('J4_no_recompute4096 failed; suite stopped; preserve log and partial timings')`. Preserve partial logs; this is not an all-cases pass.

## Formal recipes and current evidence

- Keep the approved 8192-token / activation-checkpointing configuration for formal J3/J4. Larger chunks showed no material speed gain. Disabling recomputation reached about 76.6 GiB allocated on J3 and OOM on J4; that candidate is not recommended for formal launch.
- J1/J2 retain their original COMMON and ordering; only loss weighting differs (item/token).
- J3 retains COMMON + token weighting, then appends only `--lr 5e-6`.
- J4 retains COMMON + token weighting, then appends only `--coarse-copies 8`.
- Formal chunk-tokens remains 8192, checkpoint-activations 1 and row-tokens 4096. Benchmark candidates do not change these commands.
- All five launcher files, available check-only logs and optional prior-script evidence are included. No live remote-script diff is implied.

- J1: launcher prepared `True`; formal output records observed `True`; last recorded update `117`; recorded state `training`.
- J2: launcher prepared `True`; formal output records observed `True`; last recorded update `83`; recorded state `training`.
- J3: launcher prepared `True`; formal output records observed `False`; last recorded update `unavailable`; recorded state `unavailable`.
- J4: launcher prepared `True`; formal output records observed `False`; last recorded update `unavailable`; recorded state `unavailable`.

## Measurement boundaries

- u2-5 timings are within the prescribed 20-update LR warmup, not steady-state estimates.
- Each case restores full A model/AdamW/EMA and discards all six updates; no model checkpoint is saved.
- UID/noise/time hashes reconstruct deterministic inputs after timing; they are not live captured tensors.
- Largest-eight stress update6 and six updates total do not bound all 9000-update peaks.
- Short benchmark timings do not guarantee the full 9000-update runtime or completion ETA.
- Allocated/reserved CUDA peaks are reported; no explicit VRAM headroom threshold is implemented by this harness.
- Failure/timeout/OOM stops that recipe suite. Completed measurements can still be reported from a failed suite.
- Candidate settings are for reporting only; no candidate is automatically promoted to a formal command.
- J1/J2 are user-submitted jobs; collected files are partial snapshots, not completed scientific results.
- Collection timestamps are server wall time; timing durations use monotonic time and CUDA synchronization.

No weights, prediction arrays, credentials or local result downloads are included. Original A and formal results remain on persistent `/guohaoran`; `/ssd/guohaoran` is a recreated ephemeral init cache.

[Compressed evidence and checksum](../evidence/20261010-round2-j3-j4-benchmark/README.md)
