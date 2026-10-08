# Fixed50 Topology Flow candidate campaign — 2026-10-07

## Authorization and immutable data

User confirmed original fixed50 selection and 001825=1277 vertices. Frozen OwnAE-v2 step36220, NativeTopologyAE/B_v2_teacher_blocks, latent512. GT vertices and original Stage2 8192 XYZ/normal condition. Existing cache and complete VAE reconstruction baseline reused without re-export or changing normalization.

New campaign: at most 4 GPU hours including calibration, training, checkpoint saving and generation evaluation; at most 8000 effective five-mesh optimizer updates per group (user revised 1000 to 8000). C1 first, C2 second, approximately equal remaining wall-time shares. Time limit takes precedence. Fresh branches do not resume C0 weights or reset an existing training cursor.

## Structural variants

- C1: original 36-layer width1536, 12-head Flow; add teacher-layout raw XYZ plus six Fourier bands (39 dimensions) through zero-initialized projection. Original shared seed0 tensors and RNG remain identical.
- C2: teacher-style selected combination: same topology latent and point condition interfaces; Fourier projection, no Flow QK norm, Flow norm epsilon 1e-5, affine final norm and exact GELU. No teacher weights imported.
- Both keep independent point condition encoder, RoPE scale256, FP32/MATH, microbatch1, accumulation5, fresh posterior sampling and independent posterior/noise/time streams, original AdamW/warmup/clip.
- Execution-only activation recomputation policy selected by bounded GPU calibration; raw accumulated gradients and predictions checked on identical inputs. Probe speed after a one-mesh warmup is only a selection heuristic. Real first3 updates profile full forward/backward/clip/Adam and count normally.

## Persistence and retention

Remote destination: `/guohaoran/nexus_fast_track/diagnostics/topology_flow_candidates_20261007`, persistent `storage` GPFS. `/ssdwork/guohaoran` returned errno122 Disk quota exceeded on code upload; no other experiment was deleted. Personal quota unknown.

Each populated full checkpoint is approximately 28.008 GB. Keep protected500, protected1000 and final/latest; within a group peak4 including the incoming atomic temporary. Two groups peak7, approximately196.056 GB, excluding unexpected extra best protections. Update1 writes real populated model/AdamW/RNG/cursor through flush/fsync/SHA256/atomic rename and post-rename SHA check. Previous copy remains until commit succeeds. One successful write gives no guarantee about later quota.

## Outputs

`campaign.json` binds original absolute clock, UUID, cache/VAE/selection/normalization and code/config hashes. `perf/` records every calibration attempt. `runner-receipt.json` and `runner-status.json` record commands and bounded phase state. Per-group `run/` contains full checkpoints, SHA manifest and actual training MSE/profiles; `eval*/` contains separate noise-to-data Euler50 Edge/actualFace generation, per-UID metrics and visualization, with incomplete runs clearly labeled.

Prior C0 completed1000 is unchanged at `/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006`. Its results are a reference, not a new candidate success.

The user requested launch-only interaction: after startup verification no monitoring automation or continued polling is scheduled.
