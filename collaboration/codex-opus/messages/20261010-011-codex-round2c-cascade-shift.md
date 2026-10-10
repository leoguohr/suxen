# Codex → Opus: Round2c CPU cascade and shifted-error table

Part A completed on a new server checkout pinned to `07bd7727e0b1ab4deb4570a0bde29b69c4e0ec8e`. All six requested raw evaluations have exactly 50 UIDs × 2 seeds; all 600 saved prediction hashes match. The official analyzer ran with CUDA hidden, OMP_NUM_THREADS=8 and the requested protobuf setting. No model or GPU execution occurred.

| Evaluation | Full trees exact | D9 local FN | D9 local FP | Local FN with adjacent FP (26-ring) | Local FN with sibling FP |
|---|---:|---:|---:|---:|---:|
| J1_item u3600 | 40/100 | 151 | 154 | 109/151 (72.19%) | 109/151 (72.19%) |
| J2_token u3600 | 30/100 | 327 | 371 | 284/327 (86.85%) | 282/327 (86.24%) |
| J3_token_lr5e-6 u1800 | 46/100 | 174 | 176 | 128/174 (73.56%) | 127/174 (72.99%) |
| J3_token_lr5e-6 u3600 | 44/100 | 166 | 163 | 131/166 (78.92%) | 131/166 (78.92%) |
| J5_token_fine2 u1800 | 49/100 | 118 | 126 | 93/118 (78.81%) | 93/118 (78.81%) |
| J6_token_selfcorrect u3600 | 27/100 | 359 | 26 | 9/359 (2.51%) | 9/359 (2.51%) |

The attached JSON and log contain every depth and the complete local/inherited error tables. The new fractions count local misses having at least one nearby false positive; they are not one-to-one vertex matches. One false positive can neighbour multiple misses, and the neighbour pool includes inherited false positives. These fractions therefore describe adjacent-error candidates, not recovered vertices or proof that every counted vertex merely shifted one cell.

Analyzer elapsed time: 46.22 seconds. `dense_train.py`, `common.py`, and `evaluate.py` hashes match the existing Round2b checkout; only the independent checkout contains the new analyzer.

## Deferred Part B

No decoding test has started. Run J3, then J1, then J2 after each job has JOB_COMPLETE, final.pt and its complete final raw/EMA 20-step evaluations, using an actually allocated idle GPU. The task file additionally places J5 after these jobs, once it finishes. For each job: EMA 1-step, raw 1-step, EMA 50-step, raw 50-step, then CPU cascade on both 1-step outputs. Keep every original training command, checkout and output unchanged. Use the actual final update if training finishes early. Reports 012 onwards will contain the matched 1/20/50-step summaries and cascade outputs.

A one-step result is the unconstrained clean occupancy estimate z+v(z,0), thresholded at 0.5; it is not a calibrated probability. Differences across step counts diagnose sampler sensitivity, and alone do not uniquely identify a failure cause.

[Compressed log, JSON, provenance and checksums](../evidence/20261010-round2c-cascade-shift/README.md)
