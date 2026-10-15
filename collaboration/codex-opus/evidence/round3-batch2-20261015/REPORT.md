# Round 3 Batch 2 Part A frozen evaluation

- Queue status: `complete`; complete delivery: `True`.
- GPU evaluation complete: `True`; analysis complete after any formatting fallback: `True`. Original queue status is retained.
- Validated complete evaluations: 6/6 scheduled cases. Each complete case has 500 trees: original 100 and extra 400 are reported separately.
- Additional seeds estimate fixed-object robustness; they are separate from the official acceptance seed pair.
- Frozen evaluator console `/100` and `acceptance.passed` are oriented to the official seed pair; ten-seed results use validated per-tree coverage of 500, split into official 100 and extra 400.
- This collection validates saved JSON coverage and counts. No model forward, training, or checkpoint hashing was performed.

| Case | Original queue state | Evaluator exit | Original 100 | Extra 400 | Total 500 | Original analysis exits | Boundary fallback succeeded | Validation issue |
|---|---|---|---|---|---|---|---|---|
| steps08_ema | complete | 0 | 88/100 | 347/400 | 435/500 | {'boundary': 0, 'cascade': 0} | False |  |
| steps06_ema | complete | 0 | 95/100 | 379/400 | 474/500 | {'boundary': 0, 'cascade': 0} | False |  |
| steps05_ema | complete | 0 | 96/100 | 382/400 | 478/500 | {'boundary': 0, 'cascade': 0} | False |  |
| steps04_ema | complete | 0 | 96/100 | 387/400 | 483/500 | {'boundary': 0, 'cascade': 0} | False |  |
| steps03_ema | complete | 0 | 97/100 | 388/400 | 485/500 | {'boundary': 0, 'cascade': 0} | False |  |
| steps02_ema | complete | 0 | 98/100 | 390/400 | 488/500 | {'boundary': 0, 'cascade': 0} | False |  |

## Evidence

- The compressed archive contains completed evaluation JSON, available analysis and failure logs, exact commands, CPU gate, source files and code/checkpoint identities. Incomplete evaluations are not reported as completed metrics.
- `STATUS.json` records missing evidence and per-case validation errors. A missing launcher status is a failure boundary, not an empty successful batch.
- Recovered/adopted process exits remain unknown when unavailable. Their completed metrics require the original EVALUATION_COMPLETE log marker plus exactly 500 UID/seed pairs and consistent summary counts; no exit code is invented.
- A boundary CLI NoneType formatting failure can be recovered by a separate CPU adapter calling the unchanged frozen calculation functions. Original failure logs and exit codes remain; adapter output and provenance are separate. Such a formatting failure does not mean GPU generation failed.
- Checkpoint identity includes the saved update/step and whether a hash was recomputed; this collector does not strengthen that evidence.
- Step selection uses extra-400 full-tree exact only; ties select the lower step count. Part B training is outside this queue.
