# E2 implementation review — 2026-10-03

No blocking source-review finding in the three E2 implementation files at the following hashes. This review supports deployment and the required real-source/GPU preflight. It does not report training completion, numerical success, or measured GPU throughput. The reviewer performed no SSH, GPU forward or training.

| File | Reviewed SHA256 |
|---|---|
| `e2_protocol.py` | `5329cc1543e8e38f9e89da020476849c685ffb136d429f5820927a2febca39fc` |
| `train_e2.py` | `38419a40c68a085f8e6f85517ba56cf3ebe8b95c85f8f73eabf5799eae81b7d9` |
| `evaluate_e2.py` | `1d04e20273353d66eaad6efdeb76035bd43512f01ee349d881387a14521a67ab` |
| `test_e2_cpu.py` | `e0f19e692951bbd7853f1470bbaf28eeb14330541193ef95922ed5f47eddec79` |
| `aggregate_e2.py` | `029f0d5c1d9231f23dbfcc5e7bab6e3cf75c705e860fde1638d9b3cbe53aa7eb` |
| `run_e2_queue.py` | `f5b5b9a960ae3a4aeafc55d589b3a1ad57c7d199875abcc7fd7bd017f7a0e90a` |
| `audit/source_state_verified.json` | `9e849a9b3919c03ffd2cb4ce147af39d3ea49a8a572bd5a7d4ed934be7f3db84` |

## Verified implementation boundaries

- Every arm independently loads the exact A24000 SHA, unchanged depth-9 runtime and strict full model state. Complete source Adam entries are required for 907 parameters, each at step 24000, with moment shape/dtype checks, finite-state checks and exact post-load value comparisons. Source groups preserve Adam, LR 1e-5, weight decay 0, betas 0.9/0.999, eps 1e-8 and foreach=False. Torch/CUDA/Python/NumPy RNG is restored and read back equal. No model/optimizer state is inherited from another arm.
- The persisted event table has 16000 events, 320 appearances of each original UID. Actual FP32 noise and time arrays are saved before training, hashed and revalidated. C uses zeros; N/T share the exact stored noise row; only T reads the uniform-time row. Shared preparation uses private CPU generators. Actual training and task/common development noise hashes are checked for no overlap, not merely assigned different seed labels.
- Each single-GPU arm performs exactly 8 microexample losses per optimizer update. The frozen model's FP32 per-object masked velocity MSE remains intact; all learned parameters must have gradients. One global norm clip at 1.0 follows accumulation, then one original Adam step. FP32 parameters/Adam/state and BF16 network autocast are preserved. Immutable FPS/Fourier caching keeps learned encoder features live. Activation checkpointing remains disabled, matching actual source A training.
- Evaluation at updates 0/1000/2000 uses fixed actual saved arrays. Task-matched T uses its fixed uniform table; the shared probe uses the original 20-point time grid and traced 20-step Euler with GT parents fixed at depth 9. All velocity/clean metrics split occupied/empty cells; sample endpoint-reference velocity is labeled separately from the training bridge target. Arrays, per-case rows and summaries are hashed. Per-UID two-seed metrics and exact counts are retained. C explicitly labels its duplicated zero-input seed slots as 50 distinct inputs, not 100 independent observations.
- Evaluation records actual model outputs and trajectories, preserves/restores RNG and train/gradient flags, and never steps the optimizer. Each arm independently evaluates update0. `compare_eval0` performs a final CPU comparison of their common saved inputs/outputs using the E0 tolerance; interpreting cross-arm learning changes requires this check to pass. It adds no model forward.
- Coverage at completion must be exactly 320 depth-9 events per UID, no new events at other depths, and Adam step 26000. Full model/Adam/RNG checkpoints are written atomically, with update1000/2000 retained. A disposable performance preflight performs one separate update, marks it discarded, and must never become the source of formal training.

The root-produced source audit records A24000 SHA `2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`, 907 model tensors, 907 Adam states all at 24000, all four RNG types, scheduler=None and the original parameter-group settings. This establishes that the required source fields exist in the real checkpoint; each launched arm still performs its own complete restoration checks.

## CPU validation and execution limits

The implementation agent reports all 5 CPU tests passing. Inspected tests cover table pairing and formulas, absence of global RNG consumption, complete Adam restoration and rejection of partial/wrong-step state, an 8-microexample update against an explicit eight-object mean plus Adam reference, actual traced evaluation with RNG/mode/weight preservation, metric reductions and gate rejection. These tests use small CPU fixtures; they do not verify A24000 GPU memory, kernel behavior, or throughput.

The queue's poll-before-launch ordering and pending-state fix were inspected: C/N run independently on two isolated GPUs; T uses the first available GPU after a successful arm completes; observed failure prevents pending launches while already-running arms finish their authorized budget; there is no automatic retry. The final root implementation calls `aggregate_e2.py` before declaring the queue complete and before packaging. That aggregator first calls `compare_eval0`, then validates the nine declared evaluation completions/file hashes and training event coverage, and writes `RESULTS.json`/`RESULTS.md`. Its consumed fields were checked against the actual train/evaluate schema. This final aggregation has no GPU forward. Final queue/aggregator hashes are also captured by the deployment audit.

The 2000-update budget is not a capability decision boundary. These are fixed-GT-parent diagnostics on known training objects with development noise excluded from the new training tables, not an unseen-object or full-free-tree success claim. E3–E6 remain separate conditional decisions. No supervision after startup, automatic restart, extra budget, changed objective or fake VRAM allocation follows from this review.
