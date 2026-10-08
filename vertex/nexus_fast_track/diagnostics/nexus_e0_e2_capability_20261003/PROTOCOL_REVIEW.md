# NEXUS E0 / E1 / E2 protocol review — 2026-10-03

Scope: a new, controlled diagnostic of frozen A24000 and a depth-9 training ablation. **The user has now approved E2: three independent A24000 full-model/Adam/RNG continuations, 2000 updates each. E2 may start only after E0/E1 finish, global E0 numerical consistency passes, and complete source restoration is verified.** E3–E6 are conditional later work, not automatic continuations. This document records the protocol before the new GPU results; it does not certify that a run has passed. No frozen runtime was edited for this review.

## 1. Independent formula check

Use local time `t` increasing from noise at 0 to data at 1. Let `y` be binary child occupancy, `epsilon` the actual stored FP32 noise, and `v_theta` the model's velocity output. Current frozen code implements:

```text
x_interp(t) = (1-t) epsilon + t y
v_target    = y - epsilon
loss        = masked per-object mean[(v_theta - v_target)^2]
yhat(t)     = x(t) + (1-t) v_theta(x(t), t, parents, depth, condition)
x_next      = x + dt v_theta,       dt > 0
```

These equations are mutually consistent. On the prescribed interpolation bridge, `yhat-y = (1-t)[v_theta-(y-epsilon)]`, so clean MSE has an automatic `(1-t)^2` factor. Lower clean MSE at larger t alone is not better velocity learning. Threshold `yhat >= 0.5`; no clamp, sigmoid, or occupancy postprocessing before metrics.

Current evidence: frozen `../nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/flow.py` (SHA256 `9730d962dbee230e7a18eac0be1f88b2d697a8bb7e6b42b33e8db20dfafd712b`) defines the bridge, velocity loss and forward Euler direction. The actual A/B training call uses `paired_noise` in `../nexus_d9_ab_20260930/train_ab.py:260–291`: Gaussian noise and **uniform** random time, then the original model loss under BF16 autocast. The explicit training sampler, not merely the helper's default, establishes the current time distribution.

### Pinned official comparisons

- **TRELLIS:** its time `s` increases in the opposite direction. With `a = sigma_min`, `x_s=(1-s)y+[a+(1-a)s]epsilon`, target `v_s=(1-a)epsilon-y`. It samples with decreasing s and `x_next=x-(s-s_next)v_s`. Substituting `t=1-s`, `v_t=-v_s` gives `x_t=t y+[1-(1-a)t]epsilon`, `v_t=y-(1-a)epsilon`. At `a=0` this is exactly the local convention. At nonzero a it is a related path with residual endpoint noise; do not silently drop a or call the paths identical. Its clean projection becomes `(1-a)x+[1-(1-a)t]v_t`. [Pinned trainer, lines 69–104](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/trainers/flow_matching/flow_matching.py#L69-L104); [pinned sampler, lines 32–76](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/pipelines/samplers/flow_euler.py#L32-L76).
- TRELLIS's constructor defaults to `sigma_min=1e-5` and logit-normal time; `sample_t` explicitly supports uniform as well. A constructor default is not proof of every training configuration. It sends `1000*s` to its denoiser; this is that model's time-conditioning convention, not a reason to rescale the already-trained local model's inputs. [Defaults](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/trainers/flow_matching/flow_matching.py#L52-L62), [time sampling](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/trainers/flow_matching/flow_matching.py#L130-L142), [loss](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/trainers/flow_matching/flow_matching.py#L160-L172).
- **Hunyuan3D-2.1:** its Euler implementation uses `x_next=x+(sigma_next-sigma)*model_output`; its consistency scheduler explicitly computes clean projection `x+(1-sigma)*model_output`. These confirm the corresponding update and projection algebra, not the complete training protocol or identical target domains. The generic `scale_noise` helper alone must not be treated as a complete statement of the pipeline's time convention. [Pinned Euler, lines 301–310](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/schedulers.py#L301-L310), [pinned consistency projection, lines 462–469](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/schedulers.py#L462-L469).

Official files retrieved at the pinned commits for this review:

| Source | SHA256 |
|---|---|
| TRELLIS `flow_matching.py` | `1dd40748ec814819ef5c0338caab81117b4a1d0dfe98c9a544d4ebc391939e78` |
| TRELLIS `flow_euler.py` | `77060b663ac731699249bbb3b54647cbce2f85ccf7a120bd123b22012a38bba4` |
| Hunyuan `schedulers.py` | `d3b25b84dac26f0e862cfc866da08df2d6592f01a0aaeda965782ab0fc1d8bba` |

The local teacher file `/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/objectives.py` (SHA256 `882af74504ccf714c1a5f9267e3adecf8d1db62fc6301de28eaa5dd038f58296`) is explicitly a reconstruction, not original training loops. Its topology objective, lines 33–41, uses the same noise→data bridge and velocity target. Its point objective, lines 44–60, predicts a clean coordinate endpoint and converts it to velocity `(x1-x_t)/(1-t)` before velocity MSE. Thus it illustrates an alternative output parameterization; it is not evidence that current binary occupancy should silently switch to clean-output prediction, BCE, or count loss. E0/E1/E2 preserve the current velocity output.

## 2. E0 — actual frozen-model replay, with a hard gate

Freeze checkpoint SHA256 `2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`, the original manifest's 50 UIDs, seeds `97029000` and `97029001`, and depth 6/7/8/9. Use t=0/0.25/0.5, i.e. saved steps 0/5/10, from both interpolation and actual Euler sample paths. Total: `50*2*4*3*2 = 2400` flow forwards. **Both t=0 path entries independently call the model**, even though their inputs are equal. No reuse of saved velocity counts as replay.

1. Validate checkpoint and source hashes, manifest/order, raw condition hashes, integer parents/child ordering, labels, actual epsilon, saved times and array dtype/shape. Compare arrays bitwise; do not regenerate noise from a seed. Independently reconstruct the bridge from y/epsilon/t and compare with saved interpolation state. Verify the saved Euler recursion and that x0 equals saved epsilon.
2. Call frozen `model.flow(saved_x, saved_t, parents[None], depth_tensor, context)` in eval/inference mode with original batch=1, BF16 network autocast and FP32 states/projections. Re-encode frozen conditions from their original arrays or use a cache produced by that same frozen encoder; record its provenance. Preserve the original attention backend and record GPU, PyTorch, CUDA and precision settings. Two GPUs may shard complete cases without changing a forward's shape.
3. Save every fresh velocity and clean projection; compare each against its own archived path output. Compute clean projections independently from saved x/t and **new** velocity. An exact formula test using only saved velocity is useful but does not substitute for this check.
4. Only after every required case on both shards passes and all hashes/call counts reconcile, create a global E0-pass artifact. E1 must require that artifact. Missing, failed, incomplete, NaN/Inf or a single out-of-tolerance element stops downstream. Preserve failure evidence; never widen thresholds after seeing results.

**Predeclared floating-point tolerance:** for every velocity and clean element, require

```text
abs(new - archived) <= 1e-6 + 1e-5 * abs(archived)
```

Also compare the two independently replayed t=0 path outputs using the same criterion. Report bitwise-equal count, maximum absolute error, RMSE, relative L2 (null when reference norm is zero), out-of-tolerance count, and threshold-flip count. This intentionally strict bound checks same-shape/same-runtime reproducibility; it is much smaller than typical BF16 rounding at unit scale. A changed kernel, batch shape or arithmetic precision is not excused by the bound. If necessary, diagnose and restore the original execution convention in a separately identified retry while preserving the failed run. A failure is a failed reproduction gate, not immediate proof that A24000 is wrong.

Threshold flips that occur despite numeric agreement must be identified with their old/new distances to 0.5 and carried into metric explanations. Do not hide them with only average metrics or interpret a tolerance-scale change in exact status as model progress. Integer source integrity and flow-call counts have no numerical tolerance.

## 3. E1 — fixed condition intervention and restoration

Run only after global E0 pass. Use exactly the same 2400 input cases, frozen model, states, parents, time, target and epsilon. The sole intervention is which original point cloud supplies the conditioning context.

- Construct one fixed derangement from manifest order: UID at index i receives cloud from `(i+1) mod 50`. Record all 50 original/wrong UID pairs and both cloud hashes. Assert no fixed point and no dropped or duplicate donor. A donor's parents or occupancy must never replace the recipient's.
- At each case, perform three fresh forwards in order: correct cloud → fixed wrong cloud → correct cloud restored. Save all outputs and condition IDs. No reuse of an E0 output or the first correct output for restoration. This is 7200 additional flow calls.
- Require first/restored correct outputs to satisfy the same E0 tolerance and check the first correct output against its E0 replay. Failure invalidates that intervention and stops reporting a complete E1 result. Model weights and mutable evaluation state must remain unchanged.
- Frozen encoder-context caching is valid for E0/E1 because weights do not change. It is invalid for a later E2 full-model training run whose encoder remains trainable.

For each UID/seed/depth/time/path/condition, store all-bit, occupied-bit and empty-bit MSE, plus TP/FP/FN/TN, F1, and exact-layer status (FP=FN=0). Keep sums and counts so both macro and pooled reductions can be recovered; an absent class has MSE null, not zero. Keep occupancy and empty velocity MSE separate.

**Velocity-reference names must distinguish the two paths:**

```text
interp: training_bridge_velocity_target = y - epsilon
sample: endpoint_velocity_reference    = (y - x_saved)/(1-t)
```

For sample states, `endpoint_velocity_mse` measures the direction needed for a one-step linear projection to this designated y. It is **not** the actual training epsilon target or the true ODE vector field label. Its clean MSE is exactly `(1-t)^2` times this endpoint-reference velocity MSE in exact arithmetic, so these are equivalent diagnostics at fixed t, not independent evidence. If `y-epsilon_initial` is also measured on sample states, name it `off_path_initial_bridge_reference_mse`; never call it training loss. At t=0 these references coincide. At t=0.25/0.5 all denominators are safe and must not be clamped or otherwise changed.

Aggregate intervention effects as paired wrong-minus-correct MSE/FP/FN and correct-minus-wrong F1 so positive denotes degradation. Form each UID's mean across its two seeds first, then summarize 50 UID differences by mean, median, quartiles and positive/zero/negative count, separately for each depth/time/path. Report restored-minus-first as a reproducibility control. Exact results are counts out of 100 UID/seed cases and, separately, UIDs with both seeds exact; pooled counts/F1 must be labeled separately. Do not treat 100 seeds or child bits as independent objects for an uncertainty claim.

Interpretation: a consistent wrong-cloud degradation with restoration supports **functional use of condition by this checkpoint at these inputs**. Little effect means the model is insensitive under this intervention and may also reflect target information already in x/parents; it does not prove the VecSet representation lacks information. Wrong condition paired with unchanged recipient y/parents creates a deliberately inconsistent input. It is a sensitivity diagnostic, not an in-distribution generative benchmark. Larger-t bridge states contain more of y directly, which can hide reliance on the cloud. These 50 known training objects do not measure unseen-object generalization.

## 4. E2 — approved training ablation; prerequisite checks remain mandatory

The following three arms are **independent trainings**, not just three evaluations of frozen A24000. All use only depth 9, original 50 objects and GT parents, unchanged architecture and velocity output, unchanged per-object velocity-MSE reduction and target bit order.

| Arm | Training t | Training epsilon | Model input x | Velocity target |
|---|---|---|---|---|
| C | 0 | 0 | 0 | y |
| N | 0 | fresh Gaussian | epsilon | y-epsilon |
| T | uniform in [0,1) | same paired Gaussian as N | (1-t)epsilon+t y | y-epsilon |

C is a zero-input deterministic mapping control at t=0; the name does not imply feeding y to the network. All three still predict velocity. Any later change to clean-output parameterization is a separate experiment.

**User-approved budget and source:** start each arm from an independent copy of the full A24000 model + Adam state + RNG; 2000 updates each, LR 1e-5, weight decay 0, **global** 8 microexamples per update, evaluation at updates 0/1000/2000. Each arm therefore consumes exactly 16000 object events; each of the original 50 UIDs appears exactly 320 times, always at depth 9 with its original GT parents. Checkpoint step is 24000+update; the source step is not reset to zero. No early stopping or budget extension is inferred from intermediate scores. If the source lacks required optimizer/RNG fields, stop and report it; do not silently reset.

Before the first training update, persist a shared 16000-row UID/event table and **actual FP32 noise and uniform-time arrays**, with hashes and event/UID/shape correspondence. Each UID has 320 rows. The schedule is identical for all arms. N and T read the very same epsilon for each row; C uses an exactly zero array of the same shape. C/N use exact FP32 t=0; T reads the row's saved uniform time. Do not save only seeds and regenerate later, or let a T time draw shift future noise/data order. Preparation uses a separately recorded private generator, does not consume the source global training RNG, and completes atomically before launch. Record source RNG restoration and table generation as distinct operations; table pairing intentionally replaces the previous run's input-draw schedule.

Restore all model tensors strictly, with the same architecture, parameter names/shapes/dtypes, and parameter ordering used by the saved Adam groups. Require complete Adam state for every expected parameter (the frozen full model has 907 parameter entries), including step, exp_avg and exp_avg_sq; check shape/dtype/finite values and require source steps 24000. Preserve the source group settings, including betas/eps/foreach and scheduler state; confirm LR 1e-5 and weight decay 0 rather than replacing missing metadata with defaults. Do not slice depth embeddings, migrate architecture, reset moments, use AdamW, or inherit another arm's updated state. Source torch/CUDA/Python/NumPy RNG must all be present, restored after construction/preprocessing, and read back equal before the first update; record any GPU-device remapping explicitly. Eval0 is before any optimizer step. Any disposable performance preflight must be followed by a fresh full restoration, never by continuing from the tested state.

**Precision, reduction and clipping boundary:** keep FP32 parameters, gradients, Adam moments, inputs, targets, bridge states, noise/time and projections, with the original BF16 network autocast. The frozen training forward explicitly computes `prediction.float()-target_velocity.float()`, squares in FP32, divides by each object's valid parent count times 8, then averages objects. The optimizer loss is the mean of exactly 8 object losses per global update; no weighting by total bit count, occupied/empty balancing, BCE, clean-loss replacement, new loss scaling or GradScaler. After all 8 microexamples have accumulated and all participating ranks have synchronized, call the original `clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)` once, then one Adam step. Do not clip each microexample, each object, or unsynchronized local-rank gradients. Preserve original TF32/attention/backend settings and record them. No clamping of x, v or yhat. Evaluation computes FP32 squared errors with FP64 sums/counts for stable reporting, independently of the FP32 training reduction.

All learned parameters, including the condition encoder, remain trainable. Original immutable FPS-index/Fourier-input caching may be reused; cached learned embeddings or detached contexts may not. No activation checkpointing is needed merely because the original constructor enables it: the actual A training entry disables checkpointing in both flow and encoder. Keep that execution policy unless a separately documented equivalent performance choice is necessary.

### Two evaluation views at update 0, 1000 and 2000

Before training, save new **actual** evaluation epsilon arrays for the original 50 objects, with two replicas per UID, their hashes and an independent fixed uniform t_eval array. The task-matched view and common-probe view may use separate named noise sets; each set must be shared across arms and all three checkpoints. These inputs must never appear in the new training event table. Check actual array-hash membership as well as using separate generator namespaces; do not claim knowledge of unrecorded historical training noise. Evaluation preserves/restores global RNG and train/eval modes, and makes no optimizer step.

| View | C | N | T |
|---|---|---|---|
| Task-matched | x=0, t=0, v*=y | x=epsilon_task, t=0, v*=y-epsilon_task | x=(1-t_eval)epsilon_task+t_eval*y, fixed uniform t_eval, v*=y-epsilon_task |
| Shared bridge probes | All arms: the same epsilon_common at the original FP32 grid t=0,0.05,...,0.95; preserve parents/condition/y |
| Shared sampling probe | All arms: actual 20-step FP32 Euler from that same epsilon_common, depth-9 GT parents fixed for the entire trajectory, increasing t=0 to 1 |

The T task-matched time table is fixed **uniform random**, not the common 20-point grid: it approximates that arm's training distribution. The common grid supports controlled cross-arm/time comparison and includes the t=0 boundary. Report the two views separately. C has one distinct zero-input result per UID; if duplicated into two seed slots for table alignment, mark those duplicates and do not count them as independent samples. No fresh uniforms/noises are drawn at later checkpoints.

For the shared Euler probe, save the actual trajectory and evaluate the final x20 at threshold 0.5. Also retain yhat(t) and the same occupancy/empty metrics at the stored grid points. At each sample state, distinguish endpoint-velocity reference from the training bridge label as in E1. This fixed-GT-parent depth-9 trajectory is not a full free-tree generation. Task-matched scores alone cannot rank the different objectives; compare arms within the same shared-probe column and also compare each arm's change from update0. Differences between task and common views can additionally reflect different noise draws, so do not interpret them as a same-noise intervention. Because all arms start from identical model state, same-input update0 outputs must meet the predeclared E0 tolerance before comparing learned changes.

Repeated update0/1000/2000 probes are **development evaluations with noise excluded from these new trainings**. They are not unseen-object tests or an untouched final holdout. There is no automatic additional holdout generation/evaluation. A later once-only held-out-noise claim needs a separate predeclared, unused array set. All table identities, uses and hashes must be logged. Equal update/example budgets, not equal wall-clock time, define the comparison.

Interpretation is conditional and finite-budget; **2000 updates is an execution budget, never an information/capability decision threshold**. Report continuous changes and exact reconstruction counts; do not label a weak final score as proof the arm cannot learn:

- C succeeds, N fails: evidence for a difficulty introduced by random-noise input/epsilon cancellation under this training setup; not a proof of a universal architecture limit.
- N succeeds, T fails: evidence for a difficulty introduced by the multi-time objective/coverage at the matched budget. It does not by itself isolate uniform versus logit-normal sampling as the cause.
- All fail: unresolved optimization, conditioning, capacity or protocol issues. Failure within a budget cannot establish information-theoretic impossibility.
- T succeeds on prescribed bridges but fails on actual sampling states: motivates a later trajectory/integration investigation; it does not already establish an integrator defect.
- Any success with GT parents is a conditional level-reconstruction result, not full free-tree success. GT parents themselves carry substantial structure; C success alone does not show the point-cloud context is sufficient.

## 5. Execution efficiency and follow-up boundary

Use both available approximately 80 GB GPUs for useful work, prefer independent inference shards or independent training arms where appropriate, and measure actual cases/second or updates/second plus allocated/reserved/peak memory. Approximately 70 GB is a throughput target, not a reason to allocate unused tensors, alter loss weighting, expand the statistical batch, or change numerical semantics. Do not add activation checkpointing solely from habit; benchmark within memory headroom and preserve the accepted comparison. The inference gate retains batch=1 unless a separate equivalence check is explicitly approved before replacing it.

An E2 DDP implementation must preserve the original effective microexample count and the exact mean over objects, account for DDP's gradient averaging, and validate one matched update before a long run. Parallel C/N training followed by T is permissible only if each arm loads the same declared source and does not inherit other-arm state. No extra training, automatic restart, full-tree evaluation or supervision after startup is authorized by this protocol. E3/E4 require their triggering evidence and a separate decision; E5 integrator tests come later; E6 full-tree evaluation comes only after the applicable gate/result.

Review status: formulas/source references checked; E0 numeric criterion communicated to implementation before new model forwards; E0/E1 execution results pending; E2 source/budget/evaluation views approved by user, implementation and source-restoration checks still pending.
