# Comparability

A and B use the same fixed CAD50, full Edge pairs, deterministic UID order and per-participation negative sampling, loss, optimizer, LR schedule, five-mesh accumulation, μ-only FP32/MATH path and actual reconstruction evaluation. Actual UID sequences and negative-array SHA256 are audited at every common update.

The entire architecture version differs: B combines Fourier features, changed Graph normalization/activation placement, GELU encoder FFN and sixteen decoder FFNs. Trainable parameter counts are A112,212,544 and B246,575,680. It is not a parameter-matched single-component ablation. Initial common-name/shape random tensors match; different native functions need not have equal initial predictions. Neither branch reads teacher or B2500 trained weights.

Ranking uses same-budget actual full50 reconstruction, not training loss. When one branch stops early, use jointly evaluated steps for direct comparison and report each branch's own endpoint separately. The historical runs used different protocols and are context, not matched controls. Third-server loss delays later evaluations; unchanged saved checkpoints remain the evaluation targets.
