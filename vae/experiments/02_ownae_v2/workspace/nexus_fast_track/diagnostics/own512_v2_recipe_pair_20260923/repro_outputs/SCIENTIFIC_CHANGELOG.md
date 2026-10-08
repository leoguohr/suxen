# Scientific changes authorized for this experiment

This independent from-scratch experiment applies the user's supplied GOAL_TRAIN_MY_V2.md. Both architectures use the new five-mesh AdamW recipe, Hard4 reconstruction, resampled legal nonGT Face negatives and fixed deterministic μ path. B additionally uses the explicitly requested Fourier, Graph and per-decoder-block FFN architecture. The original experiment code, data, pool and checkpoints are preserved.

The protocol is a new architecture/recipe comparison, not a continuation of B2500 and not an exact recovery of the teacher's trainer. Empty Hard4 groups contribute zero and the outside divisor stays four as an explicit implementation convention.

After training launch, no frozen training/evaluation source changed. Only CPU postprocessing and a completion coordinator were added. When the third-server evaluation process disappeared, fallback was restricted to an already allocated training GPU after its branch naturally releases it. No LR, loss, source, update budget or data changes were made to respond to intermediate quality.
