# Authorized scientific changes

The user explicitly authorized this new Face finishing experiment. Relative
to B19356, only the original Face projection weight/bias remain trainable,
and the47 non-GT triangles in the parent's actual Edge candidate graph are
added as fixed hard negatives. Original random negatives continue; set union
deduplicates overlaps, so an already sampled hard negative is not double-weighted.
All5576 GT faces remain positive, including6 not present in the Edge graph.

Preserved: complete parent weights, all AdamW slots and RNG, native V2 forward,
Hard4 whole-mesh reduction, LR1e-4, betas0.9/0.999, eps1e-8, weight decay0.01,
clip1, five complete meshes/update, original stateless sample order, thresholds,
scales, centered representations, FP32 SDPA MATH and deterministic Graph.
No new warmup. Frozen modules have no gradients; their Adam states do not advance.

Caching the final frozen Decoder/LN output is now valid because only the
Face head trains. Complete evaluations always rerun Encoder-mu-Decoder,
re-enumerate the predicted Edge graph, and verify Edge scores/candidates against
the parent. The full/cache Face gradients are compared before the first update.

This targets Face F1>=0.997, not strict50. Frozen Edge errors21/3 and at most40
Edge-perfect meshes bound joint strict coverage. Fixed100 work is a separate
random-initialization resource preflight, with zero optimizer updates.
