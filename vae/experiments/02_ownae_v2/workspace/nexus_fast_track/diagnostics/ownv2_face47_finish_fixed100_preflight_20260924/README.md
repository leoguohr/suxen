# B19356 Face finishing and fixed100 zero-update preflight

Authorized 2026-09-24. Parent SHA256:
`6f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2`.

Only the original face_embedding weight and bias train. Upstream, final LN,
and Edge head remain frozen. Restore full model/AdamW/RNG, preserving LR1e-4,
weight decay0.01, clip1, Hard4 and five complete meshes per update. Continue
the original stateless random negatives and add all47 non-GT triangles from
the parent's predicted graph, deduplicated per UID. All50 meshes participate.
Maximum500 new updates; actual full-network evaluation at0 and every50.
If Face micro-F1 reaches0.997, stop training and verify all prediction arrays
in a fresh process. Never extend this budget. Retain complete frozen Adam
slots without updating them. Cache only frozen decoder outputs, and verify
cached versus real-path head predictions and gradients before training.

Fixed100 preflight: verify original selection and all mesh/topology hashes.
Use randomly initialized identical V2, all pairs, all GT faces plus the same
random-negative ratio for one largest-mesh full forward and backward.
Optimizer updates=0; no Adam is instantiated. Preserve FP32/MATH, original
coordinates, vertex order, topology and activation recomputation. Do not
enumerate the randomly initialized model's potentially dense triangle graph:
this is training resource preflight, not reconstruction evaluation.
An OOM is recorded explicitly; any retry may only change chunking/recomputation.

Original projects, data and checkpoints are read-only. No fixed100 long
training is launched; that requires a separately specified budget.
