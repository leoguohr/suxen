# Comparison anchors

Parent: B19356, SHA2566f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2.
Actual Face TP/FP/FN5570/47/6, Face F1=0.9952648977039221,
Edge TP/FP/FN8361/21/3, joint strict38/50, Edge strict40/50.

This is a bounded continuation from this trained parent, not another
from-scratch architecture comparison and not a rerun of the historical A/B pair.
Compare actual reconstruction before/after; do not compare training losses as
if the candidate sets were unchanged. All50 meshes remain included.

The expected Face threshold is checked on complete actual reconstruction.
No Edge repair, GT candidate completion, threshold adjustment or output cleanup.
Any result >=0.997 requires fresh-process full-model bitwise prediction replay.

The fixed100 largest-mesh preflight cannot establish learning, reconstruction
accuracy or whole-epoch throughput. Its measured activation/gradient peak
excludes Adam moments and optimizer-step temporary buffers. Random weights,
unchanged2547-vertex complete mesh and all3242331 pairs are used.
