# Verified point overfit and point-to-topology cascade

Point overfit was independently verified before the cascade gate was released.
The two noise-seed coordinate RMSE values are 6.08063513e-08 and 6.05511849e-08.
Every sample has the correct vertex count and every vertex error passes the
minimum-spacing criterion. See point_prior_candidate/independent_verification.json.

The final model adds a learned text-coordinate prior to the frozen original
denoiser. Its residual scale is 1.21982403e-05; generated coordinates
are therefore dominated by the learned text prior. This is memorization of
the 50 training prompts, not demonstrated unseen-text generalization. No GT
coordinates or faces are fed to the cascade and no mesh repair was applied.

All 50 meshes and point clouds have been saved and independently checked.
Cascade count accuracy: 100%.
Cascade global matched coordinate RMSE: 6.78879493e-08.
Mean symmetric squared Chamfer: 1.23704621e-14.
Cascade matched face micro F1: 0.991093117 (TP 5508, FP 31, FN 68).
Vertex matching is used only for evaluation; it does not modify predictions.
The original fixed-GT-vertex topology F1 must not be substituted for this value.
