# CAD50 own512 architecture comparison

No teacher or B2500 weights were used. A is the original backbone with the new recipe; B is the Fourier / changed Graph order / per-block FFN V2 with the same recipe.

Latest jointly evaluated budget: 19356 optimizer updates, each involving five complete meshes.

| Branch | Step | Actual Face F1 | Edge FP/FN | Large16 Face F1 | Strict /50 |
|---|---:|---:|---:|---:|---:|
| A_v1_recipe_control | 19356 | 0.686776634 | 1039/954 | 0.653423414 | 34 |
| B_v2_teacher_blocks | 19356 | 0.995264898 | 21/3 | 0.995068061 | 38 |

## Stop status and scope

Paired recorded batch/negative/LR audit exit code: 0.
- A_v1_recipe_control: completion={"state": "training_complete", "completed_updates": 19356, "mesh_participations": 96780, "stop_reason": "original_physical_budget_conservative_common_stop", "final_checkpoint": {"path": "/guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01/A_v1_recipe_control/checkpoint-19356.pt", "bytes": 1348064782, "sha256": "df5439ff36535d5aadd35361effa0f96b63c2e7298546de9b8a9d41112544501"}, "maximum_updates_respected": true, "new_recovery_updates": 9356, "original_recorded_updates": 10643, "conservative_physical_updates": 20000}; failures={}
  Highest observed Face F1=0.686776634 at step 19356; highest strict coverage=34/50 at step 11000.
- B_v2_teacher_blocks: completion={"state": "training_complete", "completed_updates": 19356, "mesh_participations": 96780, "stop_reason": "original_physical_budget_conservative_common_stop", "final_checkpoint": {"path": "/guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01/B_v2_teacher_blocks/checkpoint-19356.pt", "bytes": 2960530710, "sha256": "6f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2"}, "maximum_updates_respected": true, "new_recovery_updates": 10356, "original_recorded_updates": 9464, "conservative_physical_updates": 19821}; failures={}
  Highest observed Face F1=0.995264898 at step 19356; highest strict coverage=38/50 at step 17000.

Face F1 >= 0.997 and strict 50/50 are separate outcomes. This compares whole architecture versions and cannot isolate Fourier, Graph order or FFN effects. Parameter counts differ (A 112,212,544; B 246,575,680).

Third-card connectivity and worker were lost during training. Any later evaluation fallback uses only a naturally released, already allocated training GPU. Checkpoint evaluation may therefore be delayed; no training budget is added.

This is an automatically generated evidence summary. Human scientific interpretation remains pending. Old fixed100 experiments and teacher diffusion/prior networks are outside this run. No follow-on training is launched.

## Artifact layout

All experiment code, configs, logs, full evaluation metrics and prediction arrays are included in independent ZIP shards. Full model/Adam/RNG checkpoints remain on the server and are listed with sizes and save-time SHA256; final checkpoint hashes are independently rechecked. Linked source data remain at their original paths.
