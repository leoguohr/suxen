# Execution and recovery evidence

- CPU parent SHA/model/Adam audit confirmed B19356 and inherited AdamW LR1e-4.
- Two newly allocated A10080GB GPUs were idle. GPU0 served Face finishing;
  GPU1 served fixed100 zero-update preflight. The unavailable single-card
  endpoint was not used.
- preflight100.py passed all100 data identities and one complete largest
  mesh forward/backward. It instantiated no optimizer and changed no weights.
- run-train attempt1 restored complete parent state and passed parent-array
  and cache-gradient gates. Exactly50 updates then completed before an incorrect
  participation-window assertion raised. This was an engineering check failure,
  not an observed NaN or a model/protocol change.
- The original failure-state, first code version and runtime records are kept.
  RECOVERY_50_AUDIT.json verifies all recorded batches, frozen weights/Adam,
  and Face Adam step19406. It promoted that complete saved state without an update.
- Evaluation50 then completed. run-train resume50 performed only remaining450
  updates. RESUME_RESTORE_AUDIT and cache recheck passed. Runtime completed
  successfully at exactly500 total new updates and11 full checkpoints.
- Face threshold0.997 was never reached. No target cold repeat and no fixed100
  training were performed. Two GPUs were confirmed idle after completion.
- RigorPilot ai-research-reproduction was loaded from the installed project skill;
  its run-train runtime retained commands, lifecycle, stderr/stdout and resources.
  No environment installation, CUDA/PyTorch changes, or optional model runner.
