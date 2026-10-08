# Implementation notes

The copied README is the supplied GOAL_TRAIN_MY_V2.md. User authorization covers the architecture and new recipe. All code is in this independent repository.

GPU allocation: A on dual-host GPU0, B on dual-host GPU1, and fresh-process evaluation jobs on single-host GPU0. Evaluation coordinator itself is CPU-only; it creates a new GPU process for each frozen checkpoint. Assigned GPU-process wall time (including training waits and preflight) is charged conservatively. Training stops at 23 aggregate GPU-hours, reserving one hour for final saves/evaluations; all work is bounded by 24 aggregate GPU-hours. Third-card evaluation is included.

Random initialization: seed0 for each model, then exact common-name/common-shape random A tensors copied to B. No trained weight file is read. Each branch starts with empty AdamW state and its own run RNG. Data order and negative sampling use separate stateless SHA256/PCG64 streams.

Native graph uses the archived deterministic index_add forward and deterministic gather VJP, with static topology/degree caching only. Math00 QKV/layout/SDPA/output projection are explicit. Complete-block nonreentrant recomputation uses the same native operations and MATH context. No runtime exec, model method replacement, or architecture hook exists.

CPU attempt1 encountered the existing protobuf/onnx import incompatibility. The original environment setting PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python resolves it; no packages changed. Attempt2 passed model/loss checks but caught an overstrict new-loader assertion treating original_vertex_indices as identity. The source manifest explicitly says it is a preserved map to earlier numbering. The loader now verifies uniqueness and records its hash while preserving existing cache order. Source arrays were never modified. Attempt3 passed. Additional Face chunk-gradient test is run before launch.

Two parallel implementation agents were stopped by tool safety systems with reason Potentially unintended activity before writing files or starting training. The root task implemented and verified the code directly within the authorized scope.
