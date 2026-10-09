# Codex → Opus: CPU gate failure before GPU preflight

Date: 2026-10-09. No training has started. Please resolve this execution blocker before changing the experiment.

## Source and execution

- Repo: `leoguohr/suxen`.
- Branch: `vertex-dense-overfit50-20261009`.
- Executed commit: `f0561d57f0ebded7d866ded60472e40bdeb9ef19`.
- The source working tree was clean after the test.
- Actual hardware: one host with four idle A100-SXM4-80GB GPUs, rather than two hosts with two GPUs. Planned assignment remains one independent process per GPU. No GPU process was launched.
- `/ssdwork/guohaoran` rejected a small Git initialization write with `Disk quota exceeded`. A fresh output directory was selected under `/tmp` as allowed by the run sheet. It had about 348 GiB free; host available RAM was about 889 GiB. Existing experiments/checkpoints were not modified.
- Output: `/tmp/nexus_vertex_dense_overfit50_20261009` (ephemeral).
- The accepted time window ends at 19:27 Asia/Singapore on 2026-10-09. No training budget T has been assigned because the CPU gate failed.

Exact CPU command (remote bash, with pipeline failure propagation):

```bash
set -o pipefail
cd /tmp/nexus_vertex_dense_overfit50_20261009/source/vertex/dense_overfit50_20261009
CUDA_VISIBLE_DEVICES= /guohaoran/envs/nexus-algo/bin/python tests/test_cpu.py 2>&1 | tee /tmp/nexus_vertex_dense_overfit50_20261009/test_cpu.log
```

Exit code: **1**. Full stdout/stderr: [test_cpu.log](../evidence/20261009-cpu-gate/test_cpu.log).

## Observed results

| Check | Result |
| --- | --- |
| Fixed 50-object data and SHA256 | PASS |
| Packed forward vs original | PASS; max absolute difference 1.79e-07 |
| Per-item loss vs original velocity MSE | PASS; max relative difference 2.20e-07 |
| Loss weights sum to 1 | PASS |
| Equal total weight per object/depth | PASS |
| Gradient comparison | Not completed: optimizer construction raised before this check could finish |
| Later CPU checks / ALL CPU TESTS PASSED | Not reached |

Failure path:

```text
tests/test_cpu.py:114 -> torch.optim.SGD(...)
-> torch._dynamo -> torch.onnx -> onnx
-> onnx/onnx_ml_pb2.py:33 -> _descriptor.EnumValueDescriptor
-> google/protobuf/descriptor.py
TypeError: Descriptors cannot not be created directly.
```

Package metadata (not a clean-install compatibility experiment):

| Package | Metadata version | Loaded/discovered location |
| --- | --- | --- |
| torch | 2.3.0a0+6ddf5cf85e.nv24.4 | /usr/local/lib/python3.10/dist-packages |
| onnx | 1.16.0 | /usr/local/lib/python3.10/dist-packages |
| protobuf | 4.24.4 | /guohaoran/envs/nexus-algo/lib/python3.10/site-packages |

The directly observed blocker is the **actually loaded ONNX generated descriptor code conflicting with the loaded protobuf runtime**. This does not establish that every installation of ONNX 1.16.0 and protobuf 4.24.4 is incompatible. We have not established why this image contains this combination of runtime/generated files.

Earlier `import torch, scipy, numpy` succeeded. Optimizer construction reaches additional lazy imports that the earlier import check did not exercise. This is not evidence of a diffusion/model failure.

## Boundary and requested response

The execution card says to stop when a gate fails, without patching. We therefore did not modify source, install/downgrade packages, set a workaround variable, retry the gate, or run GPU preflight/training.

The exception message itself suggests a pure-Python protobuf implementation or an older protobuf runtime. **Neither suggestion has been tested or applied.** Please propose the smallest environment-only handling, with an exact scoped command and the intended environment impact, or provide a corrected execution card. We will rerun the unchanged CPU gate before moving on. Model/loss/hyperparameters are unchanged.

Previous preparation separately hashed S0@34000 (27,990,535,028 bytes), matching `3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935`. The checkpoint identity gate in the current run sheet is still marked not-run because step1 stopped execution. There has been no actual model replay in this run.

## Evidence

- [Environment package metadata](../evidence/20261009-cpu-gate/environment_diagnostic.json)
- [Execution status](../evidence/20261009-cpu-gate/execution_status.json)
- [Run sheet snapshot](../evidence/20261009-cpu-gate/RUN_SHEET.executed.md)
- [File SHA256 values](../evidence/20261009-cpu-gate/SHA256SUMS)
