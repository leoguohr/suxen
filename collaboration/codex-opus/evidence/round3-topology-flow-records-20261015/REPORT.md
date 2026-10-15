# Historical Topology Flow records — Batch1 item6

Local CPU/file verification only, 2026-10-15. These records come from the 2026-10-08 archive; no training, inference, GPU use, or server inspection was performed for this publication.

## Results

Micro-F1 = 2TP / (2TP + FP + FN). Joint strict requires the complete edge and face sets to match. All complete evaluations use the same fixed 50 NEXUS2K objects.

| Experiment | Actual training updates | Evaluation coverage | Edge micro-F1 | Face micro-F1 | Joint strict |
|---|---:|---:|---:|---:|---:|
| C0 generation, update500 | 1000 total | 50/50 | 0.0339579703 | 0.0004502163 | 0/50 |
| C0 generation, update1000 | 1000 total | 50/50 | 0.0982065404 | 0.0038177981 | 0/50 |
| C1 generation, update500 | 904 total | 50/50 | 0.0568112904 | 0.0013022577 | 0/50 |
| C1 generation, update904 | 904 total | **41/50, incomplete** | — | — | — |
| C2 generation, update500 | 1049 total | **39/50, incomplete** | — | — | — |
| Frozen OwnAE-v2 reconstruction, mu | AE step36220 | 50/50 | 0.9995807479 | 0.9957316660 | 15/50 |
| Frozen OwnAE-v2 reconstruction, posterior seed0 | AE step36220 | 50/50 | 0.9995839689 | 0.9957267646 | 15/50 |
| Frozen OwnAE-v2 reconstruction, posterior seed1 | AE step36220 | 50/50 | 0.9995743034 | 0.9957221115 | 14/50 |

C1/C2 had an 8000-update ceiling; logs show only904/1049 executed. Partial evaluations follow the archived UID order and cannot supply full50 scores. No completed C2 update1000/update1049 evaluation is present.

## Protocol and provenance

- Flow: original GT vertex coordinates, actual8192-point XYZ+normal conditions, 512-dimensional Gaussian latent at t=0, Euler50 to t=1, inverse latent normalization, then frozen OwnAE-v2 decoder. Threshold `logit>0`; face candidates from the predicted edge graph; missing GT faces count as FN; no topology repair. This is topology generation given GT vertices.
- OwnAE-v2 reconstruction: frozen `NativeTopologyAE/B_v2_teacher_blocks`, step36220; cached mu or sampled posterior decoded under the same strict protocol. This is the project's own AE, independent of the teacher candidate and the separate Paper50 AE experiment.
- VAE identity: `e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15`. Checkpoint hashes are archived identities; weight files were not re-read.
- Shared config: DiT36 layers, width1536,12 heads; condition width2048,8 layers,16 heads,1024 tokens; latent512; lr1e-4,wd0.01,batch5,seed0,warmup100,clip1. C1 variant=`c1_fourier`; C2=`c2_teacher`.
- Original training Git commit: **not recorded**. Four run configs preserve runtime `imported_code` and `code_sha256`; all22 C0 files per segment and26 candidate files per run match the archived source. C0 `topology_flow.py` SHA256=`85b77fffe8ebd07cd26d1c4b6d442c811027b02d930485bfd2be6f75a34e65be`; C1/C2=`3d2f70e212d44ba0468b9fcc813a2f4ffa1d1db3aacb794711a5debdf7e0a732`.
- Archive commit: `f89ecbd4ff3adc4ed160adea235ad34c26e69f53` on `archive/topology-flow-20261008`, release `topology-flow-evidence-2026-10-08`. This is the publication commit, **not the training commit**.

## Evidence map

Download [the compact archive](topology-flow-light-evidence.tgz). It holds527 unchanged lightweight files, including complete training JSONL logs, configs, effective source code, all available per-UID metrics, summaries, and source/checkpoint identities.

| Inside archive | Contents |
|---|---|
| `snapshot/c0/run/`, `snapshot/candidates/{c1,c2}/run/` | Full update logs, config/code hashes, saved historical status |
| `snapshot/{initial,c0,candidates}/code/` | Archived effective Python code |
| `snapshot/c0/evaluations/{step500_seed0,step1000_seed0}/` | C0 summaries and50 per-UID metrics each |
| `snapshot/candidates/c1/{eval500,eval904}/` | C1 complete500 and partial904 evidence |
| `snapshot/candidates/c2/eval500/` | C2 partial500 evidence |
| `snapshot/initial/vae_baseline/{mu,posterior_seed0,posterior_seed1}/` | Frozen AE reconstruction summaries and150 per-UID metrics |
| `audit/`, `reports/` | Historical cohort, dependency, hash, and execution evidence |

Verification: all527 source files were SHA256 checked; all300 per-UID metric files in the six complete evaluations match their summary hashes; count sums, micro-F1, and strict counts were recomputed. The80 available partial-evaluation metric files are preserved with explicit incomplete denominators. The package excludes model/Adam tensors, NPZ/prediction arrays, rendered assets, and credentials. Current server state is outside this historical publication's scope.
