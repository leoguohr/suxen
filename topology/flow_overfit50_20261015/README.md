# Topology flow overfit on the fixed 50 objects, paper scale (2026-10-15)

Track 3 (topology). The frozen OwnAE-v2 VAE (step 36220, latent 512) is accepted as is; this folder trains
the topology flow at the paper's size and measures the chain: GT vertices and point cloud, then flow,
then the frozen decoder, then the strict edge/face metric.

- Design and reasoning, with every choice labelled by source: the topology analysis doc
  ("Topology flow overfit50: design analysis").
- Commands: [RUN_SHEET.md](RUN_SHEET.md). Paste-ready agent instruction: [CODEX_PROMPT.md](CODEX_PROMPT.md).

| File | What it does |
| --- | --- |
| `model.py` | Topology DiT = the vertex chat's S0 blocks (36 x 1536, 12 heads) + S0 VecSet encoder (8 x 2048), latent 512 in/out, XYZ Fourier input, XYZ RoPE in D9 units. Packed (varlen) forward for all 50 objects at once. |
| `train.py` | Dense updates (all 50 objects, stratified t, per-token loss), AdamW 1e-4, warmup 500, linear decay over the last 20%, EMA 0.999, bf16. Chain evaluations every 2,000 updates; resumable. `--init-vertex` = Run B warm start from S0; `--init-checkpoint` = cooldown from a finished topology checkpoint. |
| `cascade.py` | Gate 3: the vertex chat's generated D9 cells -> topology flow -> frozen decoder -> edges/faces, scored against GT with strict (same cell) and one-cell vertex matching. `--mode gt_reference` reproduces gate 2 through the same code path. Batch: [CODEX_PROMPT_T2.md](CODEX_PROMPT_T2.md). |
| `evaluate.py` | `--mode mu` (evaluator self-check against the archived VAE result), `--mode noise` (decoder tolerance), `--mode flow` (checkpoint, seeds x steps x samplers). |
| `metrics.py` | Frozen decoder in FP32 and the strict metric: all-pair edges, all triangles of the predicted edge graph, face logit > 0, no repair. |
| `data.py` | Loads and hash-checks the fixed-50 cache (your Nexus2K data; manifest f14f796c..., normalization a144fa69...). |
| `vendor/` | Byte-identical copies: S0 `vertex.py`, OwnAE-v2 `native_models.py`, vertex `samplers.py` (SHA256 in `SOURCES.json`). |
| `tests/test_cpu.py` | CPU gate: packed vs S0 block forward (values and gradients), FPS cache, triangles vs brute force, metric counts by hand, schedule, stratified t, chunking, exact resume, S0 weight transfer, grouping-independent sampling, tiny-VAE decode. |
| `fixtures/vae_mu_baseline.json` | The archived per-object VAE mu result that `evaluate.py --mode mu` must reproduce. |

Gate: chain face micro-F1 of at least 0.99 on both evaluation seeds (0 and 1), Euler 20, judged after
the LR decay; raw and EMA weights are both reported. The VAE itself (0.99573) was accepted by the user
below the teacher's 0.997.

T1 result (2026-10-16, raw weights, update 4,000, Euler 20): Run B 0.99504 / 0.99505, Run A 0.99425 / 0.99420
face micro-F1 on seeds 0 / 1, so the topology gate is passed. EMA weights (0.91-0.94) lagged; the runs stopped before LR decay.

T2 result (2026-10-16): the evaluator self-check reproduced gate 2 exactly (faces 102811/945/79). Run B was
cooled down for 1,000 updates (LR 1e-4 to 0, EMA 0.998 from raw; checkpoint SHA256 b71fb6f6...). On GT
vertices it reaches face micro-F1 0.99555 on both seeds (VAE mu ceiling 0.99573). Gate 3 (full pipeline: the
vertex chat's K2 u4500 generated cells, 2 vertex seeds x 50 objects, Euler 20), face micro-F1 on flow seeds 0 / 1:

| Weights | Strict face F1 | Exact-vertex trees (90) | Inexact-vertex trees (10) | Joint strict |
| --- | --- | --- | --- | --- |
| Run B u4000 raw | 0.99495 / 0.99498 | 0.99517 / 0.99520 | 0.99375 / 0.99375 | 23 / 23 of 100 |
| Run A u4000 raw | 0.99397 / 0.99397 | 0.99416 / 0.99422 | 0.99292 / 0.99261 | 16 / 15 of 100 |
| B cooldown raw | 0.99541 / 0.99534 | 0.99558 / 0.99555 | 0.99446 / 0.99421 | 27 / 25 of 100 |
| B cooldown EMA | 0.99540 / 0.99535 | 0.99558 / 0.99554 | 0.99443 / 0.99426 | 30 / 24 of 100 |

Gate 3 passes (teacher's bar about 0.99; his own pipeline 0.9911). About 93% of the remaining face errors are
the frozen VAE's own: decoding the VAE mu on the same trees gives 1,740 FP / 24 FN against the cooldown's
1,814 / 84. Evidence: `codex-opus-exchange`, `evidence/topology-t2-20261016/` and release `topology-t2-gate3-2026-10-16`.
