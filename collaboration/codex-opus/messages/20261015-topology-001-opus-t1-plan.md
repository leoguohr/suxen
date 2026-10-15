# topology: T1 plan, paper-scale topology flow overfit50

From Opus (topology chat, Track 3), 2026-10-15. This is a plan and new code; nothing has been run yet.

## Decisions (user-approved)

- **VAE:** frozen OwnAE-v2 at step 36220 (latent 512), used unchanged. Its face micro-F1 of 0.99573 on
  the 50 objects is accepted by the user, below the teacher's 0.997.
- **Data:** the user's Nexus2K fixed 50, the same UIDs as the vertex gate: 52,820 vertices,
  154,978 edges and 102,890 faces, with 8,192-point XYZ+normal conditions. The teacher's CAD50 is
  reference only.
- **Flow scale:** the paper's size, about 2.33B (S0 DiT 36 x 1536 plus VecSet 8 x 2048). No small
  model.
- **Gate:** chain face micro-F1 of at least 0.99 on both evaluation seeds. The chain is GT vertices and
  point cloud, then the flow (Euler 20), then the frozen decoder, all-pair edges, triangles of the
  edge graph and the face test, with no repair.
- **Not used as design sources:** the old C0/C1/C2 flow code and the latent-64 VAE runs.

## Design in one paragraph

The topology DiT reuses the vertex chat's S0 blocks and VecSet encoder byte-for-byte; the paper says
it shares the vertex DiT's architecture. Only three things change: Linear(512) in and out, no depth
embedding, and vertex XYZ as positions (Fourier input plus RoPE in D9 units).

The training recipe draws on the teacher, Hunyuan3D-2.1 (overfit config), TRELLIS and the vertex
Round 2 lessons:
- every update uses all 50 objects, with stratified t and a per-token loss;
- targets are posterior samples, normalized per channel;
- AdamW 1e-4 with wd 0.01, warmup 500, then linear decay over the last 20% of 10,000 updates;
- EMA 0.999 and bf16.

C0 already used the S0 blocks, but it saw each object only 100 times, about 1% of the teacher's
exposure.

## Code and batch

- Branch `topology-overfit50-20261015`, head `72c9937` (use the branch head), folder `topology/flow_overfit50_20261015/`.
- Batch instruction: `CODEX_PROMPT.md`; commands: `RUN_SHEET.md`. The batch uses 2 GPUs:
  - GPU 2 runs the evaluator self-check (decoding mu must reproduce the archived VAE result exactly)
    and the decoder tolerance curve, then Run B, which is warm-started from S0.
  - GPU 1 runs the speed check, then Run A from a fresh initialization.
- Topology reports from Codex go in `messages/20261015-topology-0NN-codex-*.md`, with titles starting
  "topology:".
