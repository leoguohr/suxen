# topology: T1 reviewed, T2 (gate 3) instructions

From Opus (topology chat), 2026-10-16. Based on `messages/20261016-completed-vertex-topology-results.md`
and its archive.

## T1 verdict: the topology gate is passed

The pass is on raw weights at update 4,000, with Euler 20 and the frozen OwnAE-v2 decoder.

| | VAE mu (ceiling) | Run A, fresh | Run B, S0 warm start |
|---|---|---|---|
| Face micro-F1, seed 0 / 1 | 0.99573 | 0.99425 / 0.99420 | 0.99504 / 0.99505 |
| Edge micro-F1 | 0.99958 | 0.99892 / 0.99889 | 0.99930 / 0.99927 |
| Face errors (FP + FN), seed 0 | 882 | 1,189 | 1,024 |
| Joint strict | 15/50 | 8 / 7 | 11 / 9 |
| Normalized latent RMSE | 0 | 0.080 | 0.072 |

Limits to record:
- **EMA lags.** EMA 0.999 fails at 0.91 (A) and 0.94 (B). The runs stopped at the first passing
  milestone at full learning rate, before any decay.
- **Gradient spike.** Run B had one spike, gradient norm 152 at u2995. It recovered.
- **Execution.** Updates 1,387 to 4,000 ran on the two-GPU data-parallel adapter: all 50 objects per
  update, gradients summed, at 4.4 s/update.

## T2

Commands are in `topology/flow_overfit50_20261015/CODEX_PROMPT_T2.md`, branch
`topology-overfit50-20261015` head `205c1b7`:
1. Stop the redundant quad_resume job.
2. Run the CPU gate.
3. Run the gate-3 self-check: the `gt_reference` mode must reproduce Run B's seed-0 gate-2 result,
   faces 102811/945/79.
4. Run gate 3 on Run B and Run A raw, using the vertex chat's best official evaluation (K2) and flow
   seeds 0 and 1.
5. Optional: a 1,000-update LR cooldown of Run B, then gate 3 again on its raw and EMA weights.
