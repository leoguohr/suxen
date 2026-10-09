# Dense 50-object Vertex overfit (2026-10-09)

Start with [RUN_SHEET.md](RUN_SHEET.md): acceptance criterion, the four runs, and step-by-step commands. [CODEX_PROMPT.md](CODEX_PROMPT.md) is the ready-to-paste instruction for the agent that runs the GPUs.

- `dense_train.py`: training (warm start from S0@34000, one GPU per run)
- `evaluate.py`: 100-tree acceptance evaluation (same protocol as S0)
- `packed.py`, `common.py`: packed forward pass, data, generation and scoring
- `tests/test_cpu.py`: CPU checks to run before any GPU job
- `analyze_margins.py`: CPU-only check of how close the wrong bits were to the 0.5 threshold
- `analyze_cascade.py`: CPU-only split of each depth's errors into local vs inherited (deleted / spurious subtrees)
- Round 2 (unattended jobs, loss-weighting test): [ROUND2.md](ROUND2.md), [CODEX_PROMPT_ROUND2.md](CODEX_PROMPT_ROUND2.md); J3/J4 on extra GPUs: [CODEX_PROMPT_ROUND2B.md](CODEX_PROMPT_ROUND2B.md)
- `vendor/`: byte-identical S0 model code; `data/`: the 50 objects, with SHA256 checks
