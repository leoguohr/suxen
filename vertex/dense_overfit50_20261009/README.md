# Dense 50-object Vertex overfit (2026-10-09)

Start with [RUN_SHEET.md](RUN_SHEET.md): acceptance criterion, the four runs, and step-by-step commands. [CODEX_PROMPT.md](CODEX_PROMPT.md) is the ready-to-paste instruction for the agent that runs the GPUs.

- `dense_train.py`: training (warm start from S0@34000, one GPU per run)
- `evaluate.py`: 100-tree acceptance evaluation (same protocol as S0)
- `packed.py`, `common.py`: packed forward pass, data, generation and scoring
- `tests/test_cpu.py`: CPU checks to run before any GPU job
- `vendor/`: byte-identical S0 model code; `data/`: the 50 objects, with SHA256 checks
