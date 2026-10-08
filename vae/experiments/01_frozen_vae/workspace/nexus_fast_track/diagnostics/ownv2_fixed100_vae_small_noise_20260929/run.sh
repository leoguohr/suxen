#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-start}"
case "$MODE" in start|resume|evaluate) ;; *) exit 2 ;; esac
export CUDA_VISIBLE_DEVICES=GPU-3534263c-6584-9f34-9273-0ef6a7852fbe
exec /opt/conda/bin/python -u "$ROOT/code/train_vae.py" --config "$ROOT/experiment_config.json" --mode "$MODE"
