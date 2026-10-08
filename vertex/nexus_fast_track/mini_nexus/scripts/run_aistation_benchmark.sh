#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/guohaoran/nexus_fast_track/mini_nexus
MANIFEST=/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/reports/manual_geometry_dedup_v1/training_manifest_manual_dedup.csv
OUTPUT=${PROJECT_ROOT}/outputs/nexus2k_single_a100_benchmark_20260826

source "${PROJECT_ROOT}/scripts/server_env.sh"

python scripts/benchmark_nexus2k_single.py \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --stages vertex topology-ae topology-flow \
  --quantiles 0.0 0.5 0.9 1.0 \
  --hidden-dim 128 \
  --condition-tokens 32 \
  --num-layers 2 \
  --num-heads 4 \
  --latent-dim 32 \
  --spacetime-dim 32 \
  --precision bf16 \
  --seed 20260826
