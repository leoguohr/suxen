#!/usr/bin/env bash
set -euo pipefail

# Generic selective-precision entry: all Topology AE computation is FP32 except
# the BF16 kernel I/O required by external FlashAttention-varlen.
cd /guohaoran/nexus_fast_track/mini_nexus
source scripts/server_env.sh
export NVIDIA_TF32_OVERRIDE=0

: "${OUTPUT_DIR:?set OUTPUT_DIR to a new persistent output directory}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
STEPS="${STEPS:-1000}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_topology_ae_fp32_flash_bf16_experiment.py \
  --manifest experiments/topology_ae_overfit20_manifest.csv \
  --negative-candidate-root /guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/derived/topology_negative_candidates_v2 \
  --output "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --save-every 100 \
  --warmup-steps 200 \
  --precision fp32_flash_bf16 \
  --pair-chunk-size 2000000 \
  --max-packed-meshes 1 \
  --max-padding-ratio 1.0 \
  --calibrate-logit-scales \
  --logit-rms-target 1.0 \
  --face-interval-factor 0.25
