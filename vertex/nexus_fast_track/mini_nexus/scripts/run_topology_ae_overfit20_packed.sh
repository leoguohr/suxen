#!/usr/bin/env bash
set -euo pipefail

# Full-FP32 entry for the final-LayerNorm, center-only 20-mesh experiment.
# Independent Edge/Face logit scales are calibrated to raw-interval RMS 1.0;
# this does not normalize embeddings. The Face area-squared factor stays 0.25.
#   Single GPU: NPROC_PER_NODE=1 OUTPUT_DIR=... bash this_file
#   Two GPUs:   NPROC_PER_NODE=2 OUTPUT_DIR=... bash this_file
# The output directory must be new; the Python trainer refuses to append to a run.

cd /guohaoran/nexus_fast_track/mini_nexus
source scripts/server_env.sh
export NVIDIA_TF32_OVERRIDE=0

: "${OUTPUT_DIR:?set OUTPUT_DIR to a new persistent output directory}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
STEPS="${STEPS:-20000}"

python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  scripts/train_topology_ae_overfit_packed.py \
  --manifest experiments/topology_ae_overfit20_manifest.csv \
  --negative-candidate-root /guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/derived/topology_negative_candidates_v2 \
  --output "${OUTPUT_DIR}" \
  --steps "${STEPS}" \
  --save-every 100 \
  --warmup-steps 200 \
  --precision fp32 \
  --pair-chunk-size 2000000 \
  --max-packed-meshes 1 \
  --max-padding-ratio 1.0 \
  --calibrate-logit-scales \
  --logit-rms-target 1.0 \
  --face-interval-factor 0.25
