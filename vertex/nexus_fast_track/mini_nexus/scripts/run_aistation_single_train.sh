#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/guohaoran/nexus_fast_track/mini_nexus
MANIFEST=/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/reports/manual_geometry_dedup_v1/training_manifest_manual_dedup.csv
RUN_ROOT=${PROJECT_ROOT}/outputs/nexus2k_scaled_h512_l12_v1

STAGE=${1:-}
case "${STAGE}" in
  vertex)
    STEPS=15000
    OUTPUT=${RUN_ROOT}/vertex
    PRECISION=bf16
    ;;
  topology-flow)
    STEPS=15000
    OUTPUT=${RUN_ROOT}/topology_flow
    PRECISION=bf16
    ;;
  *)
    echo "usage: $0 {vertex|topology-flow}" >&2
    exit 2
    ;;
esac

source "${PROJECT_ROOT}/scripts/server_env.sh"
mkdir -p "${OUTPUT}"

RESUME_ARGS=()
if [[ -s "${OUTPUT}/last_checkpoint.txt" ]]; then
  RESUME_CHECKPOINT=$(<"${OUTPUT}/last_checkpoint.txt")
  if [[ ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "last checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${RESUME_CHECKPOINT}")
fi

FLOW_ARGS=()
if [[ "${STAGE}" == "topology-flow" ]]; then
  TOPOLOGY_AE_CHECKPOINT=${RUN_ROOT}/topology_ae/checkpoint-best.pt
  if [[ ! -f "${TOPOLOGY_AE_CHECKPOINT}" ]]; then
    echo "topology-flow requires completed topology AE: ${TOPOLOGY_AE_CHECKPOINT}" >&2
    exit 1
  fi
  FLOW_ARGS=(--topology-ae-checkpoint "${TOPOLOGY_AE_CHECKPOINT}")
fi

exec python scripts/train_nexus2k_single.py \
  --stage "${STAGE}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT}" \
  --steps "${STEPS}" \
  --gradient-accumulation 4 \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --save-every 1000 \
  --validate-every 250 \
  --validation-samples 8 \
  --hidden-dim 512 \
  --condition-tokens 128 \
  --num-layers 12 \
  --num-heads 8 \
  --latent-dim 128 \
  --precision "${PRECISION}" \
  --seed 20260826 \
  "${RESUME_ARGS[@]}" \
  "${FLOW_ARGS[@]}"
