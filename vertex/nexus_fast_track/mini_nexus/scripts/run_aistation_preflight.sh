#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/guohaoran/nexus_fast_track/mini_nexus
MANIFEST=/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/reports/manual_geometry_dedup_v1/training_manifest_manual_dedup.csv
OUTPUT_DIR=/guohaoran/nexus_fast_track/mini_nexus/outputs/aistation_preflight_20260826

if [[ ! -f "${PROJECT_ROOT}/scripts/server_env.sh" ]]; then
  echo "Missing project environment script: ${PROJECT_ROOT}/scripts/server_env.sh" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing final manifest: ${MANIFEST}" >&2
  exit 3
fi

source "${PROJECT_ROOT}/scripts/server_env.sh"
mkdir -p "${OUTPUT_DIR}"

nvidia-smi
python -c 'import torch; print("torch=", torch.__version__); print("cuda=", torch.cuda.is_available()); print("gpu=", torch.cuda.get_device_name(0))'

python scripts/smoke_nexus2k_loader.py \
  --manifest "${MANIFEST}" \
  --train-uid nexus_2k_000002 \
  --val-uid nexus_2k_001440 \
  --device cuda \
  --output "${OUTPUT_DIR}"
