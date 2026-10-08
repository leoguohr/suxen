#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=GPU-208b1847-4ab0-f090-b8d4-f21ebd908b00
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -B launch_preflight.py
exec python -B code/run_candidates.py \
  --root "$PWD" \
  --cache /guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006/cache \
  --baseline /guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006/vae_baseline \
  --vae-checkpoint /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/run/checkpoints/vae-1000.pt \
  --gpu-uuid "$CUDA_VISIBLE_DEVICES"
