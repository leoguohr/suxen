#!/usr/bin/env bash
set -euo pipefail
cd /guohaoran/nexus_fast_track/diagnostics/teacher_cad50_lr03_pair_20260921
export CUDA_VISIBLE_DEVICES=GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONDONTWRITEBYTECODE=1
exec 9>launcher.lock
flock -n 9
/opt/conda/bin/python -u gpu_guard.py
/opt/conda/bin/python -u pair_train.py --branch A_control_lr1
/opt/conda/bin/python -u gpu_guard.py
/opt/conda/bin/python -u pair_train.py --branch B_lr03
/opt/conda/bin/python -u audit_pair.py
