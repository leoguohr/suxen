#!/usr/bin/env bash
set -euo pipefail
cd /guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/opt/conda/bin/python -u resume_train.py
/opt/conda/bin/python -u audit_package.py
