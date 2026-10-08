#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/opt/conda/bin/python -u train.py
/opt/conda/bin/python -u audit_package.py
