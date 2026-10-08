#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
/opt/conda/bin/python -u run_job.py >driver-console.log 2>&1
code=$?
printf '{"driver_exit":%s}\n' "$code" >driver_exit.json
exit "$code"
