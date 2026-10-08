#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
/opt/conda/bin/python -u run_pair.py >driver-console.log 2>&1
code=$?
printf '{"driver_exit":%s}\n' "$code" >driver_exit.json
exit "$code"
