#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
/opt/conda/bin/python -u train_continue.py >train-console.log 2>&1
code=$?
printf '{"train_exit":%s}\n' "$code" >runner_exit.json
/opt/conda/bin/python package_results.py >package-console.log 2>&1
exit "$code"
