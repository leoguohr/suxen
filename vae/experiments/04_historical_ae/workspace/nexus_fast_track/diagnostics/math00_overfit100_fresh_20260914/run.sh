#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
/opt/conda/bin/python -u train.py >train-console.log 2>&1
code=$?
printf '{"train_exit":%s}\n' "$code" >runner_exit.json
exit "$code"
