#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
/opt/conda/bin/python -u train.py >console.log 2>&1
train_status=$?
verify_status=99
if [[ "$train_status" -eq 0 ]]; then
    /opt/conda/bin/python -u verify.py >verify.log 2>&1
    verify_status=$?
fi
printf '{"train_exit":%s,"verify_exit":%s}\n' "$train_status" "$verify_status" >runner_exit.json
[[ "$train_status" -eq 0 && "$verify_status" -eq 0 ]]
