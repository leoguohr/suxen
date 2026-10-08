#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"
exec 9>pair.lock
flock -n 9 || exit 1
if [[ -e A_hold/checkpoint-update0000.pt || -e B_lr03/checkpoint-update0000.pt ]]; then
    echo 'Existing paired experiment: refusing to overwrite.' >&2
    exit 1
fi
/opt/conda/bin/python -u A_hold/train.py >A_hold/console.log 2>&1
a_status=$?
/opt/conda/bin/python -u B_lr03/train.py >B_lr03/console.log 2>&1
b_status=$?
printf '{"A_exit":%s,"B_exit":%s}\n' "$a_status" "$b_status" > runner_exit.json
[[ "$a_status" -eq 0 && "$b_status" -eq 0 ]]
