#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# This command starts training only when explicitly executed by the user.
if [[ ! -f preflight/result.json ]]; then
  echo 'Missing successful preflight. Run: /usr/bin/python -u train.py --preflight' >&2
  exit 1
fi
mkdir -p run
log="run/console-$(date +%Y%m%d-%H%M%S).log"
nohup /usr/bin/python -u train.py "$@" > "$log" 2>&1 < /dev/null &
pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then cat "$log"; exit 1; fi
printf 'PID=%s\nConsole=%s/%s\n' "$pid" "$PWD" "$log"
printf '%s\n' "$pid" > run/last_pid.txt
