#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/guohaoran/nexus_fast_track/diagnostics/math00_twenty_mesh_training_preparation
LOG_ROOT=/guohaoran/nexus_fast_track/job_logs/topology_ae_math00_20mesh_20260914
cd "$TASK_ROOT"
mkdir -p "$LOG_ROOT" run
export PYTHONUNBUFFERED=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export UCX_VFS_ENABLE=n
log="$LOG_ROOT/console-$(date +%Y%m%d-%H%M%S)-$$.log"
exec > >(tee -a "$log") 2>&1
ln -sfn "$(basename "$log")" "$LOG_ROOT/console-latest.log"
printf 'Task directory: %s\nConsole log: %s\n' "$TASK_ROOT" "$log"
/usr/bin/python -u check_job_environment.py
if [[ "${1:-}" == '--check-only' ]]; then
    echo 'CHECK_ONLY_COMPLETE: no training started'
    exit 0
fi
# Foreground exec: the platform observes the training lifetime and exit status.
exec /usr/bin/python -u train.py "$@"
