#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
case "${1:-}" in fresh|resume|evaluate) ;; *) echo 'Usage: bash launch.sh fresh|resume|evaluate' >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="$(/opt/conda/bin/python -c 'import json; print(json.load(open("run_config.json"))["gpu_uuid"])')"
exec /opt/conda/bin/python -u train_fixed100.py --config run_config.json --mode "$1"
