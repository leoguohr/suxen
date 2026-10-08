#!/usr/bin/env bash
set -euo pipefail

# AIStation 训练任务入口：固定 20 个 mesh，双卡 DDP，端到端 FP32。
# 保留原定义的数据、rank 分配、每 mesh 等权 loss、Adam、lr 和 clip。
# external FlashAttention-varlen 不支持 FP32 QKV，因此本入口使用数学等价的
# PyTorch standard attention。为控制 FP32 显存，每个 rank 一次只计算一个 mesh。
PROJECT_ROOT=/guohaoran/nexus_fast_track/mini_nexus
MANIFEST="$PROJECT_ROOT/experiments/topology_ae_overfit20_manifest.csv"
NEGATIVE_ROOT=/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/derived/topology_negative_candidates_v2
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/outputs/topology_ae_overfit20_final_layernorm_center_only_face025_dual_fp32_v1}
EXPECTED_MANIFEST_SHA256=21e1652f53348b48ad06d2a7ef16b146c69ae67d7b00e250025d740e19cceb06

cd "$PROJECT_ROOT"
source scripts/server_env.sh
export NVIDIA_TF32_OVERRIDE=0

actual_manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
if [[ "$actual_manifest_sha256" != "$EXPECTED_MANIFEST_SHA256" ]]; then
    echo "manifest SHA-256 mismatch: $actual_manifest_sha256" >&2
    exit 1
fi
if [[ ! -d "$NEGATIVE_ROOT" ]]; then
    echo "negative-candidate directory is missing: $NEGATIVE_ROOT" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR/train.jsonl" || -e "$OUTPUT_DIR/last_checkpoint.txt" ]]; then
    echo "refusing to overwrite an existing training run: $OUTPUT_DIR" >&2
    exit 1
fi
python -c 'import torch; assert torch.cuda.device_count() == 2, torch.cuda.device_count()'

unset MASTER_ADDR MASTER_PORT
MASTER_ADDR=127.0.0.1
MASTER_PORT="$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
export MASTER_ADDR MASTER_PORT

echo "mode=aistation_training_task"
echo "world_size=2"
echo "precision=fp32"
echo "attention_backend=pytorch_multihead_attention"
echo "tf32=false"
echo "model_profile=topology_ae_final_layernorm_no_rms_v1"
echo "embedding_transform=per_mesh_center_only"
echo "logit_scale=fixed_calibrated_target_rms_1"
echo "face_interval_factor=0.25"
echo "warmup_steps=200"
echo "steps=50000"
echo "output=$OUTPUT_DIR"
echo "manifest_sha256=$actual_manifest_sha256"
echo "master_addr=$MASTER_ADDR"
echo "master_port=$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

exec python -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    scripts/train_topology_ae_overfit_packed.py \
    --manifest "$MANIFEST" \
    --negative-candidate-root "$NEGATIVE_ROOT" \
    --output "$OUTPUT_DIR" \
    --expected-samples 20 \
    --steps 50000 \
    --save-every 500 \
    --learning-rate 1e-4 \
    --warmup-steps 200 \
    --seed 20260901 \
    --precision fp32 \
    --pair-chunk-size 2000000 \
    --max-packed-meshes 1 \
    --max-padding-ratio 1.0 \
    --calibrate-logit-scales \
    --logit-rms-target 1.0 \
    --face-interval-factor 0.25
