#!/usr/bin/env bash

# Nexus 服务器统一环境入口。该文件不包含密码。
export NEXUS_ROOT=/guohaoran/nexus_fast_track/mini_nexus
export NEXUS_ENV=/guohaoran/envs/nexus-algo

# 所有可能持续增长的缓存都放在持久化的 /guohaoran 下。
export PIP_CACHE_DIR=/guohaoran/cache/pip
export HF_HOME=/guohaoran/cache/huggingface
export TORCH_HOME=/guohaoran/cache/torch
export TORCH_EXTENSIONS_DIR=/guohaoran/cache/torch_extensions
export XDG_CACHE_HOME=/guohaoran/cache/xdg

# NVIDIA 镜像中的旧 ONNX protobuf 文件需要兼容解析器；不降低 protobuf 版本。
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 单卡任务不需要 UCX 虚拟文件系统；关闭它可避免容器 /tmp 的 inotify 警告。
export UCX_VFS_ENABLE=n

# 让 Python 能找到本项目，同时不把包安装进服务器 base 环境。
export PYTHONPATH="${NEXUS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${NEXUS_ENV}/bin/python" ]]; then
  echo "缺少独立环境：${NEXUS_ENV}" >&2
  echo "请先按照 SERVER_A100_GUIDE.md 的‘首次安装’执行。" >&2
  return 1 2>/dev/null || exit 1
fi

source "${NEXUS_ENV}/bin/activate"
cd "${NEXUS_ROOT}"

echo "Nexus root: ${NEXUS_ROOT}"
echo "Python: $(command -v python)"
