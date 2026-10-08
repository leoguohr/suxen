# 本轮命令与执行记录

所有写入仅在本目录。以下 `ROOT` 是当前独立目录，`PY` 是现有本地 Python。没有运行任何训练命令。

```sh
ROOT=/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/point_native_xyz_gate_20260925
PY=/Users/luthier/Documents/sophomore/nexus_fast_track/.envs/nexus-algo/bin/python
cd "$ROOT"
```

已执行（各脚本日志在 `evidence/`）：

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/representation_check.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/test_interface.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/teacher_gate.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/historical_d2.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/native_export.py --help
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/evaluate_exports.py \
  --generated outputs/B_historical_D2 --gt-root inputs/own_float \
  --historical-run ../Vertex_D2_final3600_20260916/snapshots/vertex_d2_resume3000_20260916/run \
  --out outputs/B_historical_D2/recheck_with_reference.json
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/local_deps" "$PY" scripts/finalize_report.py
```

`test_interface.py` 与真正生成入口使用新目录保护，重复运行应选择新输出目录或保留原结果再改测试输出位置。老师脚本此轮运行了一个 CPU seed（50个样本），另一seed复用历史且校验源文件及输入哈希；未触碰拓扑/VAE权重。

已执行的服务器检查（CPU；凭据仅交互输入，不写入文件）：

```sh
ssh -o BatchMode=yes -o ControlPath=/tmp/nexus-native-32483.sock -p 32483 root@172.16.78.10 \
 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 /opt/conda/bin/python - /guohaoran/tmp/vertex_d2_resume3000_20260916/checkpoint-last.pt' \
 < scripts/checkpoint_structure.py > evidence/checkpoint_structure.json
```

首次复用的 SSH 控制连接已过期，返回 Permission denied；交互重新认证后上述检查成功。首次完整检查结果为 `d2_checkpoint_audit.json`，再次哈希/结构检查为 `evidence/checkpoint_structure.json`。首次检查约42秒，再次40秒。只读取权重及状态，没有实例化大模型或更新参数。

实际下载并保存的是4个 UID 的 stage1/2/3 浮点数据/变换/已有标签，源数据根：
`/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k`。
`inputs/own_float/source.tar` 是本轮读取数据的传输归档；解包后的文件及哈希清单随包保留。教师资产来源路径见 `teacher_checkpoint_audit.json` 与 `outputs/representation.json`。

本机 scipy 原先缺失，安装到了独立 `local_deps` 目录，未修改共享环境；安装日志是 `evidence/dependency_install.log`。ZIP不打包该平台依赖，可用兼容环境或按日志安装。

## GPU 方案：尚未执行

唯一请求：确认 GPU0 UUID `GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d` 未分配V2，最多30 GPU分钟，仅冻结重放。未收到本轮授权；不要把空闲状态当许可。若获授权，先复核设备占用，再将本独立代码/两份条件复制到新服务器独立目录，重写 inference manifest 的 condition_file 为实际部署路径（保持条件文件SHA），不复制或修改V2。

部署后命令模板（非执行记录）：

```sh
CUDA_VISIBLE_DEVICES=GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d \
 timeout --signal=TERM --kill-after=10s 1800s /opt/conda/bin/python scripts/native_export.py \
 --checkpoint /guohaoran/tmp/vertex_d2_resume3000_20260916/checkpoint-last.pt \
 --expected-sha 2384b430fa52d5012cd793fc2781c79791814f6a4b2e8ac120a711fb91112e1b \
 --inference-manifest inputs/d2_inference_manifest.server.json \
 --output outputs/B_fresh_D2 --device cuda --gpu-budget-seconds 1800 \
 --steps 20 --seed-start 28000000 --seed-count 16
```

推理不加载GT数据文件，预测点数来自9层生成结果。检查点保存的配置含历史训练元数据，入口仅使用模型构造/精度和UID顺序，不使用其中GT坐标、目标点数或拓扑。生成结束后下载预测到本目录，运行同一独立评价器，`--generated` 改为新目录，`--historical-run` 保留原D2目录。噪声逐元素对照和误差明细将在新评估JSON中记录。
