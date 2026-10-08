# Nexus 数据和 A100 服务器交接（2026-09-17）

本文件不含登录密码。路径已在当前服务器核验；临时目录只代表本实例，换实例后需要重新准备。下方进度是交接时快照，新任务应先读服务器实时状态。

## 1. 连接服务器和运行环境

```bash
ssh root@172.16.78.10 -p 31548
```

密码使用用户最近提供的有效密码，新任务中单独提供；不要写入脚本、日志或交接包。本机已有可复用的 SSH 控制连接时，也可以：

```bash
ssh -o BatchMode=yes -o ControlPath=/tmp/nexus-d4-31548.sock -p 31548 root@172.16.78.10 hostname
```

控制连接会过期，失败后用普通 SSH 登录。当前 hostname 为 `8mddvkd6v007p-0`，GPU 为一张 `NVIDIA A100-SXM4-80GB`。核对实际机器使用 `hostname` 和 `nvidia-smi`。

Python 显式使用 `/guohaoran/envs/nexus-algo/bin/python`，实际解析到 `/opt/conda/bin/python3.10`。当前训练使用：

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd /tmp/vertex_d4_fps_speedup_20260917/code
/guohaoran/envs/nexus-algo/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

当前服务器 pytest 入口存在已知 pluggy 导入不兼容，不要把它误认为模型失败，也不要为此直接升级正在训练的环境。本地 CPU 测试18项通过，服务器已直接执行同一等价性测试函数，CUDA/BF16两个重计算设置均通过。

## 2. 当前 D4 实际训练用的四物体数据

manifest 是训练入口读取路径的依据：

```text
/tmp/vertex_d4_fps_speedup_20260917/manifest_server.csv
```

**该 manifest 中的数据实际仍位于之前的恢复目录：**

```text
/tmp/vertex_d4_resume1200_20260917/data/
```

| UID | 量化去重顶点数 |
|---|---:|
| nexus_2k_000105 | 8 |
| nexus_2k_000195 | 52 |
| nexus_2k_001045 | 54 |
| nexus_2k_001885 | 176 |

每个 UID 的文件布局为：

```text
stage2_outputs/<UID>/condition_point.npz
stage3_outputs/<UID>/mesh_quantized_training.npz
stage3_outputs/<UID>/octree_d9.npz
stage4_outputs/<UID>/topology.npz
```

- `condition_point.npz`：`points [8192,3]` 和 `normals [8192,3]`，拼接后为条件输入 `[8192,6]`。
- `mesh_quantized_training.npz`：`vertices_norm`、`faces`、`quantized_vertices`，用于 GT 和核验。
- `octree_d9.npz`：depth1..9 的 `depth_<d>_parent_xyz` 与 `depth_<d>_target [N,8]`，以及叶子坐标等。
- `topology.npz`：edge/face/incidence 等；当前通用 loader 会读取和核验，Vertex 训练并不因此训练 Topology AE。

选择记录和每个输入的 SHA 在 `/tmp/vertex_d4_fps_speedup_20260917/selection.json`。这次交接重新核对了全部16份数据文件 SHA。

本地四物体原始副本：

```text
/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d4_20260916/data/
/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d4_20260916/manifest_local.csv
```

本实例恢复副本也位于本地 `diagnostics/vertex_d4_resume1200_20260917/data/`。持久服务器备份在：

```text
/guohaoran/tmp/vertex_d4_20260916/provenance.tar.gz
```

已核验该归档包含四物体全部16份 NPZ。最新 FPS 版本的 provenance 主要保存代码与清单，不要误以为它包含这些数据。若重建目录，需把 manifest 的绝对路径一起改成新位置，并核对 selection 中的数据 SHA；无需重新随机采样点云。

## 3. 更大的筛选后数据集（不是当前 D4 的训练范围）

服务器预处理根目录：

```text
/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/
```

人工筛选与几何去重后的清单：

```text
/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/reports/manual_geometry_dedup_v1/training_manifest_manual_dedup.csv
```

当前1178条：train1060、val118。交接时检查了清单引用的4712个数据文件，均存在。应按清单的 `split` 选数据，不能把 val 默认混入训练。原始预处理清单还在 `reports/training_manifest.csv`，不能自动当作最终筛选清单。

清单已经包含服务器绝对路径，数据仍按 stage2/3/4 输出组织。原始资产根目录 `/guohaoran/nexus_fast_track/data_2k_v1/raw/`，训练当前 Vertex 阶段直接使用预处理 NPZ 即可。

对应本地筛选清单在：

```text
/Users/luthier/Documents/sophomore/nexus_fast_track/data_2k_v1/final_evidence_server/manual_geometry_dedup_v1/training_manifest_manual_dedup.csv
```

注意本地的 `final_evidence_server` 路径不存在于服务器，服务器应使用上面 `runs/.../reports/...` 路径。

## 4. 当前实际执行的代码、日志与权重

本地开发仓库：

```text
/Users/luthier/Documents/sophomore/nexus_fast_track/mini_nexus
```

服务器当前执行的是冻结代码副本：

```text
/tmp/vertex_d4_fps_speedup_20260917/code/
/tmp/vertex_d4_fps_speedup_20260917/code/scripts/train_vertex_d4.py
```

不要默认 `/guohaoran/nexus_fast_track/mini_nexus` 与运行版本相同。入口SHA：`7db542a26e81f6603baffcc9c59c791f08092f972624043e0973c47d7a0c775d`。

当前PID1009，`/proc/1009/stat` 的 start_ticks为337325829；完整启动身份和参数：

```text
/guohaoran/tmp/vertex_d4_fps_speedup_launch_20260917.json
```

运行目录与持久目录：

```text
/tmp/vertex_d4_fps_speedup_20260917/run/
/guohaoran/tmp/vertex_d4_fps_speedup_20260917/
```

关键文件：`config.json`、`status.json`、`train.jsonl`、`checkpoint_verified.json`、`checkpoint-last.pt`、`evaluation-XXXXXX.json`及同名目录中的NPZ、`evaluation_ledger.jsonl`、`resume_verification.json`、`recovery_verification.json`。

控制台日志：

```text
/guohaoran/tmp/vertex_d4_fps_speedup_console_20260917.log
```

检查实时状态的只读命令：

```bash
nvidia-smi
cat /guohaoran/tmp/vertex_d4_fps_speedup_20260917/status.json
cat /guohaoran/tmp/vertex_d4_fps_speedup_20260917/checkpoint_verified.json
tail -n 2 /guohaoran/tmp/vertex_d4_fps_speedup_20260917/train.jsonl
tail -n 5 /guohaoran/tmp/vertex_d4_fps_speedup_console_20260917.log
```

交接时训练仍在运行，不需要再次执行启动命令。`launch.json` 是已执行命令的记录，不是应无条件重跑的命令。新任务若做独立训练，当前唯一GPU已被D4使用，先确认任务安排。

如确需恢复：从最新 `checkpoint_verified.json` 读取对应D4 update、累计step及SHA；恢复模型、Adam和CPU/CUDA RNG，不重置优化器或warmup。当前脚本支持 `--resume-update`、`--resume-evaluation`、`--resume-training-log`，恢复前后同一组开发数组必须一致；输出与持久目录使用新目录。`--updates 7200` 是D4总终点，不是在恢复点上再加7200步。

## 5. 当前协议和衔接历史

D4从D2累计7400开始，目标D4 update7200，即最终累计14600。保持lr1e-5、weight_decay0、clip1、8次梯度累积、BF16，四物体固定点云/法向；每micro独立随机时间与噪声，覆盖每物体depth1..9，VecSet与DiT联合训练。

最新加速只预先计算四份固定点云的FPS索引，没有缓存可学习VecSet特征。CPU及CUDA/BF16小模型损失/梯度/Adam更新一致，完整模型恢复216份数组一致。短窗口观测平均4.30→3.65秒/step，长期速度仍看日志。

每400步开发评估，种子29000000..29000003。训练结束冻结后，终验种子30000000..30000015：64条完整树，主要目标64/64整数坐标集合正确；另报96组共同父格条件切换。20步Euler、阈值0.5，不加top-k或强制非空规则。不得因loss下降宣布完整生成通过，也不自动扩大D10/20。

有效日志衔接：

1. `/guohaoran/tmp/vertex_d4_20260916/train.jsonl`：保留update1..1200；原1201..1241另存为重放前记录。
2. `/guohaoran/tmp/vertex_d4_resume1200_20260917/train.jsonl`：update1201..2933。
3. `/guohaoran/tmp/vertex_d4_fps_speedup_20260917/train.jsonl`：从update2934继续。

本任务已有 `vertex-d4` 定期监督。新任务接手时先明确是否接管，避免两个任务同时停止、恢复或修改同一训练进程。

详细监督规则：

```text
/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d4_fps_speedup_20260917/SUPERVISION.md
```

本地当前证据目录为本文件所在目录；权重留在服务器，用户不需要下载大权重。换实例后 `/tmp` 可能丢失，先检查 `/guohaoran` 持久数据、检查点和本地归档，不能把旧status视为新实例仍在运行的证据。

## 6. 本次实时核验快照

```json
{
  "checked_at": "2026-09-17T13:30:16.197092+08:00",
  "hostname": "8mddvkd6v007p-0",
  "status": {
    "state": "training",
    "update": 3296,
    "training_complete": false
  },
  "checkpoint": {
    "path": "/guohaoran/tmp/vertex_d4_fps_speedup_20260917/checkpoint-last.pt",
    "bytes": 27990403994,
    "sha256": "94f9495f9b8698e04bb6dd73c8b12f6ad85562e883789032717c1ae877167ad4",
    "verified_readback": true,
    "phase": "D4",
    "d4_update": 3200,
    "cumulative_step": 10600
  },
  "last_eval": {
    "update": 3200,
    "full_trees": 11,
    "per_object": [
      4,
      4,
      3,
      0
    ],
    "common_pairs": 24
  },
  "full_manifest_count": 1178,
  "splits": {
    "train": 1060,
    "val": 118
  },
  "all_manifest_files_checked": 4712,
  "missing_files": []
}
```

完整数据路径与数组形状、SHA见同目录 `server_handoff_inventory.json`。
