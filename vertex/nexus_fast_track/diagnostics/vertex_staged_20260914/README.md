# Nexus Vertex 分阶段排错复核包（2026-09-14）

这是独立复现的点云条件 Vertex Diffusion 实验，包含实际输入、冻结源码、采样 oracle 检查、空间对齐核验和分阶段模型实验。GT oracle 不是模型生成成绩。

## 文件入口

- `RESULT.md`：本轮实际运行结果、停在哪一道门槛、尚未证实的部分。
- `code/`：训练时冻结的完整 Python 源码、测试和依赖清单。
- `code/scripts/check_vertex_sampler.py`：不依赖训练的 GT 解析速度检查。
- `code/scripts/train_vertex_staged.py`：A/B/C/D 顺序执行，未过门槛就停止。
- `data/`：原 20 个样本的点云、法向、归一化前后网格、八叉树和拓扑标签，以及原始质检/变换参数。
- `preprocessing_source/data_2k_v1/vendor_scripts/`：从当前服务器读取的预处理源码；文件哈希随包保存，不冒充作者源码。
- `oracle_saved_gt.json`：原保存的 20 份 GT 和一个非对称合成样本的检查；共 2268 次单层积分和 252 次完整八叉树生成检查。
- `alignment_audit.json`、`audit_alignment.py`：20 个实际样本的坐标空间、点云/法向采样、量化和父子标签核验。
- `run/`：本轮训练配置、逐步日志、固定输入回归数组、评估、门槛结果；按用户要求不包含模型权重或优化器状态。
- `previous_overfit20/`：9 月 8 日原 1000 步训练的日志和评估，供对照。
- `SHA256SUMS.txt`：本包文件校验清单；`data_source_sha256.json` 是下载时的原始数据/预处理源码校验。

## 约定与验收

路径为 `x_t=(1-t)*epsilon+t*GT`；训练目标为 `GT-epsilon`；oracle 为 `(GT-x_t)/(1-t)`，只在 `t<1` 调用。当前采样器是 Euler 基线，阈值 0.5，子格编号 `4*x_bit+2*y_bit+z_bit`，最终以 depth-9 整数坐标集合完全相等验收。没有强制保留子节点或按点数修复结果。

| 阶段 | 固定和变化 | 本轮预先设定的门槛 |
|---|---|---|
| A | 000105、depth 9、GT parents、点云、t=0.5、噪声固定 | 速度 MSE<=1e-4 且相对零速度基线 MSE<=1e-3；占据、坐标集合完全相同；连续 3 次评估通过 |
| B | 同物体同层；训练恢复随机 t 和噪声 | 20 个独立 t/噪声检查的 MSE<=0.01 且坐标完全正确；4 个纯噪声种子在 GT parents 上采样得到精确坐标 |
| C | 同物体，依次训练全部 9 层 | 4 个种子从根节点使用模型自身父层生成，最终坐标集合全部完全正确 |
| D | 000105 和 001208 两个不同形状，训练全部层 | 同一组 4 个噪声种子切换点云条件，每个物体都生成自己的精确坐标集合 |

每阶段步数上限为 1000/2000/3000/3000，每 25 步评估。达到预算仍未过门槛就停止，不自动扩到 10/20 个物体。门槛是本次排错的工程验收，不是论文指标。

保持完整模型：VecSet 1 cross + 7 self，宽 2048；DiT 36 层、宽 1536；共 2,332,430,344 参数。AdamW 的 weight_decay=0；目标学习率仍为 1e-4、最开始 100 步线性 warmup；BF16 autocast，FP32 参数/梯度/优化器，梯度裁剪 1。加载并固定实际 8192×6 点云，没有训练数据增强。A 之后继续同一模型和优化器，只有 t/噪声按阶段恢复随机。

`--smoke-model` 仅用于程序测试，任何带该标记的结果不能算完整模型成绩。

## 重跑

以下命令在本包根目录执行。`python` 需来自已安装 `code/requirements.txt` 所需依赖的环境。服务器实测版本见 `run/config.json`。

```bash
python prepare_manifest.py
python audit_alignment.py
python code/scripts/check_vertex_sampler.py --gt-directory previous_overfit20/generation-001000 --output oracle_rerun.json
PYTHONPATH=code PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest code/tests/test_vertex.py code/tests/test_vertex_overfit.py code/tests/test_vertex_staged.py -q
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python code/scripts/train_vertex_staged.py \
  --manifest data/manifest_runtime.csv --output rerun \
  --uids nexus_2k_000105 nexus_2k_001208 --last-stage D \
  --stage-steps 1000 2000 3000 3000 --eval-every 25
```

测试命令需要 `PYTHONPATH=code`（或先 `cd code` 再运行 pytest）。服务器旧 ONNX/protobuf 组合需要上面的环境变量；没有修改环境里的依赖版本或模型计算。`rerun` 必须是不存在的新目录，防止覆盖证据。

检查点只保留在服务器，包内不含大模型权重。检查点以 `torch.save` 保存，含模型、AdamW 状态、训练进度、Torch/CUDA RNG 和配置。当前训练 CLI 从头运行，没有实现断点恢复入口；检查点可用于独立恢复/评估，不能声称 CLI 已支持无缝续跑。

## 核验边界

空间核验确认点云与 GT 来自同一归一化网格，以及采样法向可按源面朝向重现，不证明所有原网格外观质量或法向朝外。预处理量化用 float64；直接改成 float32 重量化可能使边界点落入相邻格子，因此训练继续读取冻结整数 GT。即使 oracle 和 A 通过，也不代表随机去噪或多物体条件生成已经通过；以 `RESULT.md` 和每阶段 `result.json` 为准。
