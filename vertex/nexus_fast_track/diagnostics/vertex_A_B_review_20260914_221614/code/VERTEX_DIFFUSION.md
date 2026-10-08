# 点云条件 Vertex Diffusion

2026-09-08：按结构讨论实现的独立 Nexus Vertex 基线。模型代码在
`mini_nexus/vertex.py`，联合训练入口是 `mini_nexus/training.py` 中的
`VertexStageSystem`。本阶段学习八叉树占用速度场，位于 Topology AE 之前；
不依赖 Topology AE 权重。

后续已在单张 A100 80GB 上通过完整模型预检查，并启动固定 20 条、1000 步的
overfit。数据、实际显存、运行目录及监督方式见
[`VERTEX_OVERFIT20_RUN.md`](VERTEX_OVERFIT20_RUN.md)。
该轮现已结束，未通过过拟合检查；分析见
[`VERTEX_OVERFIT20_RESULT_20260908.md`](VERTEX_OVERFIT20_RESULT_20260908.md)。

## 结构和数据流

```text
点云 [B,8192,6]（同一归一化空间中的 XYZ + normal）
  → XYZ Fourier embedding + normal → 2048 维
  → FPS 1024 个点作为 query
  → 1 个 cross-attention block + 7 个 self-attention blocks
  → LayerNorm → 条件 token [B,1024,2048]

本层占用 Y [B,N,8] + 高斯噪声 ε + 时间 t
  → x_t = (1-t)ε + tY
  → Linear(x_t) + Fourier(parent_center) 投影 + depth embedding
  → 36 个 Vertex DiT blocks，宽 1536
  → LayerNorm → Linear → 速度 [B,N,8]
```

| 配置 | VecSet | Vertex DiT |
|---|---|---|
| 宽度 / heads | 2048 / 16 | 1536 / 12 |
| blocks | 1 cross + 7 self | 36，参数不共享 |
| FFN | 4 倍宽，GELU | 4 倍宽，GELU |
| 时间注入 | 无 | 各 block 独立 AdaLN-Zero |
| QK norm | 无 | self 和 cross 中逐 head RMSNorm |
| 位置 | XYZ Fourier 特征 | 父节点绝对坐标投影 + self Q/K 的 3D RoPE |

DiT 每个 block 依次执行：带时间 scale/shift/gate 的 self-attention、
普通 pre-LN cross-attention、带时间 scale/shift/gate 的 FFN。
条件 K/V 直接从 2048 投影到 1536。AdaLN 的 6H 输出层和最终速度输出层
初始化为零；cross-attention 没有零 gate，因此整个 block 并非初始恒等映射。
输出不经过 sigmoid。VecSet 与 DiT 联合训练，无额外 VAE 或 KL 项。

## 坐标、深度与损失约定

- `parent_codes` 是深度 `d-1` 的整数网格位置；`d` 是目标子节点深度，范围 `1..9`。
  父中心 `p = -1 + 2*(parent_codes+0.5)/2^(d-1)`。
  数据层的 `VertexLevelBatch.positions` 虽为 float32，但保存的是这些整数编码；
  Stage 将其转为 long 传给 DiT，模型内部计算中心，避免混用网格位置和物理中心。
- Fourier 特征为 raw XYZ 加 `sin/cos(π 2^k XYZ)`，`k=0..7`：
  父节点 51 维，带 normals 的条件点 54 维。
- RoPE 使用 `2^(max_depth-1)*p`，默认即 `256*p`，各层共享坐标原点和单位。
  每个 128 维 head 的 XYZ 各旋转 21 对维度，末尾 2 维不旋转。
  QK RMSNorm 在旋转之前计算，norm 与旋转角度采用 float32。
- 时间为 `[0,1]`，先以 `1000*t` 生成 256 维 sinusoidal embedding，
  再经 `Linear → SiLU → Linear`；乘 1000 是本实现的时间编码选择。
- 训练目标是 `Y-ε`。padding 不参与 attention 或 loss；每个对象先对有效父节点
  的 8 维误差取均值，再在 batch 内平均，避免大对象获得额外权重。
  loss 在 float32 中累加。不同对象不会互相 attention。

论文明确给出点云及法线、8192 输入点、1024 条件 token、8 层 2048 宽 VecSet、
约 2B 的 Vertex DiT、八叉树 8 维占用、3D RoPE 和深度 embedding。
**1+7 的计层方式、36×1536、heads、AdaLN 细节、QK norm、绝对坐标投影、
Fourier/时间频率及线性 flow 的具体写法是本次确定的复现选择，不能当作作者配置。**
原始核对记录位于仓库上一级 `research/vertex_diffusion_architecture_20260907/`。

## 运行入口

在本目录执行新模型和数据接口检查：

```bash
../.envs/nexus-algo/bin/python -m pytest -q tests/test_vertex.py tests/test_training.py
```

`scripts/train_nexus2k_single.py --stage vertex` 使用本模型，支持独立条件编码器
尺寸和 checkpoint/resume。此入口仍要求原冻结 Nexus2K CSV 的 train=1060、val=118；
其他数据规模尚需单独接入。以下是完整结构的训练命令模板，需替换数据 manifest
与输出目录；本次实现没有执行正式训练：

```bash
python scripts/train_nexus2k_single.py \
  --stage vertex --manifest /path/to/manifest.csv --output /path/to/vertex_run \
  --steps 1000 --hidden-dim 1536 --num-layers 36 --num-heads 12 \
  --condition-dim 2048 --condition-tokens 1024 --condition-heads 16 \
  --condition-layers 8 --activation-checkpointing --precision bf16
```

这是结构配置示例，1000 步并非论文训练计划；固定 20 条中最大样本的单卡
完整训练已实测通过，不能外推到任意更长序列或更大 batch。
`--activation-checkpointing` 用重算降低 block 激活内存，不改变权重结构。
小规模调试可将 DiT 改为 `--hidden-dim 96 --num-layers 2 --num-heads 4`，
条件编码器改为 `--condition-dim 128 --condition-heads 4 --condition-tokens 16`；
仍保留 8 个条件 blocks。Pilot、benchmark 和 loader smoke 默认是缩小配置。

新 checkpoint 标记 `vertex_architecture=point_cloud_vertex_v1`。
旧小网络权重不能续训新结构，入口会拒绝旧或结构不匹配的 Vertex checkpoint。
`models.py` 仍由旧 Topology Flow 与 toy 演示使用，它不再是 `VertexStageSystem`
的模型实现；toy 和 smoke 脚本的 Vertex 调用也接入新模型，历史 toy 结果不能
作为本次新 Vertex 的效果证据。

## 当前边界

实现覆盖模型结构、联合 flow loss、训练入口与兼容的新 Vertex 评估调用。
原 `evaluate_pilot.py` 的 Euler 采样及强制至少一个子节点、top-k 上限仍是
显式记录的演示策略。**论文每层 20 步 DPM-Solver 尚未接入**，不能把现有
Euler 结果称作论文采样复现。完整尺寸 GPU 前后向和更新已验证；20 条
过拟合第一轮未成功，尚未得到可用的真实数据生成结果。

## 本次检查结果

- 全仓测试：91 passed，3 skipped；其中新 Vertex 专项 13 项覆盖绝对坐标、
  条件/时间/深度有效性、FPS 重复点、padding 与对象隔离、联合梯度、RoPE、
  每对象 loss 和 activation checkpoint 的梯度一致性。
- 完整配置仅在 meta device 计数：VecSet 402,987,008；DiT 1,929,443,336；
  合计 2,332,430,344 参数。没有分配完整权重或执行完整模型前向。
- CPU 上用 `nexus_pilot_0032` 的真实 `[1,8192,6]` 条件运行缩小模型，
  完成 2 次更新、保存 checkpoint，再恢复到第 3 次更新；仅验证训练/续训接口。
  记录在 `outputs/vertex_smoke_20260908_2sqdaox8/`。
- 从该 checkpoint 加载后，9 层 teacher-forced 与逐层展开接口均运行通过。
  此项使用每层 2 步 Euler 和演示用 64 节点上限，只验证接口，不评判质量；
  详见上述目录中的 `interface_verification.json`。
- CPU BF16 autocast 的缩小模型前后向检查通过。随后 CUDA 完整模型在最大
  4997 父节点样本上实测约 35.19 GiB allocated；详见上述 GPU 运行记录。
- 评估脚本原本依赖 matplotlib；本次在 `.envs/nexus-algo` 中补装了它及其依赖，
  没有修改系统 Python 环境。
