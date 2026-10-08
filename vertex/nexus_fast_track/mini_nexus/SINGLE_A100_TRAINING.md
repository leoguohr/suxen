# Nexus2K 单张 A100 训练说明

本目录实现的是依据 Nexus 论文公开内容编写的 independent reimplementation，
不是作者官方代码或权重。正式训练只能读取最终人工几何去重后的 manifest：

```text
/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k/reports/manual_geometry_dedup_v1/training_manifest_manual_dedup.csv
```

该 manifest 固定为 train 1060、val 118。不得改回旧的 1244 行 manifest，也不得
重新随机划分。

## 为什么先跑 benchmark

正式数据的顶点数从 8 到 18,307 不等。Vertex Stage、Topology AE 和 Topology
Flow 都包含随 token 数平方增长的 attention。长训练前必须在顶点数最小、50%、
90% 和最大分位样本上各做一次 forward/backward，记录显存和耗时。benchmark
通过只证明单步资源可承受，不证明生成质量。

启动命令：

```bash
bash /guohaoran/nexus_fast_track/mini_nexus/scripts/run_aistation_benchmark.sh
```

输出：

```text
/guohaoran/nexus_fast_track/mini_nexus/outputs/nexus2k_single_a100_benchmark_20260826/benchmark_report.json
```

## 与论文设置的明确差异

- 论文 Vertex DiT 和 Topology DiT 约 20 亿参数；当前单卡模型维度按 benchmark
  决定，明显更小。
- 论文边监督覆盖全部 vertex pairs；单卡版保留全部正边并采样等量负边，避免
  最大样本生成约 1.67 亿候选对。
- 当前只做 point-cloud-conditioned，不包含图像/DINO 分支。
- 当前按 Topology AE、Vertex、Topology Flow 分阶段在单张 A100 上训练。
- 3D RoPE 保留；self-attention 使用 PyTorch SDPA，让 A100 选择 Flash/
  memory-efficient kernel，避免显式保存完整 attention probability。

## 正式训练入口

正式入口为：

```text
scripts/train_nexus2k_single.py
```

它提供：固定 manifest hash、确定性样本顺序、梯度累积、bf16、周期验证、原子
checkpoint、断点续训、OOM/异常 UID 记录。遇到大样本 OOM 时停止并记录，不能
静默跳过样本。

## 已冻结的第一版单卡配置

A100 80GB 上的四分位 benchmark 和三个阶段真实训练 smoke 均通过后，第一版冻结为：

- hidden dim 512、12 层、8 heads、128 condition tokens；
- topology latent dim 128、spacetime dim 32；
- bf16、梯度累积 4、学习率 `1e-4`、weight decay `0.01`；
- Topology AE 10,000 optimizer steps；
- Vertex 15,000 optimizer steps；
- Topology Flow 15,000 optimizer steps；
- 每 250 步验证、每 1,000 步保留一个常规 checkpoint，固定使用 8 条 val 样本。

这些步数是为了先得到一版可评估结果的独立实现选择，不是论文公开的训练日程。
程序将每次 optimizer update 记作一个 step；梯度累积为 4，因此每个 step 读取
4 条样本。

三阶段 smoke 的 checkpoint 单文件约为 0.47/0.69/0.84 GB。每 250 步都永久保留
会产生约 100 GB 文件，因此验证频率与落盘频率有意分开；此外始终单独保留当前
`checkpoint-best.pt`。

## AIStation 任务卡只填一行命令

此 launcher 只保留 Vertex / Topology Flow；Topology AE 统一使用
`TOPOLOGY_AE_CODE_PATH.md` 中的末端 LayerNorm、输出仅中心化入口。
下方 Flow launcher 仍是历史 latent128 配置，不能直接接本轮 latent64 AE。
运行 Flow 前必须显式匹配已训练的 AE checkpoint 与 latent 维度；加载器会拒绝
旧的无末端 LayerNorm 权重及维度不匹配，而不会静默换模型。

```bash
bash /guohaoran/nexus_fast_track/mini_nexus/scripts/run_aistation_single_train.sh vertex
bash /guohaoran/nexus_fast_track/mini_nexus/scripts/run_aistation_single_train.sh topology-flow
```

任务意外中止后，原样重新提交同一行命令即可。脚本会读取该阶段的
`last_checkpoint.txt` 并严格校验 stage 和 manifest hash 后续训，不会静默从头开始。
正式输出统一放在：

```text
/guohaoran/nexus_fast_track/mini_nexus/outputs/nexus2k_scaled_h512_l12_v1/
```

AIStation 页面中的“日志路径”应另填到持久目录，例如：

```text
/guohaoran/nexus_fast_track/job_logs/nexus2k_scaled_h512_l12_v1_topology_ae
```

页面无需填写环境变量、数据挂载或模型挂载；脚本会加载项目自己的
`scripts/server_env.sh`，数据和输出本来就在 `/guohaoran` 持久目录中。
