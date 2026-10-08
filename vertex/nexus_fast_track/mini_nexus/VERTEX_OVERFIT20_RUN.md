# Vertex 固定 20 条实验与监督

2026-09-08。用户授权在现有单张 A100 服务器运行固定约 20 条 overfit，监督
训练问题，并在结束后根据证据决定如何修改代码。没有授权改变已确认的模型尺寸。

**本轮已完成 1000 步，但未通过过拟合检查。** 20 条生成均发生节点爆增，
条件交换几乎不影响输出。完整分析、图表、代码观测改进及下一轮单变量建议见
[`VERTEX_OVERFIT20_RESULT_20260908.md`](VERTEX_OVERFIT20_RESULT_20260908.md)。
以下保留运行历史与取证方式，不代表训练仍在运行。
最终分析及修改决策完成后，监督自动化 `vertex-20` 已暂停；GPU 核验为 0 MiB、
0% 利用率，本任务的本机防睡眠辅助进程及 SSH 复用连接已清理。

## 当前运行

- SSH：`root@172.16.78.10`，端口 `31548`。复用本机
  `ssh -S /tmp/nexus-vertex-31548.sock -p 31548 root@172.16.78.10`。
  ControlMaster 的持久时间为 24 小时；本文不保存密码。
- 核验到的主机：`cc8o29a9p2r1u-0`；单卡 NVIDIA A100-SXM4-80GB。
- 本实验 PID：`683`，`/proc/683/stat` 的 starttime 字段为 `258939423`。
  操作进程前同时核对主机、该值和完整命令；不能只凭 PID。
- 远端实验根目录：
  `/guohaoran/nexus_fast_track/mini_nexus/outputs/vertex_overfit20_20260908_31548_v1`。
- 运行代码是根目录下的 `code/` 冻结副本；没有覆盖服务器原有训练源码。
  `source_sha256.json` 记录副本哈希，`pid.json` 记录实际启动命令。
- Python：`/guohaoran/envs/nexus-algo/bin/python`；实测 torch
  `2.3.0a0+6ddf5cf85e.nv24.04`，CUDA 12.4。
- 本地证据目录：
  `/Users/luthier/Documents/sophomore/nexus_fast_track/server_snapshots/vertex_overfit20_20260908_31548_v1`。
- 当前任务监督自动化：`vertex-20`，每 5 分钟检查。正常推进保持安静；新评估、
  异常、完成或需要用户处理的阻塞才通知。结束并完成分析后暂停该自动化。
- 本机已用 `caffeinate -i -t 14400` 临时防止空闲睡眠，PID=49490；身份记录在
  本地证据目录的 `local_supervision.json`。结束监督时核对命令后终止该辅助进程。
  Codex 应保持运行；服务器训练进程本身不依赖本机 SSH 会话存活。

## 数据位置和核验

清单为：
`/guohaoran/nexus_fast_track/mini_nexus/experiments/topology_ae_overfit20_manifest.csv`。

SHA-256：`21e1652f53348b48ad06d2a7ef16b146c69ae67d7b00e250025d740e19cceb06`。

数据根目录为
`/guohaoran/nexus_fast_track/data_2k_v1/runs/independent_reimplementation_v2_2k`：

- `stage2_outputs/<uid>/condition_point.npz`：8192 点和法线。
- `stage3_outputs/<uid>/mesh_quantized_training.npz`：量化后 mesh。
- `stage3_outputs/<uid>/octree_d9.npz`：逐层父节点与 8 子节点标签。
- `stage4_outputs/<uid>/topology.npz`：只为统一 loader 校验同一顶点身份；Vertex 不训练 Topology AE。

已在远端通过全部 20 条 loader/schema/几何一致性校验；均为 train/keep，
顶点数 8–4999。最大末层父节点数 4997，对象 `nexus_2k_000816`。
完整审计在远端 `data_audit.json`。

## 本轮配置

VecSet 8 层、2048 宽、1024 tokens；DiT 36 层、1536 宽；
共 2,332,430,344 参数，联合训练，D=9。单对象 micro-batch，BF16 autocast、
FP32 参数/梯度/AdamW 状态、activation checkpoint，梯度 clip=1。

先跑 1000 次更新；LR=1e-4，100 步线性 warmup，weight decay=0.01。
每个 180 步周期打乱全部 20×9 对象/深度组合，每组合恰好一次，
训练噪声与时间保持随机。第一步 VecSet 零梯度符合最终输出层零初始化，
从第二步应出现非零梯度。

0 步记录未训练基线；每 100 步保存 `checkpoint-last.pt` 并对全部对象/深度
用固定噪声评估 t=0.1、0.5 的速度 MSE 与去噪占用 Precision/Recall/F1/IoU。
去噪占用是 `x_t + (1-t)*v` 的阈值结果，不能等同于从噪声生成的顶点质量。
报告同时给出零速度基线。

最后增加 t=0.9、t=0.1 的跨对象条件交换对照，以及全部 20 条从根到叶的生成。
生成使用每层 20 步 Euler、阈值 0.5，不强制补子节点、不做 top-k 修复。
节点超过 10000 会明确报告 `capacity_abort`，不把部分树伪装成叶子输出。
该上限是本次显存保护界限；若学到合理分布但超过上限，应单独评估，
不能据此断言架构失败。空预测保留为空，回报零召回率。

## 已完成的运行前检查

- 本机全仓：96 passed，3 skipped。
- 服务器同环境专项：19 passed。
- 最大真实样本上完整模型连续 3 次 AdamW 更新通过；最后一步约 1.35 秒，
  峰值 allocated=35.19 GiB、reserved=37.67 GiB；VecSet 第二步开始非零梯度。
- 本轮正式实验重新设置种子、从头初始化，不沿用预检查更新过的权重。

## 监督与完成标准

每次检查 `status.json`、`train.jsonl`、`evaluations.jsonl`、`failure.json`、
`console.log`，并核对本实验进程和 GPU。`status.json` 区分加载、训练、保存、
评估、生成、完成、停止和失败。保存/评估不是训练卡死；先核验实际进展。

非有限 loss/gradient/输出会终止程序并写 failure；保留上一个有效 checkpoint。
若需要 SIGTERM，核验身份后只发给本 PID。程序会在更新或评估边界保存并退出；
不要 killall，也不要启动第二份。发生异常先收集证据，冻结已跑代码，
不要将架构缩小或更改数据筛选当作无声恢复策略。

结束后下载小型 JSON、JSONL、console、数据审计、代码哈希和 generation NPZ。
checkpoint 约 28 GB，保留服务器端，不在自动监督时直接下载。

判断依据：逐对象、逐深度的低 t 去噪指标相对初始基线是否持续改善；
正类召回与预测占用比例是否异常；完整生成的数量、叶子集合 IoU/F1、
归一化空间双向平均欧氏最近邻距离；正确条件是否优于交换条件；
目标/生成点云的可视化。不能用 loss 有限/下降作为通过标准。

如果生成差而去噪好，应优先检查累计误差、采样器和步数；Euler 并非论文
DPM-Solver。必要时对相同 checkpoint、相同噪声比较 20/40 步。
如果只有深层失败，检查层间采样权重和深层占用；如果条件交换几乎无影响，
检查条件梯度与条件通路。只有有证据的实现错误才局部修复、补测试；
参数/训练策略假设写成下一次单变量实验，不在正在运行的快照上修改。

完成最终结果解释及代码修改决策后暂停监督，保留当前任务。

## 启动后的首轮观察

已启动且完成 200 步；第 100 步 t=0.1 的去噪占用 macro F1 从初始 0.2916
变为 0.3251，速度 MSE 从 1.3412 变为 1.2517。t=0.5 的 F1 从 0.4878
变为 0.5058。改善较小，不能据此认定过拟合成功。数值目前有限，
第 200 步 VecSet 的裁剪后梯度范数为约 0.000104，需继续观察条件有效性。
这段是首轮观察，后续实际状态以远端 status、评估日志和最终报告为准。
