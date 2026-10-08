# NEXUS D9 A/B/C/D 最终结果（2026-10-01）

四组均完成各 2000 次新增更新，累计 step 24000；已完成 800 条配对生成尝试。训练完成不代表生成验收通过：四组、两种采样器的完整树精确恢复均为 0/100。

## 先看哪些文件

- `cd/results/summary.json`：四组最终汇总与 D 双卡执行变更。
- `ab/runs/evaluation/{A,B}`、`cd/results/evaluation/{C,D}`：逐样本评价、逐层 occupancy、实际噪声、整数顶点、解码 XYZ、生成顺序及哈希。
- `ab/runs/{A,B}`、`cd/results/{C,D}`：训练日志、配置、checkpoint 身份及状态。
- `cd/code_and_probes/`：C/D 训练代码、固定探针、条件/模块更新诊断及双卡数值验证。
- `ab/code/`、`ab/source_runtime/`、`ab/prepared_d9/`：A/B 实际代码、运行依赖源码、固定输入条件与 GT。
- `LOCAL_ARRAY_VERIFICATION.json`、`verify_delivery_arrays.py`：本轮本地 CPU 复算；不是新一轮 GPU 重放。

## 固定实验范围

共同起点为迁移后的 D9 step 22000，恢复模型与 Adam；同一批 50 个已见对象，每组 2000 更新、每次 8 个 microbatch。D9、velocity MSE、BF16、clip 1。种子 97029000/97029001，每对象每采样器两次；每层 20 次网络调用；阈值 0.5、容量上限 4096，无 GT 点数/top-k 修补。这里是配对对照种子，不是未见终验集。实际优化器为 Adam（耦合 L2），不是 AdamW。

| 分支 | 调度 | LR / WD | Euler 完整树 | DPM 完整树 | Euler 第2层全对 | Euler 最终层全局 F1 | Euler / DPM 容量中止 |
|---|---|---|---|---|---|---|---|
| A | UID-major | 1e-05 / 0 | 0/100 | 0/100 | 47/100 | 0.07289030 | 1 / 1 |
| B | depth-major | 0.0001 / 0.01 | 0/100 | 0/100 | 0/100 | 0.00000000 | 0 / 0 |
| C | depth-major | 1e-05 / 0 | 0/100 | 0/100 | 32/100 | 0.04759827 | 0 / 0 |
| D | UID-major | 0.0001 / 0.01 | 0/100 | 0/100 | 0/100 | 0.00001780 | 41 / 42 |

A 在当前部分恢复指标上最好；C 没有改善 A；高 LR＋WD 的 B/D 均没有带来收益。D 第一层仍多数正确，但第二层完整占据集合已是 0/100，并伴随后续分支膨胀。以上不能把 LR、WD 分别认定为根因，也不能证明条件通路是唯一瓶颈。DPM 未改变完整树均未通过的结论。

## 逐层与首错层

以下每一项分母均为 100。全局集合 F1 统计所有 TP/FP/FN，不等于按物体平均 F1，也不等于 generated-parent bit accuracy。容量中止仍保留在总分母中，未生成的深度按零预测和全部 GT 漏检处理。

- A_euler：depth 1–9 全对数 `[84, 47, 22, 8, 3, 0, 0, 0, 0]`；首错层分布 `{"1": 16, "2": 37, "3": 25, "4": 14, "5": 5, "6": 3}`。
- A_dpm：depth 1–9 全对数 `[83, 48, 23, 7, 3, 0, 0, 0, 0]`；首错层分布 `{"1": 17, "2": 35, "3": 25, "4": 16, "5": 4, "6": 3}`。
- B_euler：depth 1–9 全对数 `[1, 0, 0, 0, 0, 0, 0, 0, 0]`；首错层分布 `{"1": 99, "2": 1}`。
- B_dpm：depth 1–9 全对数 `[1, 0, 0, 0, 0, 0, 0, 0, 0]`；首错层分布 `{"1": 99, "2": 1}`。
- C_euler：depth 1–9 全对数 `[80, 32, 13, 5, 1, 0, 0, 0, 0]`；首错层分布 `{"1": 20, "2": 48, "3": 19, "4": 8, "5": 4, "6": 1}`。
- C_dpm：depth 1–9 全对数 `[79, 30, 13, 5, 1, 0, 0, 0, 0]`；首错层分布 `{"1": 21, "2": 49, "3": 17, "4": 8, "5": 4, "6": 1}`。
- D_euler：depth 1–9 全对数 `[83, 0, 0, 0, 0, 0, 0, 0, 0]`；首错层分布 `{"1": 17, "2": 83}`。
- D_dpm：depth 1–9 全对数 `[79, 0, 0, 0, 0, 0, 0, 0, 0]`；首错层分布 `{"1": 21, "2": 79}`。

## 连续 XYZ 的限制

预测 XYZ 使用固定 D9 格心解码 `-1 + (q+0.5)*2/512`。原始含重复 GT 和去重浮点 GT 分开评价。沿用的 Hungarian 坐标匹配要求点数相同；点数不同或生成中止时明确标记不适用，不能记作零误差。800 条中仅 5 条对去重 GT 的匹配指标适用；它们不代表整体 XYZ 成绩。本轮本地核对了全部保存预测的解码公式与原生顺序，没有重算 Hungarian XYZ 指标。

## D 双卡恢复边界

D 在 update 1550 保存完整状态，从 update 1551 开始由两张 A100 共同完成剩余 450 次更新；有效 batch、事件/noise/time、优化配方、总更新数不变。单步算术及 Adam 状态核验通过；十步轨迹门禁未通过，未改动的单卡重复对照也超出原门槛。失败报告全部保留。这不构成逐位或完整轨迹一致性的证明，D 的运行后端变化是比较边界。详见 `cd/code_and_probes/DUAL_GPU_SWITCH.md`。

## 本地复核与权重

服务器原包 SHA256：`28418cfdf62f17229df95774b8632a77801aac5b094eca675c1af3cb55a880b5`，278565495 字节。下载后全部 9425 个原始成员哈希验证通过；本地复算了 800 条生成记录与 7105 份逐层数组，父格连续性、阈值、子格展开、整数集合、解码、顺序及原生统计一致。本交付增加本说明和 CPU 复核记录，因此交付 ZIP 的哈希与服务器原包不同。

不含大 `.pt` 权重或 Adam 张量；以下是服务器保留的完整 checkpoint 身份。没有这些文件，不能仅靠本包恢复训练。

- A：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/experiment/A/checkpoint-024000.pt`
  SHA256 `2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`
- B：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/experiment/B/checkpoint-024000.pt`
  SHA256 `183f703c7718156fa9c3204a209b55a641193ad414522bf68d1141d1ff27039d`
- C：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_cd_20261001/experiment/training/C/checkpoint-024000.pt`
  SHA256 `0b8b41fc2e41b8a24113797be5304025aab9f188b07f5a3a8abebd928e19e528`
- D：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_cd_20261001/experiment/training/D/checkpoint-024000.pt`
  SHA256 `76ac339b7a448da72dfc97cc7034b087e82d35cfbbabe857bfddd93f14de3bdf`

公共迁移起点：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/migration/checkpoint-022000-d9.pt`；SHA256 `5addf330bbe1a772a4a810181d5043f2fad5c7906a8c31d7c9dd7f9679471b83`。

目录中训练 status 的 `evaluation_pending` 以及打包前的 phase 快照可能是训练进程留下的旧字段；最终结果应以每组 evaluation/summary、四组 summary 与交付身份为准。已确认后台训练/评估控制进程退出，没有自动追加训练。

## 下一步

保留 A24000 作为工作基线；下一步先基于包内固定父格、固定噪声的条件切换探针及模块更新记录判断条件通路是否不足。当前结果不支持直接采用高 LR＋WD 组合，本轮没有启动结构修改或额外训练。
