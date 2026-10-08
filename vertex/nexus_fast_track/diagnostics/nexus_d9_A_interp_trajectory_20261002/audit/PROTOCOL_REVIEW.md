# A24000：训练式插值与已有 Euler 轨迹的冻结诊断审核

本审核只读已有数据和冻结训练/采样代码，并执行 CPU 数组复算；未连接 GPU、未训练、未编辑执行脚本。结论是协议可比较，但两边回答的是诊断问题，不是两种自由生成方法的成绩比较。

## 固定比较对象

- 同一个 A24000 checkpoint：`2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`。原 50 对象、种子 97029000 / 97029001、depth 6–9；共 400 个层样本，每个使用 t=0,0.05,...,0.95 的 20 个网格点。
- parents、condition、初始 epsilon、target y 均来自同一份已保存 GT-parent 诊断。epsilon 必须取该层实际保存的 initial_noise，不重新按 seed 和新 shape 生成。原有 common/GT-only/unknown 标记保留；本次两条路径对同一层全部使用同一个 epsilon。
- 插值路径：直接复用冻结 `flow_matching_batch(y, noise=epsilon, time=t)` 得到 `x_interp=(1-t)*epsilon+t*y`；目标 velocity 是 `y-epsilon`。这是训练式输入构造。
- 轨迹路径：使用已保存的实际 `x_states[k]` 和 `velocities[k]`；不重新自由展开、不在中途用 y 修正轨迹。两边均计算投影 `yhat=x+(1-t)*v`，不裁剪、不 sigmoid，阈值统一为 `>=0.5`。
- t 使用存档的 FP32 值，`1-t` 也按 FP32 计算；两边保持相同计算顺序。网络保持原 BF16 autocast，状态和投影保持 FP32。指标归约精度须记录，并保持两边相同。
- yhat 是当前点的一步终点投影，不等于重新从当前点跑完剩余 ODE。只有在最后一个时刻，它才与原最后一步 Euler 具有直接对应关系。

## 必须保留的解释边界

插值输入在 t>0 时已经含有 `t*y`，随 t 增大获得越来越多正确答案。因此插值晚时刻更好是可预期的，不能称作生成通关，不能单凭它证明训练充分、模型容量足够或 Euler 步数不足。

在精确算术下，插值侧有：

`yhat_interp-y=(1-t)*(v_interp-(y-epsilon))`。

因此 endpoint MSE 自带 `(1-t)^2` 缩小。为避免把这个几何因子当作模型进步，必须并列报告插值侧的真实训练目标 velocity MSE，并同样拆分占据和空格；它是辅助解释，不替代 endpoint 指标。FP32 下该等式存在舍入误差，不应据此要求逐位相等。

轨迹侧已离开指定的线性桥，`v-(y-epsilon)` 不能称作它在训练分布上的 loss。若报告 `(y-x)/(1-t)-v`，必须标明这是到指定 y 的投影残差，不能与训练 loss 混称；本次主曲线无需增加这一项。

GT parent 本身已提供正确结构条件；原对象和种子也不是未见终验。只能定位“给定正确 parents 时，训练式输入与实际采样路径行为是否分开”，不能直接归因为某一种训练、容量或积分问题。

## 指标和聚合方案

每个 UID × seed × depth × t × 路径保存：

1. `occupied_endpoint_mse`：仅 y=1 的 `(yhat-y)^2` 均值。
2. `empty_endpoint_mse`：仅 y=0 的均值。全 bit MSE可以附带，但不能代替上述两项。
3. TP / FP / FN、F1，以及该完整层的 exact（FP=FN=0）。类别没有元素时 MSE记 null，同时保存分母，不能补 0。
4. 插值侧额外保存同样按 y=1/y=0 拆分的真实 velocity MSE。

四个 depth 分开画完整 20 点曲线，不先混成一个总分。主 MSE 与 F1 曲线先在每个 UID 内平均两个 seed，再对 50 个 UID 等权汇总；这也使大网格不悄悄主导训练式解释。对每个 UID 先形成两路径的配对差，再报告 mean、median、P25/P75 和正差比例。

同时单独列 pooled TP/FP/FN、由总计数得到的 micro F1、exact/100；明确它们与 mesh-equal F1不同。FP/FN可附 mesh-equal FPR=FP/空格数、FNR=FN/占据数，避免把不同大小网格的 raw count 当成可直接比较的错误率。不要把 100 个 seed 样本或所有 child bits 当成独立对象做置信区间。

## “最早明显分开时间”的透明描述

不设置事后挑选的绝对差值门槛，也不把开始分开写成通过/失败标准。建议在看到新插值结果前固定以下简单描述约定：

- MSE 差值定义为 `sample-interp`，F1 差值定义为 `interp-sample`，正值都表示插值更好。每个 UID 的两个 seed 先平均。
- 对每层、每项主指标分别寻找第一对相邻网格点（从 t=0.05 起），两点都满足“50 个 UID 的配对差中位数 > 0，正差 UID 比例 > 50%”。称为 **首次连续方向一致区间** `[t,t+0.05]`；找不到就报告未出现。
- 同时给前一个点和该区间两端的 median、P25/P75、正差比例及原始量纲的差值。之后若反转，也必须明确列出。exact/100 和 FP/FN 曲线用于解释，不据此另挑一个更早的时间。
- 该标记只说明差异方向开始连续一致，**不等于效果已经明显，也不是统计显著性**。正文若使用“明显分开”，必须紧跟实际差值大小与对象覆盖率；若值很小就如实说“方向已分开，但幅度很小”。
- 占据 MSE、空格 MSE、F1的起点可能不同，应全部展示，不能择取最早一项作为统一结论。采样网格只支持时间区间描述，不能声称定位到两点之间的精确分叉时刻。

## 已实际执行的 CPU 核验

`review_existing_sample_side.py` 已用本地 numpy 执行，输出 `sample_side_cpu_evidence.json`。该 JSON包含原结果 SHA、实际审核代码 SHA、Python/numpy 版本、400 个层样本的检查结果，以及四层 × 20 时刻的 sample-side投影指标。CPU MSE 使用 float64 归约；投影本身使用 FP32。

- 400/400：训练式 bridge 在 t=0 的输入逐位等于存档 x0。
- 400/400：使用原 Euler dt=.05 的最后一步逐位等于存档 x20。
- 使用实际 FP32 `1-t`（t=.95 时为 0.050000011920928955），最后时刻投影对存档 x20 的最大绝对差为 **1.1920928955078125e-7**；所有层合计 **0 个阈值判断差异**。这是舍入差，不是协议漂移。
- 当前执行代码在 t0 直接共用存档 velocity，因而 t0 两边相同是按构造成立，不是新 GPU 的独立重放验证。新的前向只覆盖 t=0.05,...,0.95，共 7600 次；当前 GPU 与原轨迹环境的数值差异仍是解释边界，不能把微小数值差异直接当作路径分叉。

仅作为 sample-side基线，depth9的投影 yhat：

| t | 占据 MSE，mesh mean | 空格 MSE，mesh mean | pooled F1 | FP | FN |
|---|---:|---:|---:|---:|---:|
| 0.00 | 0.41840 | 0.01990 | 0.30019 | 4382 | 86213 |
| 0.25 | 0.41544 | 0.01965 | 0.31317 | 6737 | 84780 |
| 0.50 | 0.43365 | 0.02964 | 0.31726 | 28057 | 80436 |
| 0.75 | 0.48401 | 0.06021 | 0.30555 | 68983 | 74154 |
| 0.95 | 0.50184 | 0.07103 | 0.30356 | 74886 | 73340 |

这是 `yhat(t)` 的比较，不是直接对 `x(t)` 做阈值。后段增加召回的同时显著增加 FP；仅看总 F1会掩盖变化。插值侧未产生前，不能据此宣布两路径最早分开时间。

CPU 复算命令：

```sh
/Users/luthier/Documents/sophomore/nexus_fast_track/.envs/nexus-algo/bin/python -B /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_A_interp_trajectory_20261002/audit/review_existing_sample_side.py
```

## 执行入口冻结审核

只读核对 `diagnose_interp_trajectory.py` 的最终 SHA256：`3e13653b8a2df1c4a960c0451041f1668f00fc3fe45eae9666920cc2e5715962`。关键采样/配对链无阻断项：

- 每层 archived parents 和 target_occupancy 与冻结 octree 重建结果逐位匹配，未发现 child bit 顺序错配。
- 实际原 Euler 状态、速度、时间、初始 noise、递推与哈希逐项核验；400 份原数组复制到新输出并再次验证哈希。
- 插值新前向确实使用训练 helper；sample-side不重新采样；两边使用相同 epsilon、FP32 t 和终点投影公式。
- t0 复用方式已写入 provenance；7600 次新前向和 8000 个深度时间配对的计数明确区分。
- endpoint 与真实插值 velocity 残差均拆分 all/occupied/empty；macro 与 pooled 分开；聚合实际读取 `time_rows`。均衡的每 UID 两 seed使当前100样本macro均值等价于50个UID各先平均两seed后再等权平均。
- 本审核不代表 GPU诊断已完成；运行结果仍须核对落盘数组、完整性及本报告的解释边界。
