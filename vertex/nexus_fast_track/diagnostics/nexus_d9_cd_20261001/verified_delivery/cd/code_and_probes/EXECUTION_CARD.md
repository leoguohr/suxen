# D9 A/B 冻结诊断与 C/D 交叉对照

本卡在本轮大模型诊断结果产生前固定协议。A24000 是现有工作基线；C、D 均从共同 D9 step22000 完整状态开始。仅增加观测代码，不增加全局条件 AdaLN、坐标先验或其他结构，不改变数据、目标、采样器或预算。启动一次流水线后不设置定时监督，不自动重试或追加训练。

用户最新要求充分使用两张卡，覆盖此前依次执行的调度偏好：C 使用 GPU0、D 使用 GPU1 并行。保留正在运行的 C 进程，D 从共同起点独立启动；各分支完成 2000 更新并通过保存完整性核验后，在自己的 GPU 进行最终生成评估，全部完成后复核与统一打包。只改变进程调度，不改变训练有效 batch、事件顺序、随机数协议、配方或预算。此前已完成的 A/B 冻结诊断不重跑。确认后台流程启动后收尾，不设置定时监督。

2026-10-01 用户再次要求两张卡共同加速 D。C 已完成训练及评估；D 在 update1550（累计23550）正常保存后，改用独立 `train_d_dualgpu.py`，从1551继续到2000。物理GPU1保留模型和Adam，GPU0为模型副本；每对相邻micro并发计算，奇数micro的梯度归并后才开始下一对，最后统一clip和一次Adam。有效8事件、数据/噪声键、配方及预算不变；执行代码变化和浮点边界另存 `DUAL_GPU_SWITCH.md`。C结果复用，D结束后继续原最终评估和四组合包。

## 来源与配方

冻结基线根目录：`../nexus_d9_ab_20260930`。服务器根目录在 `/ssdwork/guohaoran/nexus_fast_track/diagnostics/` 下。

| 分支 | 调度 | Adam lr | Adam weight_decay | 本轮用途 |
| --- | --- | --- | --- | --- |
| A | 对象优先 | 1e-5 | 0 | 已有 step24000，仅补冻结诊断 |
| B | 每覆盖区间内按深度跨对象 | 1e-4 | 0.01 | 已有 step24000，仅补冻结诊断 |
| C | 与 B 相同 | 1e-5 | 0 | step22000 开始，2000 次更新 |
| D | 与 A 相同 | 1e-4 | 0.01 | step22000 开始，2000 次更新 |

这是调度与 LR＋WD 配方两个因素的交叉对照；不能分别归因 LR 与 WD。

- 共同起点：`migration/checkpoint-022000-d9.pt`，SHA256 `5addf330bbe1a772a4a810181d5043f2fad5c7906a8c31d7c9dd7f9679471b83`。
- A24000：`experiment/A/checkpoint-024000.pt`，SHA256 `2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`。
- B24000：`experiment/B/checkpoint-024000.pt`，SHA256 `183f703c7718156fa9c3204a209b55a641193ad414522bf68d1141d1ff27039d`。
- 数据 manifest SHA256：`101cc76ffbbe55012a8ad5bfbbb5c54148a755f1562fd0236ec0cbca70e7a6e6`。

保持 50 UID、D9、固定点云与法向、原 VecSet＋DiT、velocity MSE、8 microbatch/更新、clip=1、BF16 前向和 FP32 参数/Adam、无激活重计算。每组 16000 个事件，每个 UID×depth 参与 35 或 36 次；最后不满一轮的 250 事件只重排、不补齐。噪声和时间继续按旧键 `(20260930, UID, depth, occurrence)` 生成，C 复用 B 顺序，D 复用 A 顺序。

恢复 model、全部 907 个 Adam 参数状态、step22000 和来源 RNG；scheduler 原本为 None。保留来源恢复与新配对私有随机数布局的区别。每支只跑 2000 次新更新，最终累计 step24000。

## 冻结探针

先执行 A24000/B24000 相同诊断，再执行共同来源 step22000 诊断，之后启动 C/D。普通固定 bank 为 50 UID × depth1–5 × t={0,0.25,0.5,0.75} × 2 份固定噪声，共 2000 次 flow 前向。噪声实际数组、UID 顺序和哈希落盘。具体 seed、pair 子集及索引在生成 bank 的 manifest 中记录；选择只依据 GT，不依据模型表现。

由 `xt=(1-t)*noise+t*target` 得 `x1hat=xt+(1-t)*velocity`，连续预测不 clamp。记录 velocity MSE、干净占据 MSE、TP/FP/FN、阈值 0.5 下集合正确性，及零速度、全空/全满占据对照。不凭高时间步的输入先验宣布学习有效，不追加事后硬门槛。这些给定 GT parents 的探针是定位证据，不是从根生成的完整树成绩，也不证明未见物体泛化。

条件切换仅采用完整有序父格集合相同、子格目标不同的对象对。当前 depth1–5 的合格无序对数量为 364、850、11、0、0。每个可用层确定性选最多 4 对；depth4/5 记 N/A，不替换为父格交集。

每对检查 t=0 共享纯噪声，以及 t=0.5 的双向 target anchor。每个 anchor 内固定完整的 xt、parents、depth、mask、time，仅切换条件，通过 `model.flow` 预测，不调用会重建 xt 的训练 loss 入口。两种预测均对照两个目标，记录对角/非对角误差、预测差与目标差的投影/方向，以及目标不同 bits 上正确翻转。非零 t 时切换条件后的正确速度为 `(target_j-xt)/(1-t)`，不是 `target_j-original_noise`。

训练固定探针为 update200、400、…、2000，共同 update0 已预先执行。每次重算 learned condition context，仅在本次诊断内复用。记录 encoder LayerNorm 前后 token RMS、各 cross-attention 注入残差相对输入的 RMS。尺度只作定位，不能单独证明条件有效或失效。

探针暂时 eval＋inference_mode，保存/还原 Python、NumPy、CPU、可见 CUDA RNG 及逐 module training flags，卸除临时 hook。不得污染下一次训练更新。CPU 隔离测试须核对开/关探针后的模型、Adam 与 RNG 一致。

## 实际更新与 L2 诊断

update1 及每 200 步记录模块参数更新。模块至少分 VecSet、输入/embedding、cross-attention、AdaLN/modulation、self-attention、FFN、输出。实际数据梯度先按原协议 clip=1，Adam 内再加 `weight_decay*parameter`。

从同一更新前参数、Adam 一二阶矩、step 和裁剪后的数据梯度，只读计算实际配方与当前一步去掉 L2 的反事实更新；optimizer.step 仍只调用一次。记录两种更新、差值/方向/相对参数范数，并对照真实 FP32 参数变化。反事实保留历史 moments，只表示当前一步边际影响，不能当成全程无衰减训练。CPU 测试须对照部署版 torch Adam；新增诊断在最大序列、8 次累积下做丢弃式 GPU 显存预检，不保存预检权重，不计入训练预算。

## 保存、终评与交付

第 1 步和每 200 步保存完整 checkpoint，经临时文件、SHA256、SSD 读回校验后提交，各新分支保留最近两份；共同来源和 A/B 基线不清理。训练逐步日志另存。失败保留证据并停止，不改超参数自行续跑。

C/D 完成后各沿原协议生成：50 UID × seed {97029000,97029001} × Euler/DPM 两方法，depth9、每层 20 次网络调用、阈值 0.5。共新增 400 条树。固定种子是重复比较种子，不称未见终验。沿用原容量上限 4096，失败纳入分母，禁止 GT 点数修补。原生输出先保存并哈希，再独立读 GT；报告完整树、整数集合、逐层误差、首错层及符合匹配定义的 XYZ 误差。

A/B 原 400 条树直接复用已校验结果，本轮不再执行其完整生成。最终轻量 ZIP 包含 A/B 原结果、四组冻结/过程探针、C/D 日志与预测、模块/L2 更新统计、代码/配置/数据哈希、checkpoint 身份及运行边界，不包含权重本体、环境或凭据。

本卡是执行约定，不是代码已测试、训练已启动或模型已通过的证明；实际状态以 audit 与运行日志为准。
