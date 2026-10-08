# NEXUS D9 A/B：同一起点的组合候选实验

本轮来源是 50 对象训练完成的累计 step 22000，不是目录名称中的 Resume7000。启动后只运行这一次预先确定的实验，不设置定时监督、不自动重试或追加训练。

## 起点与迁移证据

- 原始权重：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_overfit50_resume7000_20260930/run/checkpoint-022000.pt`
- 原始 SHA256（本轮服务器整文件重算一致）：`4c70104e6edc1490cf002598ecf79de94df9ac20a9b22881910556a2c55590b5`
- 共同 D9 起点：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/migration/checkpoint-022000-d9.pt`
- 迁移 SHA256：`5addf330bbe1a772a4a810181d5043f2fad5c7906a8c31d7c9dd7f9679471b83`
- 只截取 depth embedding 和对应 Adam 一、二阶矩的前 9 行；其他参数、buffer 和 Adam 状态保留。907 个参数状态均保留 step 22000。来源优化器是 Adam，不是 AdamW。scheduler 原本为 None。
- 实际 GPU 验证：depth 1–9 × t={0,0.31,0.79}，共 27 组 BF16 预测及条件特征均有限且逐元素相同，最大差值 0。RoPE 参考深度仍为 9。

## 两组设置

| 项目 | A | B |
| --- | --- | --- |
| GPU | 0 | 1 |
| 训练调度 | 对象优先 | 每个覆盖区间内按层级跨对象混合 |
| lr | 1e-5 | 1e-4 |
| Adam weight_decay | 0 | 0.01（耦合 L2） |
| 新增 optimizer 更新 | 2000 | 2000 |
| 起始 d9_update / source_step | 0 / 22000 | 0 / 22000 |
| 最终累计 step | 24000 | 24000 |

共同设置：同一 50 个 UID、固定点云和法向、D9、当前模型规模、velocity MSE、8 次梯度累积、clip=1、BF16 前向与 FP32 参数/Adam；VecSet 与 DiT 联合训练。只缓存不依赖可训练参数的 FPS/Fourier 预处理，不缓存 detach 后的条件特征。两组均不使用 activation recomputation。

每组 16000 份 microbatch。每个 UID×depth 得到相同的 35 或 36 次参与；最后不满一轮的 250 个事件只重排，不补样本。时间与噪声按 `(20260930, UID, depth, occurrence)` 配对。恢复来源 RNG 并保存后续状态，但本轮采用两组共用的私有随机数布局，不声称逐抽样延续旧 D15 的全局随机数序列。

## 数据与评价边界

D9 标签由已有原始浮点 GT 直接量化，并核实与 D15 整数右移 6 位后的唯一集合相同。原始记录 130328 点、浮点去重后 53200 点、D9 占据 52822 格；22 个对象有共 378 次量化合并。50 个对象全部保留，碰撞和映射单列。整数评价针对 D9 集合；连续 XYZ 与原始/去重浮点 GT 的指标另报，点数不同不伪报匹配 RMSE。

训练中不评估。两组完成且最终 SSD checkpoint 哈希验证后，分别用自己的同一个最终权重运行 Euler 与 flow 兼容 DPM，每层 20 次网络调用，阈值 0.5，从根展开 9 层。每组合 50 UID × seed {97029000,97029001}=100 树，共四组合 400 树。种子是配对比较用的已使用种子，不称未见终验集。父格仍相同时核对实际噪声相同；父格分歧后不声称噪声张量完全配对。

生成先保存原生整数点、连续占据、XYZ 和哈希，之后独立读取 GT 计分；没有 GT 点数截断、top-k、强制非空或预测修补。报告完整路径全部层正确率、最终整数集合正确率、首错层、逐层误差和 XYZ 指标。容量中止保留在分母内。两种完整树指标分开报告。

## 执行与保存

服务器工作根：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930`

- 训练日志与持久 checkpoint：`experiment/A/`、`experiment/B/`。
- 临时写入目录：`/tmp/nexus_d9_ab_20260930/A/`、`B/`。
- 第 1 步及每 200 步保存完整状态，校验后提交 SSD；各分支保留最近两份。源权重与迁移起点独立保留。
- 完整状态包括 model、Adam、step、d9_update、RNG、样本参与计数、下一事件位置、配置及数据/代码哈希。
- `audit/launch.json` 保存实际启动命令、PID；`experiment/audit/phase.json` 保存当前阶段。
- 最终评价：`experiment/evaluation/{A,B}/`。
- 成功后结果包：`experiment/NEXUS_D9_AB_results.zip`；身份记录：`experiment/delivery_identity.json`。包含代码、配置、日志、实际条件与标签、预测、指标、哈希和测试证据，排除大权重、Python 环境及凭据。

预检实际通过：最大 depth9 有 2528 parents，8 次累积及一次丢弃的 B 配置 Adam 更新；峰值 allocated 52.459 GiB / reserved 54.535 GiB，梯度有限且 VecSet 梯度非零。该更新未保存、不计入 A/B 预算；两组重新加载未更新的共同起点。

CPU 测试：训练核心 5 项、采样器 5 项、评价入口 2 项均在服务器通过；编排 2 项在本地 stand-in 测试通过。CPU oracle 和迁移相等性不是训练后生成成绩。

启动状态以 `audit/startup_verified.json` 的实时核对为准。本卡不是训练完成或模型通关证明。
