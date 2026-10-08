# 七结构实施审查

审查范围：本地只读检查结构、迁移、训练loss/事件/随机流和有界队列。审查者未连接SSH、未启动训练、未执行GPU测试。代码可读性与源码接线结论不能替代真实A24000发车核验。

## 结论

2026-10-06所审模型代码符合 `ARCHITECTURE_CARD.md` 的 S0–S6 配置，未发现剩余结构或Adam迁移阻断项。S4/S5/S6是完整配套适配；其效果必须解释成共同来源下的10000-update暖启动效果。各分支起始函数不完全相同。

已发现并闭环两个问题：

1. 构造函数原先在 `fork_rng(devices=[])` 内调用 `torch.manual_seed`，会改写未被fork保存的CUDA RNG。现已改成 `torch.random.default_generator.manual_seed`，仅修改并恢复CPU默认generator；新增参数另有按名字确定的私有CPU generator。
2. 原迁移audit只有key/shape与动作清单。现已对每个迁移model tensor和每个继承Adam的step/一阶矩/二阶矩执行复制后 `torch.equal`，失败即拒绝，并把检查数与all_equal写入audit。S5变形的shared RMS gain仍明确重置Adam。

## 已逐项核对

- S4 TRELLIS公式在float32执行，gain `[12,128]`按head/channel广播；CA norm移除，final AdaLN零初始化；旧已训练output和modulation没有被重新初始化。
- S5 VecSet32heads、共享gamma/beta `[64]`的QK-LN、无π输入频率、编码器与DiT两侧QKV bias删除、attention输出bias保留、exact GELU、VecSet末LN eps1e-5；DiT共享gain `[128]`按原head均值迁移并重置其Adam；不删除首个父格。
- S6先将无效condition tokens置0再做有效token均值；global LN+两层MLP末层全0；g分别加到入口token与每层调制条件；父格padding再次归0。CA、depth、RoPE、8维velocity接口完整保留。
- 源与目标optimizer按名字映射；当前S0参数顺序与冻结原 `VertexStageSystem` 相同，测试中显式比较；新增/变形参数step0、其余step24000；各分支独立clone，没有参数或Adam storage共享。
- `VertexStageSystem.forward`按每对象有效父格数归一化，runner分组loss乘以B/8；数学上保持每update八个事件等权。pad mask、不同depth、不同对象的batch轴保持独立。
- 新sweep cursor0、UID-major×D1–D9、私有event-keyed noise/time明确写入config；源cursor16000只作provenance。每次noise实际数组hash及time值逐事件保存；最终七套比较数据、UID顺序、exposure与noise/time摘要一致性。
- 单UUID隔离，每进程CUDA逻辑0；源逻辑CUDA RNG恢复并readback。实际noise/time来自新公共event随机流，所以此恢复不表示沿用源下一次噪声。
- checkpoint原子写入、fsync、读取后逐值核对model/Adam/RNG/config/cursor/counts；完成时验证各参数Adam step等于自身initial step+10000。
- 最终评估为每套50对象×2seed、root到D9、20步Euler。每个预测先落盘并hash，再读取GT评分；与GT-parent能力探针区分。
- 队列只启动指定两卡，GPU空闲检查与flock防重复启动；分支失败停止启动后续队列，无自动修改协议或无限重试；已运行分支按原有有界工作结束。

## 测试与发车边界

已读测试覆盖：独立非零原模型S0/S1等价、完整meta尺寸、S3 query梯度/无FPS、S4公式、S5 shared norm迁移、S6 masks、全量Adam值/独立storage/混合年龄step、RNG隔离、固定点特征缓存输出与梯度一致性、不同长度batch的per-object loss。

这些是测试内容审查，**本审查者没有执行这些测试**。root已报告12项CPU测试全部通过，正式证据以实施方保存的test结果为准。发车前仍由root负责真实A24000 CPU/GPU迁移、finite forward/loss/grad、实际显存与每卡启动验证。分支microgroup若不同，其BF16数值路径可能不同，不能声称逐数值完全等价。

本轮初始比较保存首个update中、optimizer.step前的共同8事件loss；S0/S1输出等价依赖独立非零小模型测试，不新增0步GPU输出数组或额外评估。这是足够支持本次筛查启动的边界，不能将它表述成已验证全尺寸GPU的S0/S1逐输出等价。

交付状态问题也已修复并复读确认：ZIP制作前先将queue_status写成 `all_training_and_evaluation_complete_packaging` 与 `evidence_audit_passed=true`，避免包内误留aggregating。独立runner测试已加入七分支、group1/2/4/8的loss/梯度/一次Adam更新/参数与moments对照；其执行结果仍由root收集。

## 审查时文件身份

以下路径相对于 `nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006/`。后续更改应查看差异，不应将本次审查自动套到新版本。

| 文件 | SHA256 |
|---|---|
| architecture_variants.py | `2e7390ac4e48ce3426eefe6450f772d6d1287ce39ae69760460daf01d1742ce5` |
| train_sweep.py | `e60bbc1d6e09292ec656fd8740f1862bae9a6196ae1ab8ad5206b08d211dd1ea` |
| run_sweep_queue.py | `3a8d9d9099b57a4fa05fee8b326de94ccea0dd1e70ec1d71bd25b8bf0d9a2b25` |
| test_architecture_variants.py | `b9e9366689bed7a4c5cc7920a905bfd37a4f4295d6cb41d76dc364921235d552` |
