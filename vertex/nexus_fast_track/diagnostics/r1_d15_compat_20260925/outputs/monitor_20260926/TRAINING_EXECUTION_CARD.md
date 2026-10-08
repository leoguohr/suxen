# R1_D15_COMPAT 首轮训练执行卡

## 2026-09-25 最新指令：续训，暂不评估

用户要求重新启动，并明确“启动之后不需要评估，保存checkpoint一会统一评估，只启动就好”。本节覆盖下方旧卡的自动开发和终验安排。

- 从已校验的 update800 恢复模型、Adam、CPU/CUDA/Python/NumPy RNG 和层级调度，不重新初始化。
- 端口31548，仅GPU0；累计上限6000，每200步保存一份独立完整 checkpoint，原800步权重保留。
- 启动、训练中及结束后均不自动评估，等待用户要求统一评估。
- 运行目录：`/tmp/r1_d15_compat_20260925/run-resume800`。
- 持久目录：`/guohaoran/tmp/r1_d15_compat_20260925/resume800_20260925`。
- 入口：`scripts/train_d15_resume.py`；wrapper：`scripts/run_resume800.py`。
- 本进程使用 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 兼容实例自带 ONNX/protobuf；不修改共享 Python 包。
- 原卡的模型、数据、优化器超参数及异常停止约束继续适用。

## 原始执行卡（历史评估协议已被上节覆盖）

授权：用户已确认按 NEXT_AGENT_TASK.md，并明确“你就一直训练就好了，不需要管时间问题”。因此本轮已获训练授权；不设GPU小时硬上限。不是将原30分钟推理请求自行扩大为训练。仍只占用已检查空闲的GPU0，不使用GPU1或V2工程。

- 设备：A100 GPU0，UUID `GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d`；进程仅可见这一张卡。
- 首轮更新硬上限：**6000**，update从0开始；等价每个“对象×15层”1600个microbatch。达到上限则保存并报告，不以预算耗尽宣称通过。
- 模型起点：D2累计7400/update3600完整权重；哈希 `2384b430fa52d5012cd793fc2781c79791814f6a4b2e8ac120a711fb91112e1b`。先CPU显式迁移、整模型strict load及共享参数哈希核对。
- 唯一模型变更：depth embedding 9→15行；新增6行用独立CPU Generator seed2026092515、std0.02初始化。RoPE reference保持9。无坐标先验、offset、count head、新loss。
- 数据：仅 `nexus_2k_000105`、`nexus_2k_000195` 的已有真实8192点/法向；固定输入、无增强。D15标签由原stage2浮点值按float64 floor/clamp重新编码；不能用q9乘64。原始重复记录和唯一几何点分别报告。
- Fresh Adam：lr1e-5、WD0、默认betas(0.9,0.999)/eps1e-8、foreach=False、clip1、累积8、BF16、沿用D2激活重计算；VecSet与DiT联合训练。不读老师Adam，不恢复旧D2 Adam/RNG。
- 新训练RNG：seed2026092516；每micro独立正态噪声和uniform t，速度监督Y-ε，阈值0.5。
- 均衡调度：`k=((update-1)*8+micro)%30`；mesh=k//15，depth=k%15+1；30个组合均持续参与。
- 日志：逐update保存loss、梯度范数、耗时、实际UID/depth/t/noise SHA，双写到NVME及独立持久目录。
- 固定检查：update1先落盘一份完整训练checkpoint；之后每200更新先保存完整checkpoint、哈希回读，再运行4个开发种子×2条件＝8条全树；15层每层20 Euler，非空树300次去噪调用。
- 开发seeds：91015000..91015003；开发全部完整整数树正确后，用预留92015000..92015015共32条树终验一次。点集同时匹配两个GT，预期身份矩阵为单位矩阵。缺失、重复、非complete/超时按整体incomplete失败，不筛掉失败记录。
- 终验停止：32条完整15层整数树全部正确则停止并报告；连续XYZ原始GT/显式去重GT分别记录，不把原始重复点删掉后宣告CAD50标准通过。首轮终验失败就保存失败样本停止诊断，不把这些seed再次称为未见终验。
- 异常停止：非有限loss/梯度、代码异常、OOM、存储/哈希失败、SIGTERM/INT。异常不自动调学习率、改loss或换数据。
- checkpoint：model、fresh Adam及后续完整moment/step、独立step、配置/版本、迁移身份、CPU/CUDA/Python/NumPy RNG、下一调度位置、代码与数据哈希；恒定LR，scheduler=null。原D2/D4 checkpoint只读。
- 最终报告区分：表示检查、CPU小模型、完整D2重放、D15实际训练与生成；D15两物体成绩不称CAD50生成结果。

服务器路径（新目录）：代码/NVME `/tmp/r1_d15_compat_20260925`；持久结果 `/guohaoran/tmp/r1_d15_compat_20260925`。完整权重只存服务器，交付轻量包含路径/身份与日志、预测、指标。
