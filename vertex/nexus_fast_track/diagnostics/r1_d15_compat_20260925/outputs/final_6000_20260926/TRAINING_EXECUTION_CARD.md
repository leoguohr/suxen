# R1_D15_COMPAT 首轮训练执行卡

## 2026-09-26 当前续训：5200步起，结束后一次性评测

服务器实例再次更换后，旧日志停在5386步，最后完整权重为5200步。用户要求继续。新入口 `scripts/train_d15_resume5200.py` 从完整SHA校验的5200步恢复模型、Adam与全部RNG，旧日志保留，新日志继承至5200步并重跑5201..5386。新运行目录 `/tmp/r1_d15_compat_20260925/run-resume5200`，持久目录 `/guohaoran/tmp/r1_d15_compat_20260925/resume5200_20260926`，GPU0 UUID `GPU-d59b34c9-1810-15c1-4cc9-db2bd953b574`。

每200步保存完整checkpoint，累计上限6000；训练期间不评测。`scripts/run_resume5200.py` 在训练进程正常退出并写入退出状态后，仅调用一次 `scripts/evaluate_after_training5200.py`；评测脚本再次检查状态与6000步权重的SHA后，对预留16个种子×2个物体运行完整树生成和独立评分。异常退出时不评测。不设置定时监督。

## 历史执行要求（以下路径与安排由上节覆盖）

## 2026-09-26 最新指令：训练结束后评测，无定时监督

用户要求训练期间只保存 checkpoint，训练完成后再评测，不需要任务监督。2600步续训已在后台运行，每200步保存，累计到6000步。服务器另有一次性收尾进程 `scripts/evaluate_after_training.py`：只在训练进程以0退出、status为 `update_budget_complete_evaluation_deferred` 且step为6000时读取完整权重并评测；异常退出则记录跳过。评测使用16个预留种子 × 2个物体共32条完整树，20 Euler步/层、15层，先保存预测及哈希，再独立读取GT评分。结果在持久目录 `resume2600_20260926/post_training_final_006000/`。原每15分钟的 `r1-d15` 定时任务已删除。

## 历史执行要求（以下自动评估安排由上节覆盖）

## 2026-09-26 当前续训：从2600步恢复，继续暂不评估

用户再次明确“继续”。当前入口 `scripts/train_d15_resume2600.py`，wrapper `scripts/run_resume2600.py`。本节覆盖旧800步续训的路径及设备信息。

- checkpoint：`/guohaoran/tmp/r1_d15_compat_20260925/resume800_20260925/checkpoint-002600.pt`，SHA256 `bc13e55829ab4252e18ae9a040823f660e82627209f034d572be217fa1cfc3bd`。
- 新运行目录：`/tmp/r1_d15_compat_20260925/run-resume2600`；新持久目录：`/guohaoran/tmp/r1_d15_compat_20260925/resume2600_20260926`。
- 端口31548；仅GPU0，UUID `GPU-d59b34c9-1810-15c1-4cc9-db2bd953b574`。
- 恢复模型、Adam、CPU/CUDA/Python/NumPy RNG及调度。旧日志1..2757保留不动，新日志继承1..2600，未保存的157步重新执行，记录history_boundary.json。
- 累计上限6000，每200步单独保存完整checkpoint；学习率、loss、模型与两个训练对象不变。
- 不做启动、开发、终验或结束后评估，等待用户另行要求统一评估。异常保存证据，不盲目续跑。

## 2026-09-25 历史指令：续训，暂不评估

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
