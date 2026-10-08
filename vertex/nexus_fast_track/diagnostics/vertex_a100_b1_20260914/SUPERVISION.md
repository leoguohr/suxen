# 当前任务已推进到B2

唯一当前监督入口：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_b2_20260915/SUPERVISION.md。不要启动旧任务。以下为历史证据。

# 已被2026-09-15独立单mesh B1替代

用户最新明确要求先跑独立单mesh B1，已启动新实验。当前唯一监督入口为 `/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_b1_single_20260915/SUPERVISION.md`。本文件以下均为历史，不启动或续跑A100。

# 2026-09-15 最新只读检查：再次更换实例，暂停状态

用户提供新认证并要求查看结果。本次实际主机为 `1fl3imip3trjj-0`，ControlPath `/tmp/nexus-vertex-31548-sep15.sock`。原restart1的 `/tmp/vertex_a100_b1_20260914` 不存在；GPU空闲，共享准备目录没有run或checkpoint。旧PID3850不可在新实例上使用。自动化当前PAUSED，本次未启动新训练。

本地最新完整训练日志到595步，最新全量评估500步，A未过（MSE1.0765924，坐标精确0/100）。之后旧实例是否继续运行及其结果未知，不能说A完成/通过或B1已运行。详情见server_inspection_20260915.json。以下历史记录仅用于追溯，不代表当前运行。

---

# 2026-09-14 新服务器 restart1（本记录优先）

用户通知旧服务器消失并提供新认证。旧/tmp实验目录和检查点不在新主机，不能续训。最后观察到的旧A进度301步，不构成通过。使用相同代码、数据、种子和预算，从step0重新训练；B1门槛不变。

当前主机 `duot5s8128gjk-0`，PID `3850`，start_ticks `315215871`，ControlPath `/tmp/nexus-vertex-31548-new.sock`。旧主机PID5254已经失效，勿操作。当前launch位于本地 `restart1/launch.json`。冻结入口SHA256 `b5ae0feafb4a7107e1584c0845a0c644954d98223d368b6335ad5eab0c9e43a5`。

共享盘单文件fsync通过，但展开代码阻塞于cxiWaitEventWait；解包进程已终止。实际运行仍位于 `/tmp/vertex_a100_b1_20260914`，不能声称检查点已持久化。共享盘 `/guohaoran/nexus_fast_track/vertex_runs/vertex_a100_b1_20260914_restart1` 只有不完整的准备文件，不能启动其中代码。

57个代码文件与402个数据/manifest哈希全部一致。首轮32测试中B1 CPU子进程发生signal11（31通过），未改代码复测32通过；保留两个测试日志，根因未知。完整最大样本两步GPU预检通过，峰值35.79GiB。

**每次监督将小型日志/状态同步本地 `restart1/`，阶段评估同步对应JSON/NPZ，绝不覆盖旧主机run/证据。不要下载权重。** checkpoint仅保留当前服务器/tmp，实例更换仍有丢失风险。发生连接或运行异常先核验，不重复启动。

---

# 当前优先任务：100-mesh A → 独立 B1 R0/R1

用户最新正文要求：直接换100个mesh做A；只有A再次通过，才按附件做新的B。正文的100-mesh A优先于附件原文“不要扩数据”的旧建议。不要延续旧A权重做B，不要再运行旧的500步配对诊断。

## 运行身份

- 本地 `/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_a100_b1_20260914`
- 服务器 `root@172.16.78.10:31548`，主机必须为 `duot5s8128gjk-0`。
- SSH ControlPath `/tmp/nexus-vertex-31548-new.sock`。认证来自本任务用户最近更新的信息；不要保存或打印密码。
- 当前远端目录 `/tmp/vertex_a100_b1_20260914`，代码冻结在 `code/`，实际100个样本在 `data/`，运行结果在 `run/`，主日志 `console.log`。
- 当前PID `3850`，start_ticks `315215871`；`restart1/launch.json`也保存当前身份（根目录launch.json为旧主机记录）。只在身份核验后操作本PID，不干扰其他任务。
- 单A100 80GB。原2949诊断已完成，GPU空闲后才启动新流水线。旧证据仍在旧目录，不覆盖。
- 使用 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /guohaoran/envs/nexus-algo/bin/python`。共享目录创建操作曾阻塞，因此本次在服务器本地盘运行。

## 已完成的准备

100个train/keep样本来自人工去重清单，60 Objaverse+40 ObjaverseXL，保留000105，没有放宽质量门槛；选择文件和哈希本地齐全。402个数据/manifest文件下载后SHA256已验证。100个样本全部进入训练，不按大小裁剪或降额。

最大000274有18307个顶点。完整2,332,430,344参数模型的2步最大样本前向/反向/AdamW预检通过，峰值35.79GiB，无OOM。预检不是模型验收成绩，也没有拿预检权重继续训练。本地和服务器32项相关测试通过。

源码入口 `code/scripts/train_vertex_a100_b1.py`。实际启动命令为 `--phase pipeline --expected-samples 100 --a-updates 20000 --a-eval-every 500 --b-updates 500 --b-max-updates 1000`，全尺寸模型，无smoke flag。

## A100 门槛

一个共享完整模型，从头原始初始化R0，固定100个问题：每个UID固定自己的点云、depth9 GT parents、t=.5与噪声（seed由UID+实验seed确定）。每100更新覆盖所有对象，确定性打乱。一次更新一个对象，LR1e-4、WD0、warmup100、clip1、BF16 autocast/FP32参数与优化器，无增强。

首轮上限20000更新，每500更新评估全部100个mesh。**每个**对象都要MSE<=1e-4、相对基线MSE<=1e-3、占据和完整整数坐标集合精确一致；连续三次全量通过才允许B1。平均loss、99/100或点数相同都不合格。若预算耗尽未过，就保存证据并停在A，不自行加预算、改超参数或绕过门槛。

看 `run/status.json`（oracle/evaluating/training/saving）、`run/A100/train.jsonl`、各evaluation JSON、`arrays_latest/`、`result.json`。`arrays_latest/`会被下一次评估覆盖，应以相应最新完整评估step解释，不能拼成某一次“最好结果”。

## B1（只有A100通过后才由同一进程自动启动）

R0/R1是独立新模型、独立新AdamW，均不加载A或旧B权重，使用同一初始化seed及相同训练噪声序列。均只用000105、depth9、t=.5、固定实际点云；每次更新8份独立噪声，8次forward/backward分别loss/8，再裁剪更新一次；LR1e-5、WD0、warmup100、clip1。

R0原始初始化；R1仅初始化组合：depth embedding std=.02，time MLP两层权重std=.02/bias0，cross-attention输出投影weight/bias0。保持self/FFN modulation与最终输出的零初始化。此组合是用户授权的候选，不是已证实修复或论文要求。R1 VecSet梯度头两步可为零，测试验证之后可达，不能误报断图。

每组初始500更新、每50验证16份固定校准噪声和4份FP32响应。只在最后三次评估严格改善且100更新内MSE降低>=10%时，允许一次延长到1000。

校准每例MSE<=.01且坐标精确，FP32 signed gain在[-2.2,-1.8]且relative response error<=.1。候选达标后只使用一次64份独立holdout（不用它调参），全部要求MSE/坐标/FP32响应通过；无论holdout结果如何都结束该组。不得反复在同64份上筛选。

每层RMS和每token top1/top8平方能量占比存于`responses_fp32[].layers`。B1仅固定t，不检查从0积分到1；不能称B2/C/D通过。流程执行完R0/R1后结束，后续按附件结论分析，先不擅自进入多层生成。

## 监督与交付

不能把state=complete误当所有门槛通过，要读每个result的passed。不要重复启动、修改运行中的代码、或扩大/缩小已固定100样本。遇到OOM/非有限值先保全错误证据与GPU状态，禁止静默换小模型。正常推进保持安静，只报告首次有意义的结果、门槛通过/失败、异常或需要用户输入。

日志、小型JSON/NPZ、源码/数据校验需要同步本地。checkpoint只保留服务器，用户明确不要下载/打包模型权重。根因尚未证实，不把特征放大、时间编码、学习率或BF16单独说成根因。R0/R1同时改变的训练协议与初始化对照要按实际变量解释。

详细协议见`PROTOCOL.md`，用户原文见`user_B_plan.md`。旧阶段复核包在旧目录，不能冒充新100-mesh实验结果。
