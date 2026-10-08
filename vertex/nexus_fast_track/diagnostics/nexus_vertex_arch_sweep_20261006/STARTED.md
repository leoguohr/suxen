# Vertex 七组实验：已启动

2026-10-07 最新接续见 [本次启动记录](recovery_20261007b/CONTINUATION.md)。S0/S1训练已各完成10000步，现已双卡补原终点评估；S2–S6自动接续各10000步。当前输出目录为 `/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007b/runs`；下文及recovery_20261007保留历史快照。

启动核验时间：2026-10-06 02:26:13，中国时间。下面是启动快照；之后不进行持续监督。

## 任务

| 分支 | 结构方案 | 新增 optimizer updates |
|---|---|---:|
| S0 | 原结构同期对照 | 10,000 |
| S1 | 增加末层时间 AdaLN | 10,000 |
| S2 | 关闭 cross-attention 的 QK norm | 10,000 |
| S3 | VecSet 改为 learned queries | 10,000 |
| S4 | TRELLIS1 注意力归一化＋DiT 末层调制组合 | 10,000 |
| S5 | Hunyuan 编码器与注意力细节适配组合 | 10,000 |
| S6 | 老师候选的全局条件双入口等细节适配组合 | 10,000 |

- 合计70,000次新增更新。七组独立由A24000出发，终点各为累计step34,000。
- 原50对象、D9、velocity MSE、lr=1e-5、wd=0、clip1、BF16前向和FP32模型/Adam。
- 每次更新全局8个事件，每次同时计算2个；两张卡各自运行独立分支。
- 参数迁移、源码依据及组合的适配边界见同目录 `ARCHITECTURE_CARD.md` 和 `EXECUTION_CARD.md`。

## 启动证据

| 项目 | 本轮实际结果 |
|---|---|
| 后台队列PID | 2166；独立session，终端退出后继续运行 |
| GPU0 | S0，PID2189；启动快照已到update74 |
| GPU1 | S1，PID2199；启动快照已到update73 |
| 等待队列 | S2、S3、S4、S5、S6；空闲卡自动接续 |
| 共有模型与Adam | S0/S1共有907参数及2721个Adam状态张量逐值一致；S1新增2个参数张量的Adam从0开始 |
| 配对输入 | 正式首步8个事件的UID、depth、实际noise SHA、time完全相同；首步loss差为0 |
| 正式保存 | S0、S1均完成update1完整checkpoint保存、SHA256及模型/Adam/RNG等全量读回 |
| CPU核验 | 20项通过，覆盖七组结构、非零旧模型、mask、批处理梯度/Adam等价、保存和队列失败行为 |
| GPU最长序列预检 | S0/S5：2528个父格，global8；实际张量峰值65.58/66.17GiB，预留峰值67.24/67.81GiB |

最长序列显存实测相当于约70.4/71.0GB张量。正常训练随序列长度变化，启动前几十步约44–45GiB；没有分配无用途张量填充显存。梯度重算关闭。

预检只执行了独立的启动测试，未计入正式预算。其两份约28GB测试权重已删除，日志、哈希与读回证据保留；原权重未删除。

## 服务器位置

服务器：`root@172.16.78.10:32483`。认证信息未写入文件。

```text
/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006/
  runs/queue_status.json
  runs/audit/commands.json
  runs/audit/startup/launch_verified.json
  runs/S0/train.jsonl
  runs/S0/status.json
  runs/S0/checkpoint_identity.json
  runs/S1/...
```

实际启动命令，已通过独立session在后台运行，勿重复启动：

```bash
cd /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=8 \
  /usr/bin/python -u run_sweep_queue.py --microgroup 2
```

## 保存与完成行为

- update1保存，之后每200更新保存完整状态；每组保留最近两份，最终checkpoint保留。
- 训练中不做生成评估。每组10,000更新后，按原50对象×2种子、D9完整树、20 Euler步/层、阈值0.5进行评估。
- 七组完成后自动汇总并生成：
  `runs/NEXUS_VERTEX_ARCH_SWEEP_A24000_20261006.zip`。
- 结果包包含预测数组、指标、日志、配置、代码和身份哈希；不含大权重及认证信息。原始输入数据以来源路径和哈希登记。
- 非有限、OOM、保存/验证失败或异常退出会保留证据并停止启动待运行分支；不自动修改配方或重试。另一已运行分支可完成。
- 本轮没有创建心跳或定时监督。当前确认的是启动与完整保存成功，生成结果尚未产生。

## 本地证据

- `startup/launch_verified.json`：正式启动快照和首步配对证据。
- `startup/S0/`、`startup/S1/`：实际配置、迁移核验及checkpoint身份。
- `startup/final_cpu_tests.log`：20项CPU核验。
- `startup/memory_preflight.json`：完整模型显存、速度和保存核验。
- `audit/code_manifest.json`：有效Python文件SHA256；已与本地文件逐一比较。

正式update1 checkpoint SHA256：

```text
S0  5b2e781b082ffb5eafaca6e9db559959e963f9543a604cc9f031ac001fb8a855
S1  3c47ff32da0a85339a3af3c4d27b004b2a1d670b4781de8f3fe0dc895da26cde
```

这些启动恢复点会按两份滚动保留规则被后续checkpoint替换；最新位置与身份应以服务器各组 `checkpoint_identity.json` 为准。
