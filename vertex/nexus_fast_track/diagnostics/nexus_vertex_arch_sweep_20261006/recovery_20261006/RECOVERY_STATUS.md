# 七组 Vertex 实验恢复记录

## 中断原因与恢复点

| 分支 | 原运行情况 | 完整恢复点 | 原预算终点 |
|---|---|---|---|
| S0 | update600写日志时触发Disk quota exceeded；日志完整至599 | update400 / 累计24400 | update10000 / 累计34000 |
| S1 | update600保存checkpoint时触发同一配额错误 | update400 / 累计24400 | update10000 / 累计34000 |
| S2–S6 | 未启动 | 仍分别从A24000开始 | 各update10000 / 累计34000 |

- 原队列因错误停止，没有最终生成成绩。
- 已重新读取两份完整update400文件核验SHA256，均匹配原身份记录。
- 两份完整恢复点均保留。旧401–600日志作为未提交尝试归档，重算时不重复计入权威日志。
- 仅清理本轮S1写失败的754,974,720字节临时checkpoint，保留其SHA及失败证据；未删除其他实验、基线或VAE文件。

## 持久存储

| 项目 | 旧位置 | 本轮恢复位置 |
|---|---|---|
| 挂载点 | `/ssdwork/guohaoran` | `/guohaoran` |
| 文件系统名称 | `ssdwork` | `storage` |
| 类型 | GPFS | GPFS |
| 本轮写入检查 | 16MiB仍报quota exceeded | 16MiB写入/fsync通过；430GiB真实分配/fsync通过 |

新工作目录：

```text
/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261006/
  code/                         独立恢复入口与测试
  audit/                        原故障、空间检查、启动及测试证据
  runs/queue_status.json        当前队列
  runs/S0/train.jsonl           权威日志：原1–400＋恢复后的401–10000
  runs/S0/checkpoint_identity.json
  runs/S1/...
```

`tmp`是持久挂载点里的目录名；服务器本机临时盘为`/tmp`。空间探针已经释放，其结果证明检查时可分配，并不占用430GiB等待训练。

## 继续执行的边界

- GPU0：`GPU-13261c86-f00b-6e11-11a3-437d15775a88`；GPU1：`GPU-80f199d2-afab-fad3-824d-6d2482a4c882`。
- S0/S1恢复模型、全部Adam状态、RNG、3200事件游标、各UID×depth曝光计数。S1新增参数的Adam步数为400，其余为24400。
- 固定点云、FPS索引、原数据与代码哈希保持一致。新恢复入口哈希单独记录，写入后续checkpoint的恢复元数据。
- 原结构、loss、数据、采样、lr、wd、精度、global8和microgroup2保持不变。
- 两卡并行S0/S1，之后自动调度S2–S6；每组终点仍为update10000。
- 恢复后额外在401步保存完整状态，之后仍每200步保存、保留最近两份。旧代码和旧结果不覆盖。
- 训练中不做生成评价；终点沿用原评价和最终打包流程。
- 8项独立恢复/队列测试通过；包括恢复后下一步与不中断计算的模型/Adam/事件一致性、错误状态拒绝、日志去重、GPU映射、失败停止和打包状态。

## 实际启动

后台队列PID：537；S0进程545、S1进程548。子进程与终端分离。

```bash
/usr/bin/python -u /guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261006/code/resume_queue.py \
  --base-root /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006 \
  --original-runs /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006/runs \
  --output /guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261006/runs \
  --start /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/experiment/A/checkpoint-024000.pt \
  --data /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/prepared_d9 \
  --microgroup 2
```

进程环境：`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=8`；GPU由队列按UUID隔离。

启动核验完成：服务器快照中S0已到update407，S1已到update402；两组update401完整checkpoint均已在新GPFS目录保存，SHA256和所有模型/Adam/RNG/config/游标/曝光计数读回核验通过。首个恢复更新的UID、depth、实际噪声哈希和time与原未保存尝试相同。没有最终生成成绩，不启用持续监督。

```text
S0 update401 SHA256 a814daa56a296e30ea7a3f6b79e44e4e0c7964dc60d89a2f2cb89722cf5ca7c5
S1 update401 SHA256 79cc3edf47ef1a0105263c394c5c8bd8981461e53d8401b175583ead5638fd04
```

启动证据见`audit/startup_verified.json`与`audit/startup_verified/S0/`、`S1/`。后续checkpoint按照原两份轮换规则替换，最新身份应读取新工作目录各组的`checkpoint_identity.json`。

恢复任务仍读取旧目录的冻结代码、固定数据和A24000；七组完成前继续保留这些来源文件。
