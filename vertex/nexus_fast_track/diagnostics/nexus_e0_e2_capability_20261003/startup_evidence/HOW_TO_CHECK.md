# 本轮任务查看入口

服务器工作目录：

```text
/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003
```

`e2/queue_status.json` 记录 C/N/T 的进程、GPU、排队状态及退出码。
每支的 `e2/C/status.json`、`e2/N/status.json`、`e2/T/status.json` 区分起点评价、训练、定点评价和完成；尚未启动的分支没有输出目录。

`e2/<分支>/train.jsonl` 每行是一次已完成的 optimizer 更新。`e2_update` 是本轮新增步数，`step` 是累计步数 24000 + e2_update。每行保存实际消费的 8 个事件、UID、噪声哈希、时间、loss、梯度范数、耗时和显存峰值。

固定评价保存在 `e2/<分支>/evaluation/update-000000`、`update-001000`、`update-002000`；目录内有逐例 NPZ、rows.json、summary.json 和完成身份。它们是 depth9、GT parents 的诊断成绩，不是完整树生成成绩。

完整状态每200步保存，保留最近两份，另固定保留1000和2000；`checkpoint_identity.json` 给出最新已完成 checkpoint 的位置及 SHA256。原始 A24000 不删除。

所有分支完成后，脚本核对三支起点输出、事件覆盖及预测文件哈希，生成 `e2/RESULTS.md`、`e2/RESULTS.json`，再生成：

```text
NEXUS_E0_E1_E2_A24000_20261003.zip
e2_delivery_identity.json
```

ZIP包含日志、配置、执行代码、实际噪声/时间表、点云条件、标签、预测数组和指标；不含大权重或密码。训练完成、证据审核通过和ZIP完整性通过分别记录，不能仅凭进程退出认定全部完成。

失败不自动重试、不追加预算、不进入E3–E6。启动确认后不设持续监督或定时任务。
