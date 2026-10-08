# 执行命令

以下均为本次已批准固定100条诊断的适配命令，并非论文通用复现命令。运行环境为既有 `/opt/conda/bin/python`，未安装/升级训练依赖。

## 零更新预检（已成功）

```sh
/opt/conda/bin/python -B /guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/supervise.py preflight
```

首次因新增缓存导出包装的grad模式失败，更新0；第二次在保持原入口grad模式后通过。两次运行独立日志留存于 `_runtime/preflight/`。

## 固定预算训练（已启动）

```sh
/opt/conda/bin/python -B /guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/supervise.py train
```

上层已用独立session持久调度，PID及时间见train_dispatch.json；详细内部argv/指定GPU/环境见train_launch.json，完整stdout/stderr/events/state在 `_runtime/train/`。已有分支目录时拒绝隐式重跑。不要把上述命令当作再次启动指令。

## 训练完成后的CPU验收（由supervisor在两支正常完结后运行；已执行通过；结果见ACCEPTANCE.json）

```sh
/opt/conda/bin/python -B /guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/summarize_results.py
```

CPU检查所有0..500状态、500更新/Adam计数、6个定点评价和逐UID预测数组、原Frozen权重；输出报告、趋势CSV/PNG、权重清单与两个ZIP。

## 只读查看状态

```sh
cat /guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/status.json
```

GPU UUID固定为config记录中的用户分配A100；未抢占/终止其他进程、未增加资源。原始权重与数据没有上传到外部服务。

## 本次恢复命令（已执行）

```sh
/opt/conda/bin/python -B /guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/recovery_supervise.py gate
/opt/conda/bin/python -B /guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921/recovery_supervise.py train
```

gate已通过且更新0；train从204接续，具体完成状态以根status.json与recovery_204/train_runner_result.json为准。不要再次盲目执行上述命令。
