# 服务器启动20-mesh长训

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_twenty_mesh_training_preparation`

首次启动（正式训练，仅由用户执行）：

```bash
cd /guohaoran/nexus_fast_track/diagnostics/math00_twenty_mesh_training_preparation
bash start.sh
```

恢复中断训练：查看`run/latest.json`中checkpoint绝对路径，执行：

```bash
bash start.sh --resume /完整路径/checkpoint-updateXXXXX.pt
```

预算为20-mesh阶段累计10000次实际更新，恢复不会重新增加10000步。每条完整mesh一次forward/backward，累积20条后统一clip与Adam更新。不进行子图采样。

日志在`run/console-日期时间.log`；各段逐步日志、评估与checkpoint在`run/from_XXXXX/`。每200步保存权重/Adam/RNG和μ、固定噪声完整重建；指定检查点做50组噪声验收。`run/latest.json`记录最近checkpoint。

若重复启动，进程锁会阻止第二个实例。若首次启动已产生日志但未到第200步而中断，使用step00000 checkpoint恢复；续训会创建带时间标识的新日志段，不覆盖此前输出。训练遇到非有限梯度等异常会退出；启动命令打印PID只说明进程启动，应继续查看console确认进入训练。

预检查命令（不训练，不调用optimizer.step）：

```bash
/usr/bin/python -u train.py --preflight
```

预检查报告是`preflight/result.json`；运行前检查报告与脚本及配置哈希一致。修改训练入口后必须重新预检查。完整候选池位于`pools/`，来源和哈希见`pools_ready.json`。

当前不是自包含安装包：保留本服务器原诊断源码/数据路径，不能只复制此目录到其他服务器就假定可运行。
