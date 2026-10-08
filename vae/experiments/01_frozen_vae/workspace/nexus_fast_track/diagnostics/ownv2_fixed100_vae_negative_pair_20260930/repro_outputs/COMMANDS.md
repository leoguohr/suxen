## 路径与命令

服务器根：/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_negative_pair_20260930

启动/继续整个有限流程：

```bash
/opt/conda/bin/python -B -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_negative_pair_20260930/run_pair.py
```

该入口自动对尚未完成分支选start/resume，不重置已保存状态。已有活跃进程时排他锁阻止重复启动。整段停止请求在根目录创建STOP；分支入口在更新边界保存（未完成微批组丢弃并恢复该组前RNG），不增加预算。不要同时手动启动第二份进程。

最新状态：pair_status.json；逐分支status.json、updates.jsonl、recovery-latest.json；日志A_uniform.stdout.log/B_hard.stdout.log。完整checkpoint约2.96GB，继续保存在SSD持久挂载，源父文件只读。

