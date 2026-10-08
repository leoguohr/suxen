# 实际执行命令

全部命令均由本轮用户授权协议适配；不是作者官方训练命令。工作目录：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_soft4_hard4_pair_20260922`。

```sh
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python preflight_cpu.py
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python H_paper_hard4/test_hard4.py
/opt/conda/bin/python launch_h.py
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python audit_results.py
```

launch_h.py实际通过既有RigorPilot run-train执行：
```sh
/opt/conda/bin/python -u /guohaoran/nexus_fast_track/diagnostics/teacher_cad50_soft4_hard4_pair_20260922/H_paper_hard4/train.py --mode paper_hard4
```

训练子进程固定CUDA_VISIBLE_DEVICES=GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351；物理GPU1。原始argv、启动前GPU进程、PID见launch.json；完整运行时spec/state/stdout/stderr/events/resources见`_runtime/train`。资源日志是设备整体采样，不等于独占归因。

H新增100次更新；S没有新启动。CPU审计新增optimizer更新0，不调用CUDA。完整model/Adam/RNG的读取/恢复命令和固定参数保留在train.py；不使用推理权重加fresh Adam。
