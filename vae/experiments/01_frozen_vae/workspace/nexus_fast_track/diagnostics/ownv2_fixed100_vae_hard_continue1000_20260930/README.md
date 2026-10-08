# 固定100条VAE：B35220续训1000次有效更新

用户最终授权：一张A100，B35220完整状态继续，到36220停止。原1509项固定难负例和1.5F总量规则不变，不刷新。sampling、β=1e-6、全网络、Hard4、math00、两组AdamW lr=1e-4、clip=1保持，Adam和两种RNG及数据游标完整恢复。没有warmup或初始化logvar。

新增μ评价：0、100、250、500、750、1000。起点μ重新执行核对97/52 Edge、1430/98 Face及31/100联合成功。起点五组噪声复用该完整B35220的既有真实网络评价，明确标记复用，不冒称新做；末尾重新执行相同861001—861005噪声条件。训练与评价均占当前唯一A100，顺序执行。

主判据为实际Face micro-F1>=0.997，μ和每个sampling条件独立报告。严格成功与旧31条、历史28条保留/丢失继续记录，但不以100/100作为阶段通过必要条件。结果不用于推断未测试的4F方案。

服务器目录：/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930

启动或恢复同一有限运行：
```bash
/opt/conda/bin/python -B -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/run_continue.py
```
主入口排他锁防重复；自动选择start/resume；不自动重新开始失败训练，不延长预算。运行中创建根目录STOP会在完整更新边界保存并停止。移除明确的STOP请求前不可恢复。

单独恢复命令（不与主入口同时运行）：
```bash
CUDA_VISIBLE_DEVICES=GPU-d59b34c9-1810-15c1-4cc9-db2bd953b574 /opt/conda/bin/python -B -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/code/train_continue.py --config /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/config.json --mode resume
```

每20次更新保存滚动恢复点；评价点永久保留完整checkpoint。最新指针run/recovery-latest.json，逐步日志run/updates.jsonl（保存时提交），实时状态run/status.json，启动日志train.stdout.log，根流程launcher_status.json。末尾自动生成REPORT.md、results.json、逐mesh CSV、困难负例轨迹和evaluation.zip；大模型保留服务器。

启动状态不是完成结果。真实执行证据见evidence与repro_outputs。
