# 执行命令（已部署并启动）

服务器独立根目录：/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929

```bash
CUDA_VISIBLE_DEVICES=GPU-3534263c-6584-9f34-9273-0ef6a7852fbe /opt/conda/bin/python -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/code/train_vae.py --config /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/experiment_config.json --mode start
```

恢复时同一配置使用--mode resume；不得重跑start或重置原Adam。

服务器CPU测试8/8通过。启动PID1290；初次上传元数据导致的0更新失败已单独归档，实际代码未改。step0 μ逐mesh计数已复现源AE。
