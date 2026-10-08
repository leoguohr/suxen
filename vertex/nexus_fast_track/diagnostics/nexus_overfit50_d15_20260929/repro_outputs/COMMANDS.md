# 已执行命令

```bash
CUDA_VISIBLE_DEVICES=GPU-b3c81e4b-1632-ceea-f4f4-97e3e5f84d4e,GPU-0cb18edf-c41d-02e8-3b12-14d55a283546 OMP_NUM_THREADS=8 UCX_VFS_ENABLE=n PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /guohaoran/envs/nexus-algo/bin/python -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_overfit50_d15_20260929/runtime/scripts/overfit50_pipeline.py
```

数据入口：overfit50_prepare.py；CPU梯度测试：python -m torch.distributed.run --standalone --nproc_per_node=2 runtime/scripts/overfit50_test_ddp.py（CUDA_VISIBLE_DEVICES为空）。

GPU最大样本预检：相同torchrun入口运行overfit50_train.py，--preflight --updates 2 --recompute none，输出preflight_none。预检更新不计正式训练，未保留预检模型。完整启动命令记录在audit/launch.json与audit/pipeline_status.json。
