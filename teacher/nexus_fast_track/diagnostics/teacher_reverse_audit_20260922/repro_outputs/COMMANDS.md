# 实际核验命令

远端工作目录：`/guohaoran/nexus_fast_track/diagnostics/teacher_reverse_audit_20260922`。

```bash
/opt/conda/bin/python -u run_verification.py
/opt/conda/bin/python inspect_models.py
```

`fresh_runs.json`逐条记录实际argv、开始时间、完成状态及耗时。监督器依次运行最终AE的CPU重放、GPU点34567/拓扑12345串联、GPU点98765/拓扑23456串联；每次全部50条。源码原样复制，CUDA_VISIBLE_DEVICES固定为本轮单张A100的UUID。输出非空会拒绝覆盖。

本地独立复核：

```bash
/Users/luthier/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 audit_saved_predictions.py
/Users/luthier/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 audit_fresh.py
```

首次本地审计发现当前Python没有SciPy；审计脚本改为先验证最近邻双射，再按相同最优匹配评价，不修改模型。原网络推理仍使用其原SciPy/Hungarian实现。首次上传SSH断开，重新连接后以rsync补齐，并逐资产核对SHA256后才启动推理。

所有SSH密码均仅用于认证，没有写入本交付目录、脚本、命令记录或压缩包。
