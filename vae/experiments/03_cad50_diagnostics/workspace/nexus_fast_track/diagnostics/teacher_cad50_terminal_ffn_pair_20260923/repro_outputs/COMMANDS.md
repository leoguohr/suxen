# 命令与证据索引

所有服务器命令在独立实验目录 `/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923` 执行。SSH凭据不保存在交付目录。

## 已执行启动核验

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python audit_control/audit_reused_control.py
CUDA_VISIBLE_DEVICES='' PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /opt/conda/bin/python H_terminal_ffn/test_terminal_ffn.py
/opt/conda/bin/python launch_h.py
```

Control CPU核验有两次成功运行，后一次补充设备与时间元数据。最终规范结果为 `CONTROL_REUSE_AUDIT.json`；没有重复Control训练。CPU合成测试2次toy Adam更新，真实实验0次。

`launch_h.py` 调用已安装skill的run-train监督器，实际argv、GPU UUID、runner哈希和PID记录在 `launch.json`。实际子命令：

```bash
/opt/conda/bin/python -u H_terminal_ffn/train.py --mode terminal_ffn
```

H固定使用双卡主机GPU0，UUID `GPU-0ae7719a-74e5-08ae-75de-0a6205381365`。环境：OMP/MKL/OPENBLAS线程数1，`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`、`PYTHONDONTWRITEBYTECODE=1`、`RIGORPILOT_LESSONS=0`。预算100；supervisor timeout7200秒是安全时限，不是额外训练预算。

本地skill README intake实际执行，但没有用其建议venv或安装依赖；`intake/`和`train_outputs/`中的“未执行”是启动前历史阶段记录。实际训练状态以run-train `_runtime/train/`、H `updates.jsonl`、`complete.json`及最终根 `status.json` 为准。

## 完成后的验证/收集入口

这些命令是否完成，以相应JSON和日志为证，不能仅凭此命令清单推断成功：

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /opt/conda/bin/python cold_verify.py
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python audit_control/audit_treatment.py
/opt/conda/bin/python collect_evidence.py
python3 summarize_results.py
python3 package_delivery.py
```

冷加载评价不做optimizer更新。`COLD_VERIFY.json`核对最终完整model/Adam/RNG和重新安装FFN hook后的真实网络全50输出。服务器大型checkpoint不重复下载；小证据ZIP的下载收据包含每文件SHA256。
