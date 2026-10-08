# CAD50 完整状态续训入口

每支 2000→10000，新增 8000 次 5-mesh 更新；LR 恒定 1e-4，无 warmup。
评价与轮转点锁定为 2000/3000/4000/5000/7500/9000/10000。
先 Fourier post/pre 同卡并行至共同点并退出，再 XYZ post/pre；随后进入下一点。
每进程显式 UUID、显存份额 .44；不选择空卡，不按中途成绩淘汰。

部署后在**新的远程目录**运行（以下命令由主代理执行，本工作未连接服务器）：

```bash
cd /ssdwork/guohaoran/nexus_fast_track/diagnostics/cad50_fourier_graph_continue10000_20261005
/opt/conda/bin/python -B -u run_pairs.py \
  --source /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01 \
  --parent-base /ssdwork/guohaoran/nexus_fast_track/diagnostics \
  --gpu-uuid GPU-80f199d2-afab-fad3-824d-6d2482a4c882 \
  --budget-updates 10000
```

单支零更新服务器验收（建议独立 audit_runs，避免占用生产 runs）：

```bash
/opt/conda/bin/python -B -u train_continue.py \
  --arm Fourier_LN_post \
  --source /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01 \
  --gpu-uuid GPU-80f199d2-afab-fad3-824d-6d2482a4c882 \
  --outdir "$PWD/audit_runs/Fourier_LN_post" \
  --audit-only
```

另三支改变 arm 和 outdir。入口自行逐 UID 复核父评价；不修改或依赖主代理的 prelaunch 文件。
Fourier 每次进程恢复都对下一完整 5-mesh 累积的逐 mesh FP32 loss 原始字节及全部 412 梯度
原始字节做 recompute/retain 等价检查。失败停止，无实现 fallback。XYZ 复用已保存历史证据。
全部恢复比较模型 415 entries、flat 412 参数映射、所有 AdamW tensors/group/step，及四类 RNG；
GPU UUID 仅在 runtime 元数据中，科学配置不包含设备身份。

故障后不会自动重启。检查 failure.json 和各 attempt 的审计日志，再提供**明确边界文件及 SHA**：

```json
{
  "Fourier_LN_post": {"path": "/.../runs/Fourier_LN_post/boundary-xxxxx-....pt", "sha256": "实际 SHA256"},
  "Fourier_LN_pre": {"path": "/.../runs/Fourier_LN_pre/checkpoint-xxxxx.pt", "sha256": "实际 SHA256"}
}
```

在运行命令后追加 `--resume-map /明确路径/resume-map.json`。map 须覆盖每个已创建且未完成的
分支；未创建分支直接从父开始，完成分支须从 map 省略。checkpoint 与连续日志必须在同一
更新边界；拒绝静默 replay/reset。SIGTERM 请求当前更新完整提交后保存可续训边界，停止整对；
积累或 Adam 中的异常现场标记 `resumable:false`。SIGKILL/掉电只能使用仍匹配日志的已保存边界，
不自动修剪日志。控制器和所有子进程继承锁，防止孤儿进程仍在运行时重复启动。

阶段 checkpoint、best.pt、final.pt 均保留完整可续训状态；best 与 final 在同文件系统使用
原子替换的硬链接，后续 checkpoint 不原位修改。best 选择依次按 joint strict、真实 Face F1、
Edge F1，完全同分保留较早点；best 仅保存、不影响各支训练预算。
完整评价本体未改；derived_summary.json 汇总 CAD50/固定 large16 的 Edge/Face TP/FP/FN/F1、
strict UID、候选总数、GT candidate coverage、FN missing/present。日志 loss 是更新前，step 是更新后；
每次恢复后前 3 步与每 100 步保留实际 FP32 位移、更新前参数范数、相对位移。

CPU 检查：`CUDA_VISIBLE_DEVICES='' python -B test_continue_cpu.py`。
本机 NumPy 实际执行了生产 PCG64 分组与负例代码（仅 Tensor 包装替换为 identity）；
本机无 Torch，完整模型/Adam/RNG CPU 项明确跳过。服务器仍须验证实际恢复、两格 Fourier gate、
同卡双进程、阶段退出/显式恢复和停止边界。CODE_SHA256.json 绑定全部 Python 文件；修改后须
重新审查并更新 manifest，不允许悄悄更换已运行代码。
