# 新服务器最终核验（2026-09-26）

用户提供的新 SSH 端口 36910 连到主机 `9e1vcqfgim84d-0`。本轮先做只读检查：GPU0 为 `GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351`（A100 80 GB），检查时显存 0 MiB、没有 D15 训练或评估进程。挂载的持久目录已经有完整的 6000 步 checkpoint、正常退出记录、一次性终验和全部预测。因此没有重新启动同一实验，没有修改模型、loss、数据、采样协议或使用 GPU。

| 检查 | 新服务器实查 |
|---|---|
| 训练 | `training_exit.json` 退出码 0；`train.jsonl` 从 1 到 6000 共 6000 条，step 连续且 loss、梯度范数有限。训练阶段 `status.json` 为 `update_budget_complete_evaluation_deferred`。 |
| checkpoint | `/guohaoran/tmp/r1_d15_compat_20260925/resume5200_20260926/checkpoint-006000.pt` 存在，27,990,606,810 字节；本轮完整重新计算 SHA256 为 `6e332b3f8510ffdc8b053075bdf9209404ffa4af1470d33bf488b4482d9598cb`，与身份文件一致。 |
| 训练后评估 | 独立的 `post_training_evaluation_status.json` 为 `complete`，退出码 0；16 个保留种子 `92015000..92015015` × 两物体，共 32 条预测，非训练中的开发评估。 |
| 原生生成 | `nexus_2k_000105` 和 `nexus_2k_000195` 各 16/16 完整树正确；15 个深度各 32/32 整数格子集合正确，每层漏格与多余格总数均为 0。 |
| 文件完整性 | 本轮重新核验 32 份 `prediction.npz` 和 480 份 `depth-*.npz` 的逐文件 SHA256 与 ZIP CRC，512/512 通过；实际种子、UID 顺序与全部 32 行评估记录对应。 |
| 解码 XYZ | 对显式去重后的原浮点 GT，两个 UID 的 RMSE 分别为 `1.8123419761644113e-05` 和 `1.6812323908979358e-05`，32/32 通过该诊断；原浮点 GT 有重复顶点记录，原始点数门禁为 0/32，不得冒充老师 CAD50 原标准通过。 |

实际配置仍为点云＋法向条件、VecSet＋Vertex DiT、15 层逐层占据、velocity MSE、每层 20 步 Euler、阈值 0.5。结果是两个固定对象的过拟合成绩；CAD50 仅完成不带 diffusion 的表示往返，尚无 CAD50 点云条件生成或泛化成绩。旧 D2 权重文件在新服务器持久盘存在（27,990,380,250 字节），本轮未重算其完整 SHA，也未重复既有 D2 GPU 重放；既有重放证据保留在 `outputs/d2_baseline/`。

交接顺序：先读 `PHASE_0_5_READONLY_CN.md`，再读 `R1_D15_COMPAT.md` 的最终实查节和 `outputs/final_6000_20260926/verification.json`；原始训练日志、32 条完整预测及评估文件均在 `outputs/final_6000_20260926/`。大 checkpoint 不在轻量压缩包内，服务器路径和完整 SHA 已在上文及 `checkpoint_identity.json` 给出。没有将任何 SSH 凭据写入本包。
