# 512维 step3000：后期LR配对续训

共同父checkpoint：`/guohaoran/nexus_fast_track/diagnostics/math00_latent512_804_fresh_20260914/checkpoint-update3000.pt`

SHA256：`06dc42726c816b69f06db353d4de505e9d8bceda864294c8bb75c7907ba0bdd6`

| 分支 | Encoder/μ | Decoder body | Edge head | Face head | 新增更新 |
| --- | --- | --- | --- | --- | --- |
| A_hold | 1e-5 | 1e-4 | 1e-4 | 1e-4 | 500 |
| B_lr03 | 3e-6 | 3e-5 | 3e-5 | 3e-5 | 500 |

实际LR由加载后的父Adam参数组LR直接乘1或0.3，保留其浮点值。两支均恢复同一完整权重、四组Adam历史矩和step计数、Python/NumPy/Torch/CUDA RNG。无新初始化、无warmup，不迁移架构。

单条完整804点mesh；latent512、Decoder hidden1024、Edge/Face各32维；math00和确定性Graph；μ路径；fully-diff Edge Soft4+Face Soft4，内部除4，无外部1/4；sampling/KL关闭；logvar专属参数冻结且不在Adam中；global clip1、wd0。没有顶点权重或蒸馏。

同一GPU顺序执行两支。新增0/100/200/300/400/500保存完整checkpoint、实际Edge和从预测Edge图枚举的实际Face验收。每步记录四组梯度、实际参数位移、loss、尺度、512通道梯度和裁剪。每支500步后停止，不自动续训。

目录：

- `A_hold/`、`B_lr03/`：`updates.jsonl`、`eval-update*.json`、`manifest.json`、`parent_baseline_verified.json`、完整checkpoint、`console.log`。
- `build_branches.py`、`parent_train.py`、两支的`changes_from_parent.diff`：从父脚本生成的窄范围续训改动。
- `verification.json`：完成后的只读权重、Adam、RNG、日志、LR、冻结和checkpoint哈希核验。
- `comparison.json`、`REPORT.md`、`comparison_curves.png`：配对结果。

`eval-update0500.json`使用第500次新增更新后的权重（累计3500）；`updates.jsonl`第500条loss使用这次更新之前的权重。Face训练pool指标不能替代实际Face验收。

服务器根目录：`/guohaoran/nexus_fast_track/diagnostics/math00_latent512_804_lr03_pair_20260914/`。
日志包不包含大型checkpoint；全部checkpoint仍在服务器上述目录。
