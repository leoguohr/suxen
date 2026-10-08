# Phase 0.5 只读核对（2026-09-26）

本轮未修改模型代码、未启动训练、未使用 GPU。包内 `frozen_code/`、`d15_code/`、`scripts/` 共 156 个已登记 Python 文件以及 D2 重放结果和 D15 表示审计的 SHA256 均与 `FILE_SHA256.json` 中的原记录一致。本轮重新运行了 CPU 纯表示往返；没有重新读取服务器上的大 checkpoint 或重新运行 D2 模型。

## 1. 原 D2 R1 基线

`outputs/d2_baseline/SUMMARY.json` 与 `evaluation.json` 记录了此前使用真实 D2 checkpoint 的完整模型重放：16 个种子 × 2 个固定点云条件 = 32 条完整树，原生整数顶点与全部 9 层均为 32/32 正确；288/288 层的父格、实际噪声与预测格子逐元素匹配原终验。原命令在 `outputs/d2_baseline/baseline_command.json`；生成出口在 `scripts/native_export.py`。这些是既有真实运行证据，不是本轮新 GPU 运行。

原 D2 配置：`max_depth=9`，DiT `num_blocks=36`、`hidden_dim=1536`，条件维度 `2048`，VecSet 8 层。目标为八叉树逐层 8 子格占据，训练使用线性路径的 masked velocity MSE；采样从噪声 t=0 到数据 t=1，每层 20 步 Euler，阈值 0.5。代码见 `d15_code/mini_nexus/training.py`、`flow.py`、`frozen_code/mini_nexus/vertex_evaluation.py`；D15 模型与原 D2 共享结构参数，差异单独见下文。

原 D2 保存代码 `frozen_code/scripts/train_vertex_d2.py` 第 305–306 行的 checkpoint 键为 `model`（不是 `model_state_dict`）、`optimizer`、`config`、`step`、`d2_update`、`torch_rng`、`cuda_rng`。无 `scheduler` 键，无 Python/NumPy RNG 键。包内不含大权重及 Adam 张量；保存格式和既有重放不能当作本轮已重新验证权重本体可用，更不能只凭历史路径无缝续训。

## 2. depth 15 不是仅改输出尺寸

原 D2 loader 使用 9 层预制量化数据。独立 D15 入口 `scripts/encode_d15.py` 从原浮点 GT 重新量化，生成两个训练物体的 15 层标签；`scripts/train_d15.py` 第 38–46、77–80 行核验实际点云条件与 D15 标签，并在 2 个物体 × 15 层之间调度 microbatch。`VertexStageSystem.forward()` 按实际父格 mask 计算 velocity MSE；`scripts/d15_evaluate.py` 通过 `native_export.export_tree(max_depth=15)` 逐层采样。RoPE 使用归一化父格中心，参考深度固定 9，见 `d15_code/mini_nexus/vertex.py` 第 358–371 行。

迁移代码 `scripts/d15_model.py` 仅为 `flow.depth_embedding.weight` 新增 6×1536 行；其余 906 个共享参数张量保持原值，见 `evidence/migration_audit.json`。D15 新建 Adam/RNG 从 update 0 训练，不是旧 D2 Adam 状态续训。该 D15 两物体实验后来已完成至累计 6000 步，预留终验 32/32 完整树正确，证据在 `outputs/final_6000_20260926/`。这只涉及两个固定点云条件，不是 CAD50 生成成绩。

## 3. CPU 纯表示往返重算

本轮从 `point_native_xyz_gate_20260925/outputs/roundtrip/cad50_XX/arrays.npz` 的 `gt_float` 读取 CAD50 浮点 GT，调用现有 `scripts/encode_d15.py::encode` 与八叉树量化/叶中心解码，对 depth 9、15 各 50 例重算。逐例叶数量、真实量化碰撞数、RMSE、间距比值和通过标记均与 `inputs/d15/label_audit.json` 完全一致。没有 diffusion 参与，没有写回任何 GT 或预测。

| 深度 | CAD50 原评价口径通过 | 点数保持 | 真量化碰撞 | 最大匹配坐标 RMSE | 最大间距比 |
|---|---:|---:|---|---:|---:|
| 9 | 0/50 | 48/50 | cad50_00: 60；cad50_13: 8 | 0.001953125 | 1.4379548182 |
| 15 | 50/50 | 50/50 | 无 | 0.000030517578125 | 0.0906827473 |

两个当前训练物体在 depth 9 和 15 均无新量化碰撞，但原浮点 GT 各自有重复顶点记录（24→8、140→52）；不能将去重后的连续 XYZ 诊断冒充原始点数门禁通过。CAD50 depth 15 的表示通过只说明离散分辨率足够，尚无 CAD50 点云条件生成验证。

## 判断

Phase 0.5 的证据支持当时选择 D15 来解决 CAD50 的 depth 9 表示瓶颈，而且独立 D15 两物体工程已接通 loader、loss 和 sampler 并完成训练。旧 D2 闭环没有失败，因此无需进入“先修 D2”分支；当前也没有证据要求改为 leaf residual coordinate head。下一步应单独取得 CAD50 可用的真实点云条件，再做冻结生成或条件适配实验；不得用 GT 顶点充当条件。
