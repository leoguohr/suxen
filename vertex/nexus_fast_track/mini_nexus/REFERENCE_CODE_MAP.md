# 官方代码参考映射

本文件固定本次阅读过的代码版本，避免以后上游仓库变化后行号失效。

## 版本

- Hunyuan3D-2.1：`82920d643c0dc2f7bfd7255f45f62d386edfe60c`
- MeshFlow：`b78e2f6ee317056ec820e8d7cecc61a8b62fb441`
- 3DShape2VecSet：`8df9b7a55c42d4dcad152294755250a2ab1e34e5`
- Nexus：论文 `arXiv:2607.13563v1`；截至本次实现未发现作者官方代码或权重。

## VecSet：condition encoder 参考

| 要理解的内容 | 本地文件 | 关键位置 |
|---|---|---|
| FPS 输入相关 anchors | `external_references/3DShape2VecSet/models_ae.py` | `AutoEncoder.encode()` |
| Fourier xyz embedding | 同上 | `PointEmbed` |
| anchor → full points cross-attention | 同上 | `cross_attend_blocks` 与 `encode()` |

mini-Nexus 保留 `xyz + normal` 六维条件，以纯 PyTorch 的确定性 FPS 替代
`torch_cluster.fps`，从而不增加编译扩展依赖。

## Hunyuan3D-2.1：应当读哪些位置

| 要理解的内容 | 本地文件 | 关键位置 |
|---|---|---|
| 训练配置总入口 | `external_references/Hunyuan3D-2.1/hy3dshape/configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml` | `model`、`first_stage_config`、`cond_stage_config`、`denoiser_cfg`、`scheduler_cfg` |
| 线性 flow 训练 | `.../models/diffusion/transport/transport.py` | `training_losses()`，约 158–200 行 |
| velocity ODE | 同上 | `get_drift()`，约 202–233 行 |
| 推理循环 | `.../hy3dshape/pipelines.py` | 条件编码、latent 初始化、scheduler step，约 590–656 行 |
| latent 到 mesh | `.../models/autoencoders/model.py` | `latents2mesh()`，约 211–216 行；`ShapeVAE` 从约 238 行开始 |
| 图像条件编码 | `.../models/conditioner.py` | `SingleImageEncoder` / `DinoImageEncoder` |
| DiT 主干 | `.../models/denoisers/hunyuandit.py` | `HunYuanDiTPlain` |

读代码时抓住一条主线：

```text
image → DINO condition
Gaussian latent → flow DiT + Euler steps → clean ShapeVAE latent
→ ShapeVAE volume decoder → implicit grid logits
→ Marching Cubes surface extractor → mesh
```

## MeshFlow：应当读哪些位置

| 要理解的内容 | 本地文件 | 关键位置 |
|---|---|---|
| triangle soup | `external_references/MeshFlow/datasets/mesh_dataset.py` | `tokenize_mesh()`，约 247–263 行 |
| Nested OT | `external_references/MeshFlow/utils/ot_utils.py` | `optimal_sum_numpy()`，约 7–36 行 |
| flow 公式和 loss | `external_references/MeshFlow/flow_matching.py` | `LinearPath` 与 `training_losses()`，约 20–148 行 |
| ODE 采样 | 同上 | `sample()`，约 150–173 行 |
| EquiDiT | `external_references/MeshFlow/models/equidit.py` | `XEmbedder`、`DiTLayer._forward_v3()`、`DiT.forward()` |
| 训练接线 | `external_references/MeshFlow/train.py` | 搜索 `training_losses`、`coords`、`noise` |
| triangle soup 转 mesh | `external_references/MeshFlow/inference.py` | 搜索 `faces`、`merge_vertices`、`export` |

主线是：

```text
(V,F) → vertices[faces] → [F,3,3] triangle soup
Gaussian triangle soup → Nested OT 对齐真实面和噪声面
→ flow-matching EquiDiT → clean triangle soup
→ 每三个点建面 → 合并重合顶点/去重 → indexed mesh
```

## mini-Nexus 如何借鉴，而没有照抄

| mini-Nexus 部分 | 参考思想 | 为什么没有直接复制 |
|---|---|---|
| `flow.py` | 两个项目都使用的线性 flow + velocity MSE + ODE | 数学公式很短，独立实现更适合教学和测试 |
| `models.py` | Hunyuan 的条件 cross-attention、3DShape2VecSet 的 FPS VecSet、Nexus 的 3D RoPE | 原模型规模和依赖不适合 pilot，且 Nexus token 语义不同 |
| `octree.py` | Nexus 论文的 8-child multi-hot | Hunyuan、MeshFlow 都没有这个目标 |
| `topology.py` | Nexus 论文的 topology AE 和 Spacetime Interval | 两个参考仓库都没有 Nexus 第二阶段 |
