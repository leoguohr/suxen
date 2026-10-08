# DiT 层宽与注意力头数依据

核查日期：2026-10-05。只读取源码及发布配置。

| 配置 | blocks | hidden | heads | 每头维度 | 证据 |
|---|---:|---:|---:|---:|---|
| 当前 R1 | 36 | 1536 | 12 | 128 | [VertexDiT](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:296) |
| 原 DiT XL | 28 | 1152 | 16 | 72 | [DiT_XL_2](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L328-L335) |
| TRELLIS image L | 24 | 1024 | 16 | 64 | [官方 SS 配置](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L4-L17) |
| TRELLIS.2 shape | 30 | 1536 | 12 | 128 | [官方 shape 配置](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L4-L18) |
| Hunyuan3D 2.1 shape | 21 | 2048 | 16 | 128 | [发布配置](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml#L1-L19) |

- `36×1536×12`：未在本轮核验的外部发布配置中找到原样组合。1536 宽和 12 heads 有 TRELLIS.2 依据；36 blocks 是本地容量配置。
- 上述配置包含各自不同的 block、条件宽度和输出接口。TRELLIS.2 用共享调制及 FFN ratio 5.3334；Hunyuan 发布模型包含末六层 MoE 和 U-shaped skip。不能仅替换层宽就沿用原模型参数量标签。
- NEXUS 约 2B 参数约束：先确定 block、调制共享和 FFN，再对适配后的完整 DiT 统计参数。
- 论文：[DiT](https://arxiv.org/abs/2212.09748)、[TRELLIS](https://arxiv.org/abs/2412.01506)、[TRELLIS.2](https://arxiv.org/abs/2512.14692)、[Hunyuan3D 2.1](https://arxiv.org/abs/2506.15442)。
- 当前冻结 `vertex.py` SHA256：`39491f94fadec84797b59a92d969dd923353abeea29a91a02e1a0f904ae1f386`；与 E2 C 保存的 code_sha256 清单一致。
