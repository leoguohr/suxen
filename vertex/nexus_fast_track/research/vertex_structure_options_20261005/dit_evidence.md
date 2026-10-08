# Vertex DiT 结构选择：可核查证据笔记

核查日期：2026-10-05。只核查论文、作者代码、公开配置；没有修改训练代码。本笔记供教材文档整合，层数、宽度、head 数由另一份笔记负责。

这里的直接用途是：为 NEXUS 每个父节点输出 8 个 child occupancy velocity 的 DiT，区分“论文已确定的接口”和“本地自行选择的内部结构”。其他模型多数生成连续 VAE latent，它们的存在只证明某个结构有真实先例，不证明该结构对 octree occupancy 更有效。

## 0. 当前本地实现与证据等级

当前文件：[vertex.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:262)。`_VertexBlock`：SA → CA → FFN；SA/FFN 前无 affine 的 LN，经时间独立生成 6C 的 scale/shift/gate；CA 前有 affine LN，但无时间 gate；DiT 的 SA/CA 做 QK RMS 类归一化；`_ffn` 是 4× GELU(tanh)；最后无 affine LN → Linear(C,8)；各块 modulation 和最终 Linear 零初始化。**CA 普通残差使整个 block 在初始化时并非恒等映射**。VecSet 内部的注意力没有开启 QK norm。

- **已启用**：有作者实际构造调用或公开发布配置支持。
- **源码选项**：代码明确可选，但本次未证明官方权重启用。
- **跨任务借鉴**：原任务/表示不同，迁移接口是我们的设计判断。
- **本地选择**：不能由 NEXUS 引用 DiT 推导为 NEXUS 作者设置。

NEXUS 本身明确约 2B DiT、3D RoPE、depth embedding、条件 CA；精确 block 顺序、AdaLN、QK norm、FFN、head、初始化等没有由其公开说明确定。[NEXUS v1](https://arxiv.org/html/2607.13563v1)。原 DiT 论文比较 in-context、CA、AdaLN、AdaLN-Zero 四种方案；它的最终发布 AdaLN-Zero block 只有 SA 和 FFN，没有 CA。因此“引用 DiT”并不能证明采用本地的 AdaLN-Zero+CA 组合。[DiT §3.2](https://arxiv.org/html/2212.09748v2#S3.SS2)。

## 1. 来源与固定版本

| 简称 | 论文 | 作者仓库固定 commit | 核查到的实际模型/配置 |
|---|---|---|---|
| DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | [facebookresearch/DiT @ ed81ce2229091fd4ecc9a223645f95cf379d582b](https://github.com/facebookresearch/DiT/tree/ed81ce2229091fd4ecc9a223645f95cf379d582b) | `DiTBlock`, `FinalLayer`, `DiT.initialize_weights`；发布代码选 AdaLN-Zero |
| PixArt-α | [PixArt-α](https://arxiv.org/abs/2310.00426) | [PixArt-alpha/PixArt-alpha @ cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892](https://github.com/PixArt-alpha/PixArt-alpha/tree/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892) | `PixArt`, `PixArtBlock`, `T2IFinalLayer`；`configs/PixArt_xl2_internal.py` |
| Hunyuan3D-2.1 | [Hunyuan3D 2.1](https://arxiv.org/abs/2506.15442) | [Tencent-Hunyuan/Hunyuan3D-2.1 @ 82920d643c0dc2f7bfd7255f45f62d386edfe60c](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/tree/82920d643c0dc2f7bfd7255f45f62d386edfe60c) | `HunYuanDiTPlain`；[官方 HF config @ 07d6dc9694e0ea942683bf6e3e374887d9f5b054](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml) |
| TRELLIS | [Structured 3D Latents](https://arxiv.org/abs/2412.01506) | [microsoft/TRELLIS @ 442aa1e1afb9014e80681d3bf604e8d728a86ee7](https://github.com/microsoft/TRELLIS/tree/442aa1e1afb9014e80681d3bf604e8d728a86ee7) | [image-L SS 配置](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L4-L17) |
| TRELLIS.2 | [Native and Compact Structured Latents](https://arxiv.org/abs/2512.14692) | [microsoft/TRELLIS.2 @ 75fbf0183001ed9876c8dbb35de6b68552ee08bd](https://github.com/microsoft/TRELLIS.2/tree/75fbf0183001ed9876c8dbb35de6b68552ee08bd) | [shape 配置](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L4-L18)，SS 配置也开启 share_mod/scaled/self+cross QK |
| TripoSG | [TripoSG](https://arxiv.org/abs/2502.06608) | [VAST-AI-Research/TripoSG @ fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c](https://github.com/VAST-AI-Research/TripoSG/tree/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c) | `TripoSGDiTModel`；[官方 HF config @ 2c1c516d22d58db486a058d98d31bb6177344e06](https://huggingface.co/VAST-AI/TripoSG/blob/2c1c516d22d58db486a058d98d31bb6177344e06/transformer/config.json) |
| Meta MeshFlow | [MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer](https://arxiv.org/abs/2606.04621) | [facebookresearch/meshflow @ 55f56f60e1bbf98d1c1991670ac998094d5f59ae](https://github.com/facebookresearch/meshflow/tree/55f56f60e1bbf98d1c1991670ac998094d5f59ae) | `MeshFlowDiTBlock` 的 7C 调制是直接构造；HF config 下载返回 401，本笔记不据此声称 RMS 主干/DropPath/zero_init 等选项已用于发布权重 |
| 3DShape2VecSet | [3DShape2VecSet](https://arxiv.org/abs/2301.11445) | [1zb/3DShape2VecSet @ 8df9b7a55c42d4dcad152294755250a2ab1e34e5](https://github.com/1zb/3DShape2VecSet/tree/8df9b7a55c42d4dcad152294755250a2ab1e34e5) | `models_ae.py` 的 GEGLU、PreNorm 和 decoder DropPath；这部分为 AE，不是发布 DiT |

注意仓库同名冲突：`qiisun/MeshFlow`（论文 2606.23489）与 **Meta 的 `facebookresearch/meshflow`（2606.04621）是两个不同项目**。本笔记主表的“Meta MeshFlow”只指后者。前者虽然有 SwiGLU/GEGLU 源码选项，但其 `DiTLayer` 默认实参为 SiLU；不能混用来证明 Meta 模型用了这些选项。

## 2. Block 组织与 CA 分支：三个真实方案

| 可选方案 | 精确结构及状态 | octree 迁移与限制 |
|---|---|---|
| 平铺、CA 前做 LN：TRELLIS 系 | 每块 SA → CA → FFN；SA/FFN 做时间调制，CA 输入有 affine LN，CA 无 gate。已启用。与本地最接近。 | 保留 `[B,N,C]`、父节点坐标/有效 mask 和条件 `[B,M,Ccond]`；不需要引入其 occupancy-VAE。参考对象仍是连续 latent。 |
| 平铺、CA 不另做 LN：PixArt | SA → CA → FFN；`cross_attn(x,y,mask)` 直接读 SA 残差结果，CA 没有前置 LN，也无 AdaLN gate。已启用。 | 改的是 CA 输入预处理；仍要保留本地父节点、padding 和 VecSet 条件接口。PixArt 的二维 patch/unpatchify 不能迁移到父节点序列。 |
| U 型长 skip：Hunyuan3D/TripoSG | block 内仍 SA → CA → FFN；后半网络把早期 block 输出与当前特征 concat，Linear(2C,C) 后 LN，再进入当前块；另有 time token。已启用。 | 需要确定同一父节点 token 顺序的长 skip 对齐；time token 不是真实空间父节点，必须单独处理 mask/RoPE。原模型不提供跨八叉树深度的坐标定义。 |

证据：[TRELLIS.2 `ModulatedTransformerCrossBlock` L104–157](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L104-L157)、[PixArt `PixArtBlock` L25–54](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L25-L54)、[Hunyuan block L378–405](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L378-L405)、[Hunyuan skip 存取 L657–666](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L657-L666)。

**排列证据缺口：**此次优先模型的兼容 CA block 都是 SA → CA → FFN。没有查到这些作者发布了 CA → SA → FFN 或 SA → FFN → CA 的替代证据，不能为了凑两种排列而编造。原 DiT 的 in-context/纯 AdaLN 模型可只有 SA → FFN，但直接删除 CA 会偏离 NEXUS 明确的条件接口，只能当作更远的设计参照。

## 3. 时间调制：独立、共享、time token

令 `A(x;s,b)=LN(x)·(1+s)+b`，gate 在分支输出进入残差前相乘。

| 可选方案 | 结构及实际来源 | octree 迁移与限制 |
|---|---|---|
| 每块独立 AdaLN | 每块 `SiLU→Linear(C,6C)` 产生 SA/FFN 各自 shift、scale、gate；TRELLIS v1 image-L 的 `share_mod=false` 默认实际生效；本地相同大类。 | 输入仍是时间向量；保留 CA。每层独立投影比共享版本有更多参数，参数规模核算需重算。 |
| AdaLN-single | 模型只计算一次 `M(t)∈R^{6C}`，每块加自己的可学习 `E_l∈R^{6C}`；PixArt 与 TRELLIS.2 已启用。 | 改的是层间调制参数共享；每块仍有独立 attention/FFN，不能误读为整个 block 共享权重。 |
| 时间 token | 把时间向量变成一个额外 token 拼到序列，SA 传播时间信息；Hunyuan/TripoSG 发布模型已启用，没有每块 6C AdaLN。 | 添加和最后删除时间 token；需给它特殊 RoPE/位置处理，避免把它当 octree 父节点。它可与 CA 共存。 |

证据：[TRELLIS v1 模型默认 L68–73](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/sparse_structure_flow.py#L68-L73)、[TRELLIS.2 shared block L132–144](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L132-L144)、[PixArt 一次 t_block L121–135](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L121-L135)、[TripoSG 时间 token L664–718](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L664-L718)。论文解释：[PixArt §2.3](https://arxiv.org/html/2310.00426v3#S2.SS3)。

全局条件可单独作为另一条轴：原 DiT `c=time_embedding+label_embedding` 调制 SA/FFN；如果把 label 换成池化 VecSet，需要我们新增 masked pooling 与维度投影，原 DiT 没有验证这一点。[DiT forward L233–246](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L233-L246)。Hunyuan 源码有 attention pooling 条件分支，但官方 2.1 配置 `use_attention_pooling=false`，不能声称发布版已用它。[源码 L647–651](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L647-L651)。

## 4. CA 是否 gate：三个不同的机制

| 可选方案 | 来源证据 | octree 迁移与限制 |
|---|---|---|
| 普通 CA 残差 | 本地/TRELLIS：`x←x+CA(LN(x),cond)`。SA/FFN gate 为零不等于整块恒等。 | 保留现有机制；条件路径不受时间 gate 控制。 |
| 为 CA 增加时间 gate | Meta `MeshFlowDiTBlock` 固定 `Linear(C,7C)`，多出的 C 是 `gate_cross`；CA 前仍普通 norm，没有额外 CA scale/shift。`_zero_init_adaln_modulation` 无条件把7C输出置零。 | 将本地6C变7C，并将第7组乘 CA 输出；可保留空间接口。但原模型的条件可为图像，非 NEXUS 联合训练 VecSet 的效果证明。 |
| 不加 gate，CA 输出投影零初始化 | PixArt 的 CA 无 gate；初始化把 `cross_attn.proj.weight/bias` 置零。 | 只使初始化时 CA 分支为零；它不是每次 forward 都随 t 改变的 gate。适合作为独立初始化候选，与7C机制不可混称。 |

证据：[Meta 7C 构造 L538–539](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L538-L539)、[Meta 分支 L578–607](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L578-L607)、[PixArt 初始化 L203–210](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L203-L210)。Meta 若启用 U 型 skip，整个 block 的 skip merge 本身不受这三个 gate 覆盖，故也不能一概称整个 block 恒等。

## 5. QK norm：位置、形式、参数共享分别选择

QK norm 位于 Q/K 线性投影之后、点积之前；主干 LN 位于 attention/FFN 的输入，两者不是同一个操作。

| 可选方案 | 已启用证据 | octree 迁移与限制 |
|---|---|---|
| SA/CA 都不加 QK norm | PixArt `WindowAttention.forward`、`MultiHeadCrossAttention.forward` 从 QKV 直接进 attention；作者实际 `PixArtBlock` 调用它们。 | 删除两处 Q/K normalize，保留主干 LN、RoPE 与 mask。不能由图像结果断言它在长 octree 序列稳定。 |
| 只对 SA 加 | TRELLIS image-L：`qk_rms_norm=true`，cross 默认为 false。 | SA=true、CA=false 两个独立开关；不能把 v1 概括为所有 attention 都开启。 |
| SA/CA 都加 RMS 类 | TRELLIS.2 发布配置两项均 true；Hunyuan/TripoSG 同样都启用 RMS QK。 | 本地目前属于这一类；仍需选择数值公式和 gain 参数化，不可直接声称实现逐位相同。 |
| QK LayerNorm | Hunyuan/Meta 源码可通过 `qk_norm_type` 从 RMS 切换 LN；**源码选项，未证明发布启用**。 | LN 还减去 Q/K 均值，会改变注意力方向；不可与 RMS 当同义词。 |

证据：[PixArt QKV→attention L109–125](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L109-L125)、[PixArt CA L29–57](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L29-L57)、[TRELLIS v1 发布配置 L13–17](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L13-L17)及前述默认、[TRELLIS.2 发布配置 L15–18](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L15-L18)、[Hunyuan norm 选择 L571–574](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L571-L574)。

**同名 RMS 类仍有两个细分方案：**本地 `_HeadRMSNorm` 是 `x/√(mean(x²)+1e-6)`，可学习 gain 为 `[heads,head_dim]`；TRELLIS `MultiHeadRMSNorm` 是 `F.normalize(x.float(),dim=-1)·√head_dim·gamma[heads,head_dim]`，epsilon规则不同；Hunyuan 使用 `nn.RMSNorm(head_dim,eps=1e-6)`，gain 是 `[head_dim]`，在 heads 之间共享。[TRELLIS.2 L9–16](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/modules.py#L9-L16)、[Hunyuan L161–168](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L161-L168)。

TRELLIS.2 的 SA 顺序是 QK norm → RoPE → attention；CA 不用该3D RoPE。[L66–99](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/modules.py#L66-L99)。本笔记不把任何 QK 改动解释成对 Topology AE embedding 的归一化。

## 6. FFN：激活、gating、宽度与专家是不同轴

| 可选方案 | 精确结构与来源状态 | octree 迁移与限制 |
|---|---|---|
| Dense GELU，4C | `Linear(C,4C)→GELU→Linear(4C,C)`；本地/DiT/PixArt/TRELLIS v1。DiT/PixArt 是 tanh 近似，Hunyuan dense MLP 是 `nn.GELU()`。 | 与 token 表示无关，可保持全部 octree接口。只改 GELU 近似形式并不等于换了 FFN 类型。 |
| Dense GELU，更宽中间层 | TRELLIS.2 官方 `mlp_ratio=5.3334`，C=1536 时 `int(C·ratio)=8192`，依旧是 GELU，不是 SwiGLU。 | 单独选择 FFN 宽度；保持总参数目标时需重新核算整个 DiT。 |
| GEGLU | 3DShape2VecSet AE `Linear(C,8C)` 拆成 u/v，各4C；`u·GELU(v)` 后 `Linear(4C,C)`。真实已调用。 | 可作为局部 FFN 替换；但来自 AE。相同“4×中间宽度”时，两路门控使矩阵参数约12C²而普通FFN约8C²，不能当作等规模。 |
| 稀疏专家 FFN + shared expert | Hunyuan3D-2.1 发布 `num_moe_layers=6,num_experts=8,moe_top_k=2`；最后6层8个路由专家选2个，加1个始终执行的共享专家；每个专家为 GELU FFN。 | 改动大于换激活；保留 token I/O 但引入路由与容量分配。仅证明结构可查，不建议自动扩大当前任务范围。 |

证据：[DiT FFN L105–116](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L105-L116)、[TRELLIS.2 `FeedForwardNet` L49–59](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/blocks.py#L49-L59)、[3DShape2VecSet `GEGLU/FeedForward` L51–68](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L51-L68)、[Hunyuan `MLP` L126–135](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L126-L135)、[MoE 层构造 L362–376](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L362-L376)、[shared expert L112–153](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/moe_layers.py#L112-L153)。

**常见误读修正：**TripoSG `DiTBlock` docstring 有 GEGLU 字样，但 `TripoSGDiTModel` 实际调用硬编码 `activation_fn="gelu"`；不能把注释当发布配置。[L440–462](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L440-L462)。

## 7. 主干归一化：PreLN 的形式与 epsilon

| 可选方案 | 来源/状态 | 迁移差异 |
|---|---|---|
| PreLN，无 affine，外部 AdaLN scale/shift | DiT/TRELLIS 的 SA/FFN；`eps=1e-6`。 | 当前本地方案；affine由时间函数提供，静态LN不再重复learnable scale/bias。 |
| PreLN，有 affine，无每块 AdaLN | Hunyuan/TripoSG 的 SA/CA/FFN；Hunyuan eps=1e-6，TripoSG 构造明确 eps=1e-5。 | 需要配套 time token 或其他时间通道；不能简单删掉AdaLN却不给时间入口。 |
| PreRMSNorm | Meta `MeshFlowDiT` 的 `norm_type!="layer"` 分支；代码 builder 默认 `rms`，但发布bundle配置本次不可读，故只列源码支持。 | 不减均值；静态gain及外部调制需明示。不能从QK使用RMS推导主干也RMS。 |

证据：[DiT L105–121](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L105-L121)、[Hunyuan L331–354](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L331-L354)、[TripoSG norm 构造 L173–202](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L173-L202)及其上节实际调用、[Meta norm 选择 L714–716](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L714-L716)、[builder L870–884](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L870-L884)。

本地 `_LayerNorm`显式在float32中算统计再转回；TripoSG明确使用`FP32LayerNorm`。epsilon 的 `1e-6` 与 `1e-5` 有真实配置先例，可单列为小选择，不应推导哪一个对 NEXUS 更好。所核相关发布主干均为 PreNorm；此处不凭空增加 PostNorm 建议。

## 8. 输出头：至少三种独立选择

所有兼容 NEXUS 的候选最后都必须得到 `[B,N,8]` 的 velocity，不是16维 DDPM 噪声+方差，不是3维坐标，也不是 latent64。

| 可选方案 | 结构/实际来源 | octree 迁移与限制 |
|---|---|---|
| 无条件、无 affine 的 LN → Linear | 本地；TRELLIS.2 `F.layer_norm(features,features.shape[-1:])` 后 `out_layer`（它使用函数默认epsilon，不等于本地显式1e-6）。 | 输出投影改为8；保持当前无需额外条件向量的接口。 |
| 条件化 final AdaLN → Linear | 原 DiT `FinalLayer` 对条件做 `SiLU→Linear(C,2C)` 得到final shift/scale；无 affine LN再调制。已启用。 | 给输出头传时间/全局条件；替换 patch输出为8，无需unpatchify。 |
| 共享风格 final调制 → Linear | PixArt `T2IFinalLayer` 用 `learned_table[2,C]+time[:,None,:]` 得shift/scale；没有独立2C MLP。已启用。 | 时间向量宽度必须为C；保持8维输出。它与原DiT final AdaLN参数化不同。 |
| 有 affine LN → Linear | Hunyuan `FinalLayer`，先LN，再去掉time token，最后Linear。已启用。 | 若只借输出头、仍用AdaLN时间，不应凭空删首个父节点；仅time-token架构才移除额外token。 |

证据：[TRELLIS.2输出 L193–199](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L193-L199)、[DiT FinalLayer L125–142](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L125-L142)、[PixArt T2IFinalLayer L172–188](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L172-L188)、[Hunyuan FinalLayer L450–465](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L450-L465)。

## 9. 初始化：不能把三个“零”混在一起

| 可选方案 | 准确初始化与来源 | 含义/迁移限制 |
|---|---|---|
| 每块调制输出为零，最终投影为零 | 原DiT的6C modulation、final2C modulation、final Linear均零；普通Linear Xavier，time MLP std0.02。当前本地只有SA/FFN modulation和finalLinear零，timeMLP也走Xavier。 | 原DiT完整块恒等依赖它只有受gate控制的SA/FFN。本地未gate的CA破坏该前提。 |
| CA projection为零，finalLinear为零 | PixArt仅将CA输出投影与finalLinear置零；共享`t_block`权重std0.02；每块6C table随机初始化。 | 不是所有gate=0，不是整个block恒等；与本地R1历史CA-out zero是类似初始化手段，但不能把R1训练状态当作重新初始化。 |
| 按网络深度缩小残差投影 | TRELLIS.2 `initialization="scaled"`：普通Linear std√(2/(5C))；SA/CA输出与FFN第二层std1/√(5LC)；输入std1/√Cin；time MLP std0.02；共享modulation最后Linear和final输出零。 | 发布开启share_mod时，每块6C偏置仍随机，故初始分支并不全零；独立于它是否用3D RoPE。 |
| 为SA/FFN/CA全部gate零初始化 | Meta7C modulation在block构造时直接zero；是否额外zero attention/FFN输出是另一个`zero_init`开关。 | 局部残差分支为零；若有U型skip其merge仍改变输入。不能将`zero_init=false`误解为7C gate未zero。 |

证据：[DiT初始化 L182–216](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L182-L216)、[PixArt初始化 L176–210](https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L176-L210)、[TRELLIS.2 scaled初始化 L128–167](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L128-L167)、[TRELLIS.2随机block偏置 L132–144](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L132-L144)、[Meta无条件gate-zero L475–477](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L475-L477)、[Meta额外zero开关 L770–777](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L770-L777)。

输出头是否为零，可独立列两个选择：原DiT/PixArt/TRELLIS明确zero；Hunyuan `HunYuanDiTPlain.__init__`直接构造普通 `nn.Linear`，本次核查类内没有显式zero final的步骤（后续加载发布权重当然会覆盖构造值）。后者只能叫“发布推理构造代码未显式zero”，不能据此补写未公开的历史训练初始化。

## 10. QKV bias 与 Dropout/DropPath 小项

| 轴 | 选择1 | 选择2 | octree边界 |
|---|---|---|---|
| QKV projection bias | 本地/DiT `qkv_bias=true` | Hunyuan官方配置、TripoSG实际调用 `qkv_bias=false` | 仅改变QKV线性层是否有bias；输出投影仍可有bias。不要写“整个attention无bias”。 |
| 残差随机丢弃 | 本地/原DiT FFN drop=0；当前attention dropout=0 | 3DShape2VecSet AE decoder的SA/FFN `DropPath(0.1)`已启用；Meta代码另支持逐层调度 `drop_path_rate`，发布启用未核实 | DropPath丢整个样本的残差支路，不是丢attention概率。VecSet来源在AE decoder，不能说原NEXUS/原VecSet条件encoder用了0.1。 |

证据：[DiT bias/drop L108–112](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L108-L112)、[TripoSG实际bias L453–461](https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L453-L461)、[3DShape2VecSet decoder构造 L204–225](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L204-L225)、[其FeedForward DropPath L57–68](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L57-L68)、[Meta逐层dpr L737–755](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L737-L755)。

## 11. 覆盖与未确认项

| 当前本地选择 | 可查替代覆盖 |
|---|---|
| Block组织/CA前归一化 | TRELLIS平铺有LN；PixArt平铺无CA-LN；Hunyuan/Tripo U型skip |
| 每块时间AdaLN | 独立6C；共享6C+每块偏置；time token |
| CA普通残差 | 7C时间gate；CA输出zero初始化（不同轴，已明确） |
| QK RMS self+cross | 都关；只SA；SA+CA；LN源码选项 |
| QK gain参数化 | 每head独立；跨head共享；数值epsilon差异另列 |
| FFN4×GELU | 更宽GELU；GEGLU；带shared expert的MoE |
| PreLN/eps | 无affine+AdaLN；有affine+time-token；RMS源码选项；1e-6 vs 1e-5 |
| final LN+Linear8 | final AdaLN；PixArt table+time；有affine LN |
| zero初始化 | per-block调制zero；CA-output zero；scaled残差；全部3分支gate-zero |
| bias/no dropout | bias true/false；无drop vs AE decoder DropPath借鉴 |

尚未确认：NEXUS作者这些内部细项；不同排列顺序的第二种兼容作者实例；Meta受限发布config中具体选项；把上述模块迁移到NEXUS octree后的效果。此处没有把源码可选当已发布、把AE模块当DiT、或把连续latent经验当二值占用生成结论。

本次新取回的只读源码和公开配置在相邻 `source_snapshots/`；已有本地checkout HEAD均经本次git核验。引用是固定commit，不代表最新分支。未下载权重、未执行模型。
