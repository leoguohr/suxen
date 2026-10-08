# Vertex 结构覆盖补查 v2

日期：2026-10-05。只读核查，不改模型、主报告或训练。本文补充上一版遗漏，不重复已经覆盖的 VecSet、DiT blocks、RoPE、时间编码、FFN 与输出头方案。

核查对象为当前冻结的 [vertex.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:287)。SHA256：`39491f94fadec84797b59a92d969dd923353abeea29a91a02e1a0f904ae1f386`，与 [E2 C 启动配置](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/startup_evidence/e2/C/config.json:1) 的代码清单一致。该源码入口本身不是 NEXUS 官方实现。

## 一、论文未确定：应补入网络结构选择的项目

### 1. 连续8维 occupancy 如何投影到 hidden

当前：`Linear(8,1536,bias=True)`；论文明确一父格的8个值为一个 token，但未给输入投影形式或偏置。

| 候选 | 官方源码依据 | 适配边界 |
|---|---|---|
| 单层 Linear，带 bias | Hunyuan3D 2.1 `HunYuanDiTPlain.__init__` 的 [x_embedder][hy-input]；[实际发布配置][hy-config] 是64维 latent 输入、2048 hidden | 将输入通道改成8、hidden改成1536。是连续 latent denoiser 的投影先例，不是已验证的八叉树 occupancy 实现。当前属此形式。 |
| 单层 Linear，不带 bias | 原3DShape2VecSet `LatentArrayTransformer.__init__` 的 [proj_in][vec-input]，`bias=False`；其 [forward][vec-input-forward] 实际调用该投影 | 同样需替换为8→1536。该模型输入集合 latent；bias选择独立于QKV是否带bias。 |

本轮只确证上述两种 Linear 构造。没有核实到外部 `N×8` 连续 occupancy 输入采用多层 MLP 的官方实现；若另列 MLP，必须标为跨角色适配。不能用256类整数 embedding或softmax替代连续加噪8维状态：那会改变论文规定的建模接口。

### 2. 可学习 depth embedding 的尺寸与融合

论文确定存在可学习深度编码，并在 transformer blocks 前使用；未给表的尺寸、参数是否逐块共享、相加还是其他融合的代码定义。[NEXUS §3.1.1][nexus-training]

当前：`Embedding(9,1536)`，以目标深度`d−1`查表，与 occupancy 投影及父格位置投影相加一次，然后进入36个 blocks。[构造](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:313)、[融合](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:369)

| 候选 | 确证内容 | 适配边界 |
|---|---|---|
| 共享H维表，在整网入口相加一次 | 当前代码已明确；OctGPT [AbsPosEmb][oct-depth]提供 learned depth table→H维→与空间编码相加的官方机制 | 入口只加一次属于本地选择，不能写成 OctGPT 原配置。D9的9个目标深度对应9行，是当前索引约定。 |
| 每个block各自H维表，在block入口相加 | OctGPT `OctFormerBlock.__init__/forward` [逐块构造和相加][oct-block]；`OctGPT._init_blocks` [实际传递pos_emb_type][oct-caller]，构造默认 [AbsPosEmb][oct-default] | 若用于NEXUS，可只重复注入depth部分，保留已另选的空间编码；这一步属于适配。原OctGPT同时重复加入空间编码，且每block有自己的depth参数。源码路线及构造默认已核对，发布权重配置是否采用该默认未确认。 |

两种均用H维深度向量；本轮没有找到“低维深度表→MLP/Linear升维”在所核官方深度编码中的直接证据，不补造第三方案。OctGPT生成 split信号及VQVAE表示，并最终解码SDF；其机制先例不能证明NEXUS occupancy效果。[OctGPT论文][oct-paper]

### 3. 另一项完整性缺口

NEXUS [§4.5][nexus-lod]明确第一阶段支持 face-count conditioning，但没有给其编码与接入方式。当前 `VertexDiT.forward` 只有time、depth与condition tokens，没有face-count参数；当前8192点+normal条件数据也没有这一输入。它应作为“论文存在、编码融合未公开、当前未实现”的独立条件项补录。它不是预测顶点数的count head，也不取代逐层occupancy生成。具体候选交由主报告的face-count专项汇总。

## 二、论文已确定：表示与接口，不列自由结构菜单

| 项目 | 论文确定的边界 | 当前实际对应 |
|---|---|---|
| token与输出 | 每个已占据父格的8个子格值组成一个token；实值0/1目标采用velocity预测 | 输入`[B,N,8]`，输出`[B,N,8]`。多子格可同时占据，是multi-hot，不能改成8类one-hot。 |
| 父格序列长度 | 第d层只对第d−1层的已占据节点预测；N随形状和层级变化 | `parent_codes[B,N,3]`提供这些已知父格位置；这不是固定数量的可学习生成query。 |
| 最终顶点数量 | 从单个根节点逐层扩展；最终全部已占据D层叶格各给一个顶点 | 点数等于最终占据叶格数，无独立count head或固定顶点slots。1024是条件token数，8192是输入点数，均不是输出点数。 |
| 顶点坐标表示 | D-bit整数格；最终输出为叶格中心；实验使用D=9和整数范围[0,511] | D9 labels是唯一叶格集合；父格中心用于位置编码。不能把最终坐标改成直接回归XYZ还称保持论文表示。 |
| 深度与空间信息 | 一个网络跨层共享；可学习depth embedding及3D RoPE存在 | 精确深度融合与RoPE数值方案仍属于未公开结构细节；“是否保留”已确定。 |
| 点条件规格 | VecSet联合训练；点坐标及法向；8192输入点、1024 tokens、hidden2048、8层；CA接入denoiser | FPS/learned query、8层计数、特征频率、heads/norm已在上一版展开。固定条件query数不约束变长父格数。 |

以上依据：[NEXUS §3.1][nexus-vertex]、[§3.1.1–3.1.2][nexus-training]、[§4.1][nexus-implementation]。`[-1,1]`包围盒归一化另见[§4.2][nexus-point-eval]；论文未给精确中心/缩放公式，不把某一种公式冒充作者公开实现。

### 表示实现的必要说明

- 当前D9真实数据链使用`floor((v+1)*256)`后clamp[0,511]、再unique；并核验与原D15整数右移6位一致。量化碰撞合并并记录，不能声称当前D9拒绝全部碰撞。[prepare_data.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/prepare_data.py:41)
- 当前child位序为`4bx+2by+bz`，解码中心为`−1+2(q+0.5)/512`；这是需一致的标签/解码约定，不作网络方案菜单。`OctreeLevel.metadata`虽含`depth/max_depth`，当前DiT不消费它；当前深度来源是整数查表。[octree.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/octree.py:41)、[child编码](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/octree.py:79)
- mask表示padding有效性，不是occupancy，也不是预测点数。当前SA屏蔽无效key，逐block清零无效query输出；CA另有condition mask。packed/padded等不列作结构候选。[attention](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:157)、[block](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:272)
- **输入密度接口缺口**：当前FPS要求每个对象至少1024个有效输入索引，即使XYZ有重复，仍保持采样索引唯一；512点会直接抛错。[FPS检查](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:36) 论文[Table 5][nexus-density]包含512点条件测试。因此当前条件编码器不能原样覆盖这一案例；这是接口覆盖差异，不能据此判定8192点训练故障。learned-query路线不需要从输入抽取1024个不同索引，但迁移效果仍未验证。
- 根到叶生成函数确实按占据子格扩展；E2的GT-parent单层诊断只检查给定父格上的预测，不能当作最终点数生成验证。二值阈值、排序、容量保护及padding实现不纳入本轮网络结构清单。[generate_cells](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex_evaluation.py:135)

结论：上一版核心网络项目之外，需补录 occupancy输入投影、depth embedding融合和face-count条件。坐标量化、child位序、mask语义与输出点数是表示边界；不新增固定point query/count head方案。

[hy-input]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L578-L585
[hy-config]: https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml#L1-L40
[vec-input]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L179-L200
[vec-input-forward]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L211-L225
[oct-depth]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L205-L269
[oct-block]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octformer.py#L274-L304
[oct-caller]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octgpt.py#L97-L113
[oct-default]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octgpt.py#L24-L34
[oct-paper]: https://arxiv.org/abs/2504.09975
[nexus-vertex]: https://arxiv.org/html/2607.13563v1#S3.SS1
[nexus-training]: https://arxiv.org/html/2607.13563v1#S3.SS1.SSS1
[nexus-implementation]: https://arxiv.org/html/2607.13563v1#S4.SS1
[nexus-point-eval]: https://arxiv.org/html/2607.13563v1#S4.SS2
[nexus-lod]: https://arxiv.org/html/2607.13563v1#S4.SS5
[nexus-density]: https://arxiv.org/html/2607.13563v1#S4.SS2.SSS3
