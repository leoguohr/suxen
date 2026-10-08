# 沿 Nexus 参考文献追源：对 Vertex Diffusion 的新增帮助

2026-09-07。补充上一轮Hunyuan/TRELLIS横向调研。此次核查Nexus相关引用的实际位置，再读原论文和可获得的官方源码。没有改动模型、训练、下载权重或新增模型依赖。

**新增最值得参考的是3DShape2VecSet自己的denoiser、OctGPT的位置/深度实现、OAT的八孩子结构码与几何query，以及MeshCraft的变长flow DiT。** 它们让候选结构更有依据，但没有消除Nexus未公布的精确结构空白。

## 1. 引用关系：明确采用与相关工作要分开

下列页码取本地Nexus v1 PDF，L为`pdftotext -layout`每页非空行编号，不是作者印刷行号。双栏必须结合左右栏阅读。

| 文献 | Nexus具体位置 | 引用所能证明的内容 |
|---|---|---|
| 3DShape2VecSet，Zhang2023 | 第4页右栏L013–016；第6页左栏L055–059 | 点云条件使用联合训练VecSet；Nexus自己另给8层/2048宽/8192点与法线/1024tokens |
| DiT，Peebles/Xie2023 | 第6页左栏L051–054 | Vertex主干是约2B的DiT；不指定时间调制变体 |
| RoFormer，Su2023 | 第4页右栏L011–012 | 采用3D RoPE，另有learnable depth embedding；不给三轴实现 |
| Flow Matching与Rectified Flow | 第4页右栏L008–010 | 使用连续0/1目标和velocity prediction |
| DPM-Solver，Lu2022 | 第6页右栏L023–024 | 每个八叉树深度20步DPM-Solver |
| OctFusion、Hyper3D、OAT、OctGPT | 第2页右栏L055–057，§2.1 | 八叉树相关工作；没有明确继承其网络的声明 |
| MeshCraft，He2025 | 第2页左栏L044–046 | 作为latent-space mesh diffusion前作讨论 |
| TripoSG，Li2025 | 第2页左栏L018–020 | 作为3D场生成背景引用 |

[Nexus正文](https://arxiv.org/html/2607.13563v1)；[本地逐页行号文本](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt)。参考文献主要在第13页；完整对应和来源链接见下面专题笔记。

## 2. 3DShape2VecSet：不只可以参考点云encoder

原论文§5.1把learned/FPS queries的cross聚合定义为shape encoder；§5.3把self堆栈置于decoder；§6明确partial point cloud条件使用§5.1的encoder。这次用原论文确认了代码观察，不是凭`encode/decode`函数名推断。

更有用的新发现是，原官方denoiser本身已有如下block：

```text
AdaLN(t) → Self-Attention → residual
AdaLN(t) → 条件Cross-Attention → residual
AdaLN(t) → GEGLU FFN → residual
```

三条分支都有时间scale/shift，但没有AdaLN-Zero式时间gate。因此，除了TRELLIS.2，我们现在有一套来自Nexus直接引用文献的具体block对照。Nexus引用它用于encoder，并没有说denoiser也照搬；这仍是候选来源，不是继承证明。

仍然不能确定8层怎么数。原代码的`kl_d512_m512_l8`中`l8`是latent维度8，不能解释为8层。原AE输入是XYZ，不含Nexus所需法线，原denoiser采用EDM而非flow velocity。

[原论文§5.1](https://arxiv.org/html/2301.11445v3#S5.SS1)、[§5.3](https://arxiv.org/html/2301.11445v3#S5.SS3)、[官方denoiser L118–168](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L118-L168)。

## 3. OctGPT：父节点位置与深度的更具体参考

Wei2025的OctGPT论文§3.2.2明确使用3D RoPE与learnable scale embedding。官方代码实际采用mixed RoPE，对XYZ的混合频率进行学习并旋转Q/K；还在每个block加入额外空间/深度编码。

这对当前最有帮助的是：我们可以具体讨论父节点坐标怎样进入attention、depth如何区分层级、是否保留额外绝对坐标通路。它并不能证明Nexus同样使用mixed RoPE、额外绝对位置或每块重复加depth。

OctGPT是离散masked autoregression，最终经VQVAE/SDF恢复表面；我们不需要引入它的离散头、窗口mask和VAE。

[论文位置编码小节](https://arxiv.org/html/2504.09975v1#S3.SS2.SSS2)、[mixed旋转 L109–147](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L109-L147)、[空间/深度编码 L154–202](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L154-L202)、[每块注入 L292–304](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octformer.py#L292-L304)。

## 4. OAT / OctreeGPT：每父节点8孩子与几何query

Deng2025的OAT/OctreeGPT与上一项OctGPT是不同论文。它把节点表示为量化latent与8-bit child-existence码；用256类结构头逐token生成树。其点云encoder以leaf center位置及尺度形成query，以点XYZ/normal形成K/V，先全局CA聚合，再接多层SA。

这提供两处可借鉴：父子节点的结构编码，以及“带位置的query→点云CA→SA”的聚合布局。需要保留Nexus差异：Nexus每父节点的8维直接进入连续flow；OAT做离散256类分类。OAT的query数量随leaf数量变化，也不能代替Nexus固定1024条件tokens。

[OAT §3.2 Eq7–8](https://arxiv.org/html/2504.02817v2#S3.SS2)、[§3.3八孩子结构码](https://arxiv.org/html/2504.02817v2#S3.SS3)。本轮证据为论文，没有核实到可读的作者模型源码入口。

## 5. MeshCraft：变长flow DiT的直接设计说明

论文§4.2明确把SiT适配为变长mesh latent模型：attention与loss都mask padding；面数embedding加入时间条件，由AdaLN-Zero控制；外部图像条件另用CA。另描述QK-norm、sandwich normalization、SwiGLU和logit-normal时间采样。

对当前Nexus最有用的是mask与条件注入的分工，而非其面latent/VAE。可以把padding/loss处理作为实现检查点；归一化、激活和时间分布应分别选择，不能全部标为Nexus要求。其[官方仓库](https://github.com/XianglongHe/MeshCraft)当前仍为“Codes coming soon.”，因此这里是明确的论文先例，尚非可逐行核实的实现。

[MeshCraft §4.2](https://arxiv.org/html/2503.23022v1#S4.SS2)。

## 6. 三个仍不能由引用确定的问题

1. **时间调制不是唯一答案。** 原DiT§3.2/Fig3把cross-attention、AdaLN和AdaLN-Zero列为不同变体；其发布AdaLN-Zero代码没有CA。VecSet用三分支AdaLN，TripoSG用时间token，TRELLIS.2用AdaLN-single。引用DiT不能在这些候选中替作者作选择。[原DiT](https://arxiv.org/html/2212.09748v2#S3.SS2)、[TripoSG Eq2–6](https://arxiv.org/html/2502.06608v1#S3.SS1.SSS1)
2. **3D RoPE的细节不是RoFormer直接给出的。** 原RoFormer的“2D case”指特征维度为2，随后推广特征维度，仍以序列位置m为索引。XYZ分轴、mixed频率和八叉树坐标尺度都需要另外确定。[RoFormer §3，Eq11–16](https://arxiv.org/pdf/2104.09864v5)
3. **训练公式有更强依据，采样适配仍未公开。** RF Algorithm1直接支持线性插值、端点差速度、均匀t和平方损失；这可作为Nexus第一版的引用依据。但FM也允许其他路径，Nexus没公开其t分布。DPM-Solver的标准`v`接口不等于线性flow导数，仍需时间/输入/输出转换。[RF Algorithm1](https://arxiv.org/html/2209.03003v1#S2)、[DPM-Solver接口](https://github.com/LuChengTHU/dpm-solver#supported-models-and-algorithms)

## 7. 对下一步结构讨论的更新

建议参考顺序：**VecSet确定点云聚合与一个基础block候选；OctGPT/OAT检查坐标、depth及八孩子接口；MeshCraft/TRELLIS.2比较时间调制和变长训练实现；RF/DPM明确训练—采样契约。** 原先TRELLIS.2的工程参考价值仍在，但现在不应仅凭它一个模板决定全部细节。

还要定的结构是：8层VecSet具体布局，三分支是否时间调制/是否有gate/是否共享投影，父节点空间坐标与depth的注入位置，以及最终层宽参数预算。本轮没有找到完整公开的Nexus等价前作，所以不把这些选择说成已确认的作者结构。

SpaceMesh与GraphSAGE仍分别服务于Nexus的拓扑表示与拓扑encoder；PolyGen提供顶点/拓扑分解的思想。它们不改变当前讨论点云VecSet与Vertex DiT的范围。Hyper3D、LATTICE等补充边界见八叉树专题笔记。

专题证据：[VecSet](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/reference_vecset_notes.md)、[DiT/RoFormer](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/reference_dit_rope_notes.md)、[八叉树相关文献](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/reference_octree_notes.md)、[Flow/MeshCraft/TripoSG](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/reference_flow_meshcraft_triposg_notes.md)。
