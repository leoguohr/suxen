# 沿 Nexus 参考文献追踪八叉树与位置编码

核对日期：2026-09-07。依据 Nexus arXiv v1 官方 HTML、相关论文和官方代码。未 clone、下载权重、启动训练或修改模型代码。

## 1. 参考文献精确对应

| Nexus 引用 | 准确论文 | 在 Nexus 中的位置与作用 |
|---|---|---|
| Xiong et al., 2025 | **OctFusion: Octree-based Diffusion Models for 3D Shape Generation**，arXiv:2408.14732，CGF/SGP 2025 | 仅 §2.1 的八叉树 related work |
| Guo et al., 2025 | **Hyper3D: Efficient 3D Representation via Hybrid Triplane and Octree Feature for Enhanced 3D Shape Variational Auto-Encoders**，arXiv:2503.10403 | 仅 §2.1 related work |
| Deng et al., 2025 | **Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization**，arXiv:2504.02817，ICCV 2025；方法名 OAT / OctreeGPT | 仅 §2.1 related work |
| Wei et al., 2025 | **OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation**，arXiv:2504.09975，SIGGRAPH 2025 | 仅 §2.1 related work |
| Lai et al., 2025a / 2025b | **LATTICE: Democratize High-Fidelity 3D Generation at Scale**，arXiv:2512.03052 | 两个条目题名/arXiv号相同，引用用于§1背景和§4.3语义评测。本地PDF在背景标a、评测标b，HTML的a/b标记相反；引用位置与用途以具体版本为准 |

来源：[Nexus §2.1](https://arxiv.org/html/2607.13563v1#S2.SS1)、[Nexus References](https://arxiv.org/html/2607.13563v1#bib)。HTML references 每条 `Cited by` 可核对实际引用位置。

**这些引用不等于 Nexus 显式采用了它们的模型代码、block 或超参数。** Nexus 在方法部分为 VecSet 引 Zhang2023、为 RoPE 引 Su2023、为 DiT 引 Peebles/Xie2023；上述八叉树论文可以补设计依据，不能补成作者未公开细节的“证明”。

## 2. 优先对照一：OctGPT（Wei et al.）

它最直接支持“八叉树空间位置与深度应分别编码”，而且有官方代码。

- [论文 v1 §3.2.2 Positional Encoding](https://arxiv.org/html/2504.09975v1#S3.SS2.SSS2)：每个 token 对应 octree node 的3D位置与深度；3D RoPE 编码空间，learnable scale embedding 区分深度。
- [§3.2.1](https://arxiv.org/html/2504.09975v1#S3.SS2.SSS1)：OctFormer 式固定 token 数窗口，交替 dilated / shifted window，允许不同深度 token 交互。窗口大小固定，不代表总序列固定，也不是 VecSet 查询压缩。
- [§3.2.3](https://arxiv.org/html/2504.09975v1#S3.SS2.SSS3)：depth-wise mask 保留浅层到深层依赖，层内可并行预测多个被遮挡 token。仍是离散分类的 masked autoregression，不是连续 flow matching。

**目标语义和 Nexus 不同**：OctGPT 序列含各层节点 split/no-split bits，以及最细层 VQVAE 二值 latent；解码为 SDF，再 Marching Cubes。没有 Nexus 每父节点 `N×8` 速度场，也没有“逐层生成全部 mesh vertex occupancy”的相同任务定义。

官方仓库 `octree-nn/octgpt` 本次 `git ls-remote HEAD` 为 `9eb824296bb6888219f34eb78fe81d9d1c5912d4`。

| 代码证据 | 确认内容 |
|---|---|
| [positional_embedding.py L69–86](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L69-L86) | 提供 axial 3D RoPE，XYZ 频率分开计算后拼接 |
| [L109–147](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L109-L147) | 另支持 mixed/learned 3D frequency；从 `octree.xyz` 计算旋转，应用于 Q/K |
| [L154–202](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L154-L202) | 显式 `depth_emb`；深度索引来自 `octree.depth_idx`，与空间编码相加 |
| [octformer.py L209](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octformer.py#L209) | 注意力实际构造 `RotaryPosEmb(..., rope_mixed=True)` |
| [octformer.py L292–304](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octformer.py#L292-L304) | 每个block在attention前把空间/深度编码相加到hidden；并非仅整网入口加一次 |
| [octgpt.py L75–113](https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octgpt.py#L75-L113) | binary split embedding/head；encoder和decoder OctFormer；条件CA可按间隔注入 |

实际帮助：确认“3D RoPE + depth embedding”已有成熟先例，并暴露一个需要显式选择的细节：**axial 与 mixed RoPE 是不同实现，不应从‘3D RoPE’四个字推断 Nexus 采用哪种。** 不建议为当前复现引入 OctGPT 的离散token头、MAR mask、跨深度窗口或VQVAE。

尤其需要区分：该源码不仅有RoPE，还会在每块加绝对空间编码和depth embedding。这是OctGPT自己的具体设计，不能用“二者都写了RoPE”来证明Nexus也有额外绝对位置投影。

## 3. 优先对照二：OAT / OctreeGPT（Deng et al.）

它与“每父节点8个孩子”的数据合同最接近，但生成建模不同。

### 八孩子结构码

[论文 v2 §3.3](https://arxiv.org/html/2504.02817v2#S3.SS3) 明确：BFS 序列中，每个节点 token 为 `(q(v), χ(v))`；`q(v)` 是量化 latent index，`χ(v)∈{0,1}^8` 是8个潜在孩子存在与否的结构码。结构头做256类交叉熵分类，推理逐token预测并据结构码确定最终变长序列。

空间层级信息为 `Embed_x + Embed_y + Embed_z + Embed_depth`，坐标使用量化cell center。它不是3D RoPE，也不是一层 `N×8` 连续速度场。

### 点云编码器提供的结构证据

[§3.2，Eq.(7–8)](https://arxiv.org/html/2504.02817v2#S3.SS2)：

```text
point K/V = concat(PE(point XYZ), normal)
leaf queries = concat(PE(cell center), scale encoding(depth))
latents = Le 个 Self-Attention(Cross-Attention(leaf queries, point K/V))
```

每个leaf query全局读取所有点，随后leaf之间自注意力。这支持“几何位置形成query → CA聚合 → SA细化”的可解释设计，但 **query数等于自适应leaf数、是变长的**，不能证明Nexus固定1024条件token或“1+7”的层计数。

输出路线：残差量化 latent / octree → Perceiver occupancy-field decoder → 查询规则网格 → Marching Cubes。Octree细分由点及法向的QEM误差驱动，平面区域可早停；不是Nexus根据目标mesh顶点分布生成到固定深度9。

[官方项目](https://oat-3d.github.io/)本次未提供可核实的作者代码入口；上述结论为论文级证据，不声称代码已复查。

## 4. 辅助对照：LATTICE

重点是“条件与生成 token 的空间锚点”，不提供新的八叉树占用头。

[§3.1–3.2](https://arxiv.org/html/2512.03052v1#S3.SS1) 把 VecSet latent 锚定到 active coarse voxel center；先用现成 Hunyuan3D-2 等产生粗形状并体素化，再用 rectified-flow Transformer 生成细节 latent，RoPE用于这些有已知锚点的噪声tokens。VAE还是点云 → latent → SDF → Marching Cubes。

训练随机抽固定数量的结构tokens并从1024逐步增至6144；测试可增token数。这是对**生成表示本身**的抽样，不是Nexus点云条件编码器的固定query压缩。不能据此把Nexus每层真实parent集合截到固定N。

论文训练时对点query加小抖动，推理/DiT训练改用voxel query，以减小空间锚点分布差异；这一点与Meta MeshFlow的经验相互呼应：坐标输入不只是实现位置编码，还要定义训练/推理时坐标究竟来自哪里。

[官方项目](https://lattice3d.github.io/)的GitHub按钮本次指向 `https://github.com/Zeqiang-Lai/LATTICE`，打开404；因此本次没有可核验的官方模型源码。只能作为论文设计依据，不能声称已经核查其block实现。

## 5. 另两项保持有限范围

- **OctFusion**：前一份 [other_papers_notes.md](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/other_papers_notes.md) 已核对；8通道孩子split status + unified U-Net + SDF路径，不重复扩大调研。
- **Hyper3D**：[官方论文 §3.2–3.4](https://arxiv.org/html/2503.10403v1#S3.SS2) 重点是预训练octree feature extractor与hybrid triplane/grid VAE；不是共享八叉树Vertex DiT。它对当前 `N×8` occupancy结构不如上面两篇直接，未深入代码。

## 6. 收敛到 Nexus 结构的判断

1. 在Nexus自己的引用中，**OctGPT最适合参考3D位置与深度的分离编码，OAT最适合参考8孩子结构码与几何query聚合**。
2. 它们只提供候选实现的依据。Nexus的flow目标、shared DiT、`N×8`token、点云条件规格仍以Nexus正文为准。
3. 本轮没有找到可以完整当作“Nexus Vertex Diffusion官方等价实现”的前作；不会因为都叫octree就把SDF生成、VAE编码或AR结构头混入当前任务。
4. 下一版必须明确：父节点坐标尺度、RoPE轴向或混合形式、depth embedding注入点、VecSet query如何初始化、CA/SA的层数计数。这些依然是复现选择，不能写成Nexus作者配置。
