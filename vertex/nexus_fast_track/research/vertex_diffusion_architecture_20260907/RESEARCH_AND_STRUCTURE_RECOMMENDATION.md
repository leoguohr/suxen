# Nexus 点云条件 Vertex Diffusion：参考实现调研与结构建议

核查日期：2026-09-07。此文是结构讨论材料，未修改模型、删除原型、下载权重或启动训练。结论来自官方论文、已有官方源码 checkout 和在线官方源码；没有运行这些模型，因此不作速度、稳定性或生成质量的实测排名。

**建议以 TRELLIS.2 的 DiT block 为主要对照，以 VecSet/Hunyuan 的点云聚合器为编码器对照，以 OctFusion 检查八叉树逐层生成的接口。Meta MeshFlow 补充条件位置设计的经验。** 这表示参考实现的适配程度，不表示已经还原 Nexus 作者未公开的结构。

## 1. 必须保留的 Nexus 任务定义

```text
8192个表面点及法线 → 联合训练的VecSet → 1024×2048条件tokens
                                              ↓ cross-attention
当前层N个占用父节点 + noisy [N,8] → 共享Vertex DiT → [N,8]速度
                   父节点位置提供3D RoPE；另有可学习depth embedding
```

一个token对应一个父节点的八个孩子占用，连续化目标为0/1。生成逐层展开到D=9，最终取占用叶格中心作为顶点。此阶段独立于Topology AE的训练。

论文明确：VecSet为8层/hidden2048，Vertex DiT约2B；但没有给出DiT精确层宽、heads、FFN比、时间调制形式、VecSet的8层计数规则和完整采样器配置。模型约2B的精细统计口径也不能从一句规模描述中反推。

证据：[Nexus §3.1](https://arxiv.org/html/2607.13563v1#S3.SS1)，PDF第4页左栏L031–046、右栏L008–027；[§4.1](https://arxiv.org/html/2607.13563v1#S4.SS1)，PDF第6页左栏L051–059、右栏L018–024。这里的L为本地`pdftotext -layout`每页非空行编号，不是论文印刷行号，左右栏需一起指明。

[本地逐页行号文本](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt)；[原PDF](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1.pdf)。

## 2. 这些模型究竟生成什么

| 参考 | 本次核验的生成路径 | 与Nexus Vertex的对应关系 |
|---|---|---|
| Hunyuan3D-2.1 Shape | 图像条件→ShapeVAE latent→SDF→Marching Cubes | 可对照Transformer/点云聚合器；VAE的点云encoder并非生成条件encoder |
| TRELLIS.1 | SS latent→占用VAE decoder→稀疏位置；再生成SLat，交表示decoder | 分阶段结构相似，但SS flow没有直接预测N×8占用 |
| TRELLIS.2 | SS latent→稀疏位置；shape latent→O-Voxel/dual-grid表面重建；另有material模型 | 空间token DiT适合作为主干参考，不能把其VAE或表面提取搬成Nexus Vertex |
| Meta MeshFlow | MeshVAE latent→连续顶点/有效性/邻接特征→三角面 | 原生mesh latent生成，包含几何拓扑；不是单独八叉树Vertex模型 |
| OctFusion | octree split signal与leaf latent→局部SDF/MPU→Marching Cubes | 对照层级生成和二值信号连续化；八通道split语义不同 |
| Hunyuan3D-Omni | 图像+几何控制→ShapeVAE latent→SDF | 有点云控制，但公开流程不是纯点云条件VecSet |
| BPT / MeshAnything V2 | 点云编码→自回归mesh tokens→三角面 | 对照点云编码/注入；不用于决定flow DiT或采样器 |

来源：[Hunyuan §3.1](https://arxiv.org/html/2506.15442v1#S3.SS1)、[TRELLIS §3.3](https://arxiv.org/html/2412.01506v1#S3.SS3)、[TRELLIS.2 §3](https://arxiv.org/html/2512.14692v1#S3)、[Meta MeshFlow §3](https://arxiv.org/html/2606.04621v1#S3)、[OctFusion §3](https://arxiv.org/html/2408.14732v2#S3)。条件分支与具体代码证据见文末四份专题笔记。

## 3. DiT内部：名称相同，结构并不唯一

SA = token间自注意力；CA =生成token读取条件tokens；FFN =逐token前馈网络。AdaLN用时间特征调节归一化后的缩放、偏置，gate控制残差分支的幅度。

| 项目 | Hunyuan3D-2.1发布模型 | TRELLIS.1公开image-L配置 | TRELLIS.2 SS/shape配置 | Meta MeshFlow |
|---|---|---|---|---|
| 层数/宽度 | 21/2048 | 24/1024 | 30/1536 | 论文16/1536 |
| heads | 16 | 16 | 12 | 本文不把代码默认值当权重配置 |
| block顺序 | LN→SA；LN→CA；LN→FFN/MoE | AdaLN→SA+gate；LN→CA；AdaLN→FFN+gate | 同左，时间投影共享 | SA→CA→FFN，三个分支均有时间gate |
| 时间进入方式 | sinusoidal+MLP形成额外token | 每块独立6C调制投影 | AdaLN-single共享6C投影+每块独立偏置 | 每块7C AdaLN-Zero |
| 位置 | 发布shape配置无显式位置编码 | APE | self-attention 3D RoPE | 3D RoPE，CA可选Q旋转 |
| QK norm | self/cross均用逐head RMS | self开，cross默认关 | self/cross均开 | 源码可选，权重flag未核实 |
| 特殊外围 | U形长skip；末6层MoE，8专家top2 | SLat含稀疏卷积压缩与skip | shape主干Linear→DiT→Linear | 可选长skip |

Hunyuan实际加载`HunYuanDiTPlain`，不是附近的Flux式`Hunyuan3DDiT`；mini-overfitting YAML也不是正式发布模型配置。TRELLIS.1上述只代表所选L配置，不能泛化为全部型号。TRELLIS.2论文约4B是三个约1.3B模型的合计。Meta论文给895M，但checkpoint配置获取需要授权，本次只核实论文规模和源码机制。

固定源码证据：[Hunyuan block L304–405](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L304-L405)、[Hunyuan时间token L637–667](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L637-L667)、[TRELLIS.1 block L99–149](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/modulated.py#L99-L149)、[TRELLIS.2 block L104–157](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L104-L157)、[Meta block L578–607](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L578-L607)。各配置链接及初始化细节见专题笔记。

## 4. 点云条件：最容易混淆的三个接口

**VAE目标编码器、生成条件编码器、生成token的位置编码，是三个不同接口。** Hunyuan ShapeVAE拿点云制造训练目标latent，不能据此说它的DiT已采用点云条件。Meta最终把点云位置送入RoPE，也不能据此删掉Nexus要求的VecSet。

### VecSet的层数需要纠正

原3DShape2VecSet：FPS query→坐标Fourier→一次CA+FFN；构造的self-attention堆栈在`decode`中运行。Hunyuan ShapeVAE：一次CA+FFN后再接8个self blocks。因此，之前的“1 cross+7 self=论文8层”只有算术和草案含义，没有找到能确定Nexus这样实现的一手证据。

可取的共同结构是先聚合8192输入点为1024 tokens，再在较短序列上处理。最终采用1+7还是1+8、query采用FPS还是可学习queries，必须作为独立复现选择写明，不能伪装成论文已确认值。

[原VecSet L229–266](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L229-L266)；[Hunyuan encoder L561–581](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L561-L581)。

### Meta MeshFlow的条件经验

附录F报告早期使用32768表面点→预训练Shape2VecSet类编码器→2048条件tokens→CA，收敛较慢；最终使用XYZ的3D RoPE，并以32³体素化缩小训练真实mesh顶点与推理表面点之间的分布差异。这是这篇论文的经验，不能推广成“CA不好”：其编码器训练方式、latent含义和Nexus均不同。

对我们最有用的启示是分别检查外部条件点云、父节点空间位置及训练/推理的坐标约定，不能仅检查tensor shape。[论文附录F](https://arxiv.org/html/2606.04621v1#S6)。

### Omni与自回归模型的补充

Omni的点云控制是XYZ复制成6维（不是法线），经Fourier/Linear/RMSNorm/GELU形成逐点tokens，再把控制类型tokens和图像tokens拼入CA。没有VecSet压缩或8层堆栈。BPT采用可微调Michelangelo点云encoder；MeshAnything V2则把257个条件tokens作为自回归前缀。这些都是有效参考，但不覆盖Nexus已规定的8192×6→1024×2048联合训练VecSet+CA。

[Omni point分支 L462–468](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8/hy3dshape/models/conditioners/omni_encoder.py#L462-L468)；[BPT配置](https://github.com/Tencent-Hunyuan/bpt/blob/main/config/BPT-open-8k-8-16.yaml#L4-L20)；[MeshAnything条件前缀](https://github.com/buaacyw/MeshAnythingV2/blob/main/MeshAnything/models/meshanything_v2.py#L93-L121)。

## 5. 位置与采样：不能随block一起盲目移植

- TRELLIS.2的RoPE只进self-attention；cross-attention读取条件KV。head dim128允许剩余维度零相位，因此不必为XYZ拆分强制选择96维/head。[RoPE L14–47](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/rope.py#L14-L47)
- Nexus还要确定父节点坐标以当前层整数格坐标还是统一物理坐标进入RoPE；depth embedding不能自动定义这个尺度。
- 仅在SA中使用相对位置RoPE，不等于已把父节点绝对位置传入点云CA。作为结构推导，若某层只有一个父节点、其余输入固定，纯SA RoPE的自身Q/K相位相消。应检查模型如何区分不同位置的单父节点；是否需要绝对坐标投影/位置化CA query仍是待定实现。这里没有运行实验，也没有声称Nexus作者遗漏位置。
- QK RMS只归一化attention内部每个head的Q/K，与此前Topology AE整体spacetime向量RMS归一化不同；不能直接挪用旧实验结论支持或否定它。
- Hunyuan2.1公开flow为噪声t=0→数据t=1；TRELLIS与Meta为噪声t=1→数据t=0。它们都能表示连续流，但速度符号、时间输入及采样步长必须一致。
- Hunyuan2.1、TRELLIS两代、Meta公开路径都采用Euler flow采样，不能提供Nexus所写DPM-Solver的直接复刻。OctFusion则是另一套扩散目标，不能仅因为参数名`ddim_steps`就归成同一flow。
- Nexus写每层20步DPM-Solver。旧DPM-Solver的`model_type="v"`指特定噪声/数据组合，不可直接等同于任意flow导数；需要确定路径与转换。Euler可作为以后验证ODE接口的参考基线，不能替换后仍称论文采样一致。[DPM-Solver官方说明](https://github.com/LuChengTHU/dpm-solver#supported-models-and-algorithms)

## 6. 下一版结构讨论建议

**主干建议以TRELLIS.2为主要代码对照**：普通Linear输入/输出、Pre-LN、SA→CA→FFN、self侧3D RoPE、时间调制self和FFN。将CA的context维度设置为Nexus规定的2048；主干hidden可以独立选择。此建议来自结构适配性，不是新训练结果。

接下来按顺序定四项：

1. **时间调制与初始化**：优先讨论TRELLIS.2的AdaLN-single，连同其共享投影、每块偏置和初始化一起审查；若选传统逐块AdaLN则重新预算参数。不能只说“照搬DiT”。
2. **VecSet精确定义**：保留论文输入/输出与联合训练；明确聚合层是否计入8层、FPS query、XYZ和法线编码。先不把1+7当既定答案。
3. **空间接口**：确定父节点RoPE坐标、跨深度尺度、depth加入位置，以及CA query如何保留父节点定位。候选结构须能通过单父节点位置区分与点云条件改变的功能检查。
4. **容量与训练/采样契约**：上述确定后重新核算约2B。此前36层×1536仅是基于特定逐块调制的估算，暂不定案；再统一flow路径、t分布、DPM-Solver转换与occupancy阈值。

第一版不需要为借鉴这些论文再引入MoE、occupancy VAE、MeshVAE、图像encoder或新的拓扑模型。它仍是点云VecSet与Vertex DiT联合训练，再加确定性的八叉树展开和顶点提取。

## 7. 可追溯的专题笔记

- [Hunyuan3D-2.1与Omni：发布配置、代码行号、条件与时间](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/hunyuan_notes.md)
- [TRELLIS.1/2：实际配置、AdaLN、RoPE与采样](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/trellis_notes.md)
- [Meta MeshFlow、OctFusion与另一篇同名MeshFlow](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/other_papers_notes.md)
- [原VecSet、BPT与MeshAnything V2](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_diffusion_architecture_20260907/vecset_and_point_condition_notes.md)
