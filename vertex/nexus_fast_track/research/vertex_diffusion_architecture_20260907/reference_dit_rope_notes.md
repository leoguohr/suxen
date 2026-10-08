# 沿 Nexus 直接参考文献追源：DiT / RoFormer / Flow Matching

核查日期：2026-09-07。仅阅读 Nexus 本地逐页原文、原论文和作者官方代码；未改模型或启动训练。

## Nexus 引用位置

Nexus v1 PDF 没有印刷行号。下列 Lxxx 是已有逐页、非空文本行号；双栏同行须结合左右栏使用：[编号文本](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt)。

| 直接引用 | Nexus正文位置 | 本身确定的内容 |
|---|---|---|
| Peebles and Xie 2023，DiT | PDF第6页左栏 L051、L053–L054；文本文件350–353行 | Vertex Diffusion采用约2B参数DiT |
| Su et al. 2023，RoFormer | PDF第4页右栏 L011–L012；文件191–192行 | 采用3D RoPE，并有可学习depth embedding |
| Lipman et al. 2023；Liu et al. 2023 | PDF第4页右栏 L008–L010；文件188–190行 | 占用值为0/1，flow matching，velocity prediction |
| 独立的条件机制说明 | PDF第4页右栏 L013–L016；文件193–196行 | VecSet与生成网络联合训练；条件通过cross-attention注入 |

参考文献对应条目位于PDF第13页：DiT左栏L052–L053，RoFormer左栏L074–L076，FM左栏L037–L038。在线：[Nexus v1](https://arxiv.org/html/2607.13563v1)。

## 1. 原 DiT 并不能证明 Nexus 采用 AdaLN-Zero + Cross-Attention

[原 DiT §3.2 / Fig.3](https://arxiv.org/html/2212.09748v2#S3.SS2)比较四种**并列变体**：in-context、cross-attention、AdaLN、AdaLN-Zero。cross-attention变体把时间和类别嵌入作为长度2的条件序列；AdaLN类变体把两者相加后用来调制归一化。原论文§5/Fig.5比较它们后，才选择AdaLN-Zero进行后续实验。因此，原论文没有把AdaLN-Zero与cross-attention的组合定义为唯一标准DiT。

作者官方发布代码固定commit：`ed81ce2229091fd4ecc9a223645f95cf379d582b`，来自`facebookresearch/DiT`，本次经GitHub API和raw源码核查。

- [models.py L101–L122](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122)：发布的AdaLN-Zero block只有self-attention与MLP两个残差分支，每个分支有shift、scale、gate；**没有cross-attention**。
- [L233–L246](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L233-L246)：条件是时间嵌入加类别嵌入，传给各block。
- [L207–L216](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L207-L216)：调制输出和最终输出零初始化。
- [L328–L329](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L328-L329)：XL/2为28层、宽1152、16heads；该值不能替代Nexus自己的约2B配置。

**证据边界：** Nexus引用DiT，只证明采用这一类主干；Nexus另行明确了点云cross-attention，但未交代时间如何注入。可以将“时间AdaLN-Zero + 点云cross-attention”列为有其他实现支持的候选，不能声称已沿引用链确认作者这样做。反过来也不能据此断言Nexus没有AdaLN-Zero；目前是未公开。

原DiT模型还采用二维图像patch、固定sincos位置和DDPM噪声/方差输出，与Nexus的八叉树占用、3D RoPE、flow速度输出有明显接口差异，不能整套照搬。

## 2. RoFormer证明旋转机制，不能补全3D父节点定位

[RoFormer v5 PDF](https://arxiv.org/pdf/2104.09864v5)已在线读取，共14页：

- **第4页 §3.1，Eq.(11)**：要求QK内积的位置依赖表现为序列索引之差`m−n`。
- **第4页 §3.2.1，Eq.(12–13)**：所谓“2D case”是**特征向量维度d=2**，不是图像XY空间坐标。
- **第5页 §3.2.2，Eq.(14–16)**：把特征分为二维旋转子空间，位置仍是单个序列索引m；给出旋转相乘形成相对位移的恒等式。

作者官方仓库固定commit：`dfc678ad506fc527ba17ead8db23cbe4d947a9b4`。[README L13–L35](https://github.com/ZhuiyiTechnology/roformer/blob/dfc678ad506fc527ba17ead8db23cbe4d947a9b4/README.md#L13-L35)提供self-attention伪代码：输入序列位置sin/cos，对Q、K成对旋转，再做内积；未定义XYZ三轴分配或八叉树深度尺度。

**表达应准确：** RoPE确实按绝对位置旋转Q和K；不能简化为“RoPE完全没有绝对位置信息”。但是在其标准self-attention用法中，成对内积的位置依赖是相对位移。若希望生成父节点与外部点云建立空间对应，原RoFormer没有给出这个跨输入坐标接口。

因此仍需确定：XYZ如何分配旋转通道、坐标采用网格索引还是统一物理尺度、跨深度如何缩放、RoPE用于哪些attention、父节点绝对空间位置是否通过另一条路径传入条件交互。原论文既不证明必须额外加absolute embedding，也不证明Nexus已经用纯self-RoPE完整解决它。

## 3. Lipman FM也不能独自固定噪声路径

[Flow Matching §4 / §4.1](https://arxiv.org/html/2210.02747v2#S4)给出一族Gaussian路径，包含diffusion路径和OT线性路径。Eq.(20–23)提供线性均值/标准差、常方向目标速度和CFM平方误差，可支持线性flow方案。但“Nexus引用FM”不足以确定它选择哪条路径、sigma_min、t采样分布或DPM-Solver接口；这些仍需要Nexus额外证据。Nexus还引用了Liu et al.，本笔记不替代对该引用的进一步核查。

## 对结构讨论的修正

目前能够从Nexus确定：8-child整体为token、共享深度网络、3D RoPE、learnable depth embedding、VecSet条件cross-attention、flow velocity、约2B。

引用链能补上：DiT有多种条件化方案，AdaLN-Zero的具体数学和官方实现；RoPE的QK旋转及相对位置恒等式；FM的可选路径理论。

引用链尚不能把以下内容升格为作者设置：AdaLN-Zero与cross组合、AdaLN-single、QKnorm、block精确顺序、3D轴划分、absolute parent conditioning、精确层宽和flow/solver约定。
