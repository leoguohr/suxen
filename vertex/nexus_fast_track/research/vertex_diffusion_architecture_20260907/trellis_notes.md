# TRELLIS / TRELLIS.2 对 Nexus Vertex Diffusion 的结构参考

核查时间：2026-09-07。只读核查已有官方 checkout 和在线官方论文；没有修改模型、安装依赖或启动训练。下文结构值来自已发布训练配置，不能当作 Nexus 作者配置。

- TRELLIS 官方 remote：`https://github.com/microsoft/TRELLIS.git`；本地 HEAD `442aa1e1afb9014e80681d3bf604e8d728a86ee7`。
- TRELLIS.2 官方 remote：`https://github.com/microsoft/TRELLIS.2.git`；本地 HEAD `75fbf0183001ed9876c8dbb35de6b68552ee08bd`。
- [TRELLIS 论文 v1 §3.3](https://arxiv.org/html/2412.01506v1#S3.SS3)、[附录 A.1](https://arxiv.org/html/2412.01506v1#A1.SS1)；[TRELLIS.2 论文 v1 §3.3](https://arxiv.org/html/2512.14692v1#S3.SS3)、[附录 A.2 / Table 5](https://arxiv.org/html/2512.14692v1#A1.SS2)。本次浏览器 PDF 取回失败，因此不编造 PDF 页内行号；代码下面给固定 commit 的真实行号。

## 1. 生成目标先分清

TRELLIS 两阶段：SS flow 先生成 sparse structure **VAE 的连续 latent**，经独立 decoder 转成占用格；SLat flow 再在已有占用位置上生成局部 latent。论文 §3.3 明确先把二值格压成低分辨率连续 latent，再训练 flow；这不是 Nexus 每个父节点直接生成 8 个孩子的二值占用。

TRELLIS.2 分 sparse structure、geometry latent、material latent 三个生成模型。本次仅参考前两者。论文 §4 明确**每个 DiT 约 1.3B**，30 blocks、宽1536、12 heads、FFN8192；宣传中的约4B是三个模型总计，不是单个 Vertex DiT 的规模。

TRELLIS.2 的真实推理代码先采样 `[B,8,16,16,16]` 的 SS latent，交 decoder 并阈值化，再取坐标；随后在这些坐标上初始化每个位置32维的 shape latent 噪声。见 [pipeline L204–L267](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/trellis2_image_to_3d.py#L204-L267)。这里的8通道是VAE latent维数，不是Nexus八个孩子。

## 2. 已发布配置对比

| 项目 | TRELLIS image-L 的 SS / SLat | TRELLIS.2 SS / shape |
|---|---|---|
| 主干 | 24 blocks，宽1024，16 heads，head dim64 | 30 blocks，宽1536，12 heads，head dim128 |
| FFN | GELU，4倍宽度4096 | GELU，`mlp_ratio=5.3334`，实际8192 |
| 条件宽度 | 1024，DINOv2-L | 1024，DINOv3-L |
| 位置 | 配置实际选择 APE | 配置实际选择 3D RoPE |
| 时间调制 | 每块独立 `SiLU → Linear(C,6C)`，默认 `share_mod=false` | AdaLN-single：共享一次 `SiLU → Linear(C,6C)`，每块再加可学习6C偏置 |
| QK RMS | self 开，cross 默认关 | self 和 cross 均开 |
| SLat外围 | 稀疏卷积压缩2倍、上采样和skip | 删除卷积packing/skip，直接Linear–DiT–Linear |

配置证据：[v1 SS L4–L17](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L4-L17)、[v1 SLat L4–L19](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json#L4-L19)、[v1 defaults L68–L73](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/sparse_structure_flow.py#L68-L73)、[v2 SS L4–L18](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/ss_flow_img_dit_1_3B_64_bf16.json#L4-L18)、[v2 shape L4–L18](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L4-L18)。

注意：这里选择的是v1公开image-L配置，并非声称v1所有尺寸都只有24层。

## 3. Block 可以直接核验

两代均是：

```text
x → 无affine LayerNorm → 时间scale/shift → Self-Attention → 时间gate → 残差
  → 有affine LayerNorm → 条件Cross-Attention → 残差
  → 无affine LayerNorm → 时间scale/shift → FFN → 时间gate → 残差
```

条件分支在两者中**没有自己的时间gate**；不要笼统说三个子层都是AdaLN-Zero。归一化使用`LayerNorm32(eps=1e-6)`，FFN是两层Linear加GELU，不是SwiGLU。见 [v1 block L99–L149](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/modulated.py#L99-L149)、[v2 block L104–L157](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L104-L157)、[v2 FFN L49–L55](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/blocks.py#L49-L55)。

v2 shared modulation：模型只做一次6C时间投影，[模型L53–L58](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L53-L58)；每块`self.modulation`是可学习偏置，加到共享结果，[block L132–L144](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L132-L144)。这会实质改变参数预算，所以需要先定是否shared，再核算约2B；不能沿用之前36×1536的估算。

v2 released配置还选了`scaled`初始化：attention输出投影和FFN第二层按网络深度缩小，时间MLP另定std，共享时间投影与最终输出层零初始化。每块可学习modulation偏置却是随机初始化，不能简单称“所有残差初始为零”。见 [初始化L183–L222](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/sparse_structure_flow.py#L183-L222)。

## 4. 位置、QKnorm、条件接口

- TRELLIS.2 self-attention：QK投影 → **逐head QK RMS** → 3D RoPE → attention；cross-attention：Q来自生成token、KV来自条件token，可加QK RMS，但无3D RoPE。见 [attention L66–L99](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/modules.py#L66-L99)。稀疏版本明确assert“RoPE只支持self-attention”，[L43–L47](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/sparse/attention/modules.py#L43-L47)。
- RoPE沿XYZ分配频率，使用空间坐标索引。head dim128也可用：不能整分的剩余维度补零相位，不要求head dim必须被6整除。[RoPE L14–L47](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/rope.py#L14-L47)。Nexus仍需自己确定跨八叉树深度的坐标单位，TRELLIS.2不是九层统一八叉树预测器。
- QK RMS发生于attention内部，并非对Topology AE最终spacetime embedding整体做RMS；两种归一化不应混为一谈。
- 条件接口是`cond:[B,M,Ccond]`，由cross投影到主干宽度。Nexus可以让VecSet输出`[B,1024,2048]`，将`ctx_channels`定为2048；不要求条件宽度等于主干宽度。这是接口可迁移性的工程判断，不是TRELLIS做过点云条件训练的证明。
- DINO extractor使用eval/no_grad；Nexus要求VecSet联合训练，若借鉴训练框架，不能把这个冻结逻辑一起搬过去。[DINOv3训练条件extractor L61–L120](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/trainers/flow_matching/mixins/image_conditioned.py#L61-L120)。

## 5. Flow和采样实际约定

两代源码都采用 `x_t=(1-t)x0+[sigma_min+(1-sigma_min)t]*noise`，目标速度`(1-sigma_min)*noise-x0`；t=1是噪声，t=0是数据，模型时间输入`1000*t`，MSE训练。见 [v2 trainer L69–L104](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/trainers/flow_matching/flow_matching.py#L69-L104)、[L162–L172](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/trainers/flow_matching/flow_matching.py#L162-L172)。这是与此前Nexus草案`t=0噪声→t=1数据`相反的时间方向，数学可转换，但不能混用速度符号。

两代公开采样器是Flow Euler，从1走到0，`x_prev=x_t-(t-t_prev)*v`，可重参数化时间表和加CFG，并非DPM-Solver。[v2 Euler L79–L80](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/samplers/flow_euler.py#L79-L80)、[时间表L114–L118](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/pipelines/samplers/flow_euler.py#L114-L118)。TRELLIS.2 SS训练选logitNormal(1,1)，shape配置选uniform，不应声称两个阶段统一logitNormal。

## 6. 对Nexus讨论稿的结论

有一手实现支持的候选：Pre-LN；Self → Cross → FFN；时间调制self与FFN并设gate；self-attention 3D RoPE；生成token读点云条件KV；QKnorm作为待确认稳定性选项；AdaLN-single作为减少时间调制参数的可选方案。

仍然无法由这些论文确认：Nexus的精确DiT层宽、VecSet的8层如何计数、depth embedding加入位置、跨深度坐标尺度、Nexus是否采用shared modulation/QKnorm、binary占用训练与阈值后处理、DPM-Solver具体适配。这些必须继续标成我们的复现选择或论文空白。

最不应照搬：TRELLIS的occupancy-VAE、SLat卷积packing、固定SS输入网格和图像encoder冻结方式。借鉴block不等于替换Nexus的逐层八叉树表示。
