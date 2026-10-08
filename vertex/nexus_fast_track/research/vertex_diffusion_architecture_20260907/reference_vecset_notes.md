# 从 Nexus 直接引用追溯 VecSet：原论文与官方实现

2026-09-07，只读调研；没有修改模型或启动训练。核查 3DShape2VecSet 最终 arXiv v3（2023-05-01），以及现有官方仓库 `/Users/luthier/Documents/sophomore/nexus_fast_track/external_references/3DShape2VecSet` 的 commit `8df9b7a55c42d4dcad152294755250a2ab1e34e5`。固定源码链接均指此 commit。

## 引用关系先分清

- **直接结构引用**：Nexus §3.1.1（PDF第4页右栏、项目逐页非空行 L013–L016）和 §4.1 Vertex Diffusion（第6页左栏 L055–L059）均明确引用 **Zhang et al. 2023，3DShape2VecSet，arXiv:2301.11445**。第二处同时给 Nexus 自己的 8层/2048宽/8192点及法向/1024条件tokens。
- **另一篇 Zhang et al. 2024**：Nexus参考文献中对应 **CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets**（Longwen Zhang等），不是3DShape2VecSet的新版本。它出现在更广泛的生成工作讨论中，不能替代条件编码器段落的直接引用。
- 本地 Nexus 对照文本：[第4页来源行](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt:193)、[第6页来源行](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt:354)、[Zhang2023/2024文献项](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt:676)。L编号是项目自制文本编号，不是原PDF印刷行号。

## 原论文对 encoder / decoder 的划分

核对的不是单纯函数名：[论文v3 §5.1](https://arxiv.org/html/2301.11445v3#S5.SS1) 的 Eq.(16)/(17) 将编码定义为 learned queries 或 FPS point queries 对输入点云的位置特征做cross-attention；Fig.4区分这两种query。最终采用point queries，见 §8.1。§5.2及Fig.5把KL压缩放在编码结果后。[§5.3](https://arxiv.org/html/2301.11445v3#S5.SS3) 的 Eq.(21) 将self-attention堆栈描述在Shape decoding部分；Fig.5进一步明确KL latent先升维再进入该解码过程。[§6及Fig.7](https://arxiv.org/html/2301.11445v3#S6) 说明partial point cloud条件用§5.1的shape encoder得到条件集合，通过denoiser的cross-attention注入。

**因此：原文与代码共同支持“cross聚合是原shape encoder，self堆栈位于decoder”这个判断。原论文不能证明Nexus的8层是1+7或1+8。** 原论文的结构讲解也没有给出名为“8层点云条件编码器”的完整规格。

## 官方代码能逐项确定什么

| 问题 | 已核实的原实现 | 固定源码位置 |
|---|---|---|
| 点输入 | AE默认2048个XYZ；`PointEmbed`明确处理 `[B,N,3]`，无normal输入 | [models_ae.py 109–139](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L109-L139)、[285–310](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L285-L310) |
| Query来源 | FPS从输入点选512个query；query/full point cloud使用同一PointEmbed | [338–366](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L338-L366) |
| 坐标编码 | 默认每轴8个频率：2^k×π；sin/cos共48维，再拼原XYZ为51维，Linear到hidden | [109–139](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L109-L139) |
| 编码聚合块 | 1个CA+1个FFN，各残差。CA为单头、head_dim=hidden（默认512） | [308–310](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L308-L310)、[363–366](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L363-L366) |
| LN与FFN | CA的query/context分别Pre-LayerNorm；FFN Pre-LayerNorm；GEGLU，4倍中间宽度；无QK norm | [34–106](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L34-L106) |
| KL与self堆栈顺序 | encode在CA+FFN后直接预测mean/logvar并采样；decode先latent升维，再执行self堆栈 | [363–394](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L363-L394) |
| self堆栈规格 | factory设24层、8 heads、head_dim64，hidden默认512；不是8层encoder | [404–427](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L404-L427) |
| `l8`名字 | `kl_d512_m512_l8`的l8表示 **latent_dim=8**，不是8层 | [441–443](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L441-L443) |

补充：原论文评估中出现“4096点及法向”是PointNet++评测特征输入，不能把它归到shape encoder输入。该项只用于提醒阅读时分清训练模型和评测模型；本表的实际输入契约直接由源码核实。

## 官方 latent denoiser 的可借结构

公开的 `models_class_cond.py` 可直接核实：

```text
AdaLayerNorm(t) → Self-Attention → residual
AdaLayerNorm(t) → Conditional Cross-Attention → residual
AdaLayerNorm(t) → GEGLU FFN → residual
```

三个分支分别有scale/shift时间调制；不是AdaLN-Zero的六路门控。代码声明的LayerScale在此配置为Identity。噪声时间经正弦编码和两层MLP后送各block；没有启用latent位置embedding。输出投影zero-init。主干默认heads=8、head_dim=64；公开README训练例为24层。

证据：[AdaLayerNorm及block 118–168](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L118-L168)、[time与output projection 187–232](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L187-L232)。

它采用EDM的噪声级sigma与预条件化clean-latent denoising，不是Nexus的flow velocity。`c_noise=log(sigma)/4`以及`D_x=c_skip*x+c_out*F_x`见 [491–510](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L491-L510)。因此可借block次序与调制形式，不能原样继承训练目标、solver或无空间位置设定。

## 点云条件复现的公开边界

原论文有partial-point-cloud条件生成；已核查的官方仓库发布完整AE和 **category-conditioned** 训练/采样入口（README 37–61、`main_class_cond.py`等）。本次在该固定树中未找到独立点云completion训练配置/入口。`EDMPrecond.forward`可把float条件直接送为context，但这不足以确定原partial-point-cloud条件编码器是否复用AE权重、是否冻结、是否另训或是否包含KL头。

不要把“论文明确复用§5.1结构”扩写成“作者公开了完整点云conditioner训练配置”。

## 对 Nexus 可以缩小到哪里

- **有直接引用支持的候选**：FPS选query；query和full-point分别嵌入绝对XYZ；cross-attention压缩成较短集合；条件集合通过denoiser的cross-attention注入。
- **Nexus自身优先确定**：8192点及normal→1024×2048；8层；与Vertex DiT联合训练。原VecSet的2048点/512tokens/512宽/单头CA不能覆盖这些规格。
- **仍需复现选择**：8层如何计数与排列；加入normal的投影方式；是否延用单头CA、GEGLU或Pre-LN；是否加条件端self堆栈；条件末尾LN；query点是否保留坐标元数据。不能从引用唯一推出。
- 若采用“FPS+1CA+若干self”，应写为以原VecSet聚合方法为基础、按Nexus规模扩展的独立实现；Hunyuan的“1+8”只属于后续实现先例，不能冒充Nexus直接来源。
