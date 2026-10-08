# 点云条件编码器的结构候选与固定源码证据

核查日：2026-10-05。只读研究；仅新增本笔记和 `sources/` 中的公开文本快照，未改模型、训练或权重。与 NEXUS 的直接联系：这些候选影响8192个表面点及法线如何成为1024×2048条件集合，供八叉树 occupancy velocity DiT 使用。它们是结构候选，不能由 shape latent 重建成功推论到 NEXUS occupancy flow 成功。

## 1. 固定约束与当前实现

NEXUS §4.1 Vertex Diffusion 明确 **VecSet、8层、hidden2048、8192表面点及法线、1024×2048条件tokens**。本节不能把层数、宽度、token数或法线输入一起列为作者未说明的细节。未定的是8层的计数/排列、query构造、点嵌入、heads、norm/FFN等。论文见 [NEXUS v1](https://arxiv.org/abs/2607.13563v1)；本地逐页文本第6页 L055–059：[来源文件](/Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt:354)。

当前冻结源：[vertex.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:219)。`VertexConditionEncoder` 实际为 **1CA+7SA，总8个 attention+FFN blocks；FPS1024；共享点投影；XYZ8个频率 π·2^k，XYZ保留，加raw normal成54维；16heads×128；Pre-LN；GELU4倍FFN；末尾LN；encoder无QK norm**。1024tokens只有特征，未向DiT传query点坐标。`_VertexBlock`用这些特征作为每层CA的K/V；其DiT QK RMSNorm与encoder无QK norm需分开。

## 2. 官方模型的实际启用配置

代码SHA为本次核查的固定版本，HF SHA为本次读取的发布配置固定版本；均不是“作者训练时必然使用的commit”。本地VecSet/Hunyuan2.1的SHA复核成功，Hunyuan2.1工作树无修改。

| 来源 | 论文/任务 | 固定代码与实际配置 | 编码过程/输入 |
|---|---|---|---|
| 3DShape2VecSet | [v3 §5.1/5.3/6](https://arxiv.org/html/2301.11445v3#S5.SS1)；AE occupancy field + latent EDM；论文另有partial point cloud conditioning | `1zb/3DShape2VecSet` `8df9b7a55c42d4dcad152294755250a2ab1e34e5`；factory [V-factory] | 默认2048 XYZ→512×512；**encode只有1CA+FFN**，single-head CA；24SA在decode；π·2^k，8bands；无normal。不能把24SA当encoder。 |
| Michelangelo | [v2 §3.1](https://arxiv.org/html/2306.17115v2#S3.SS1)；CLIP语义对齐的shape VAE | `NeuralCarver/Michelangelo` `6d83b0bacef92715dd5179d45647ed9a3d39bc95`；[M-config][M-global] | 配置256local+1global learned queries、width768、12heads=64/head、1CA+8SA、normal3、8bands、**include_pi=false**、末LN。global token用于语义对齐；不能直接把257改称256。 |
| Hunyuan3D-2 | [v1](https://arxiv.org/abs/2501.12202v1)；shape VAE latent→field→mesh；DiT实际条件是图像 | `Tencent-Hunyuan/Hunyuan3D-2` `f8db63096c8282cb27354314d896feba5ba6ff8a`；HF `9cd649ba6913f7a852e3286bad86bfa9a2d83dcf` 的 **withencoder** [H2-config] | num_latents3072、width1024、16heads=64/head、8bands/noπ、point_feats4、qk_norm=true；`num_encoder_layers`配置省略，由 [H2-model] 默认8，实际**1CA+8SA**。ordinary VAE的decoder-only配置不能证明encoder启用。 |
| Hunyuan3D-2.1 | [v1 §3.1.1](https://arxiv.org/html/2506.15442v1#S3.SS1.SSS1)；shape SDF VAE，点云是被压缩目标，DiT条件为图像 | `Tencent-Hunyuan/Hunyuan3D-2.1` `82920d643c0dc2f7bfd7255f45f62d386edfe60c`；HF `0b94677654c57bb9a6b6845cd7b704ccf551d327` [H21-config] | num_latents4096、width1024、16heads=64/head、**1CA+8SA**、8bands/noπ、point_feats4、qk_norm=true、末LN。encoder head Q/K norm是 **LayerNorm**，不是其DiT的RMSNorm。发布配置pc_size81920、sharpedge_size0，未启用uniform+sharp混合路径。 |
| TripoSG | [v1 §3.2](https://arxiv.org/html/2502.06608v1#S3.SS2)；SDF VAE + latent rectified flow，DiT图像条件 | `VAST-AI-Research/TripoSG` `fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c`；HF `2c1c516d22d58db486a058d98d31bb6177344e06` [T-config] | encoder width**512**、8heads=64/head（decoder1024）、1CA+8SA、8bands/noπ、raw normals、无QK norm、末LN；`_sample_features`随机先选4×num_tokens，再XYZ FPS得到query。公开encoder的`torch_cluster.fps` import默认注释，使用encoder须按README启用，不能声称开箱即跑已验证。 |
| Dora | [v2 §3.2.2 Eq8–11](https://arxiv.org/html/2412.17808v2#S3.SS2.SSS2)；SES+dual CA的shape VAE；本次配置为Dora1.1 TSDF | `Seed3D/Dora` `a166e21e900cf1e1230a67db2efc8b5a2c022e7d`；[D-config] | width768、12heads=64/head、8SA；uniform/salient各自FPS，再拼query；两个独立CA+FFN的输出相加后SA；normal raw，noπ，末LN；`use_downsample=true`实际启用。验证代码各取1024+1024=>2048query，不能看到候选表1024就称总1024。 |

Hunyuan `point_feats4`不是4维法线：其数据代码将normal3与sharpedge标记1组合；[H21-data]。这一路输入不是NEXUS的固定XYZ+normal6维，可以借方法但不应额外带入标签。Hunyuan2.1论文的最大latent3072与发布4096也应按各自证据分别记录。

## 3. 分项覆盖表：可选方案、实际来源、适配限制

以下“适配”明确表示按NEXUS的2048宽/1024tokens/8层重定规模；不是原配置逐项照搬。

| 自行补定项 | 方案A | 方案B | 方案C/证据边界 |
|---|---|---|---|
| **8层如何计数** | **1CA+7SA，总8个blocks**：当前实现。聚合+SA路线有Hunyuan/TripoSG先例，但7是NEXUS总8约束下的本地适配；原VecSet不能唯一导出它。 | **1CA聚合视为tokenizer，另8SA**：Hunyuan2/2.1、Michelangelo、TripoSG参数都这样计数，物理是9个blocks。[H21-encoder][T-encoder][M-encoder]。若把NEXUS“8层”理解为全部blocks，则不符合；只能作为层数解释候选，不能静默更改。 | **dual CA聚合+SA**：Dora实际有两个并行CA blocks。[D-encoder]。总8blocks适配可是2CA+6SA；若把dual聚合算1stage再7SA，实际9个attention blocks。需单独写清计数和新输入分组；不是当前1+7的来源。 |
| **query构造** | **FPS XYZ embedding作Q**：原VecSet `KLAutoEncoder.encode`，[V-encode]；FPS XYZ，和全点云共用`PointEmbed`。在NEXUS保留normal约束时可使normal进入K/V，Q保持仅XYZ，但“normal只给K/V”是适配，原VecSet没有normals。 | **FPS point-feature embedding作Q**：当前实现；Hunyuan [H21-sample]、TripoSG [T-sample][T-encode]实际把所选XYZ Fourier与该点normal/feature一起作Q。query来自表面几何及normal；共享投影。 | **learned Q**：Michelangelo `CrossAttentionEncoder.query=nn.Parameter`，[M-encoder]，实际配置已启用；Q与输入点索引无绑定，normal只通过数据进入CA。NEXUS可设1024learnedQ，去掉语义global token并改宽2048属于适配；不继承其CLIP对齐目标。 |
| **点特征频率/编码** | **8bands π·2^k + rawXYZ + rawnormal →54 →2048**：当前；π频率和Linear来源原VecSet [V-embed]，加入normal是NEXUS适配。最高角频率128π。 | **8bands 2^k + rawXYZ + rawnormal →54 →2048**：Michelangelo/Hunyuan2/2.1/TripoSG/Dora发布/仓库配置实际 `include_pi=false`。[M-config][H21-config][T-config][D-config]。仅频率谱变，不能声称noπ必更好；最高角频率128。 | 本次只确证这**2种实际启用**的编码谱。Fourier(normal)、learned Fourier、relative-MLP等不能仅凭文件中存在就冒称发布模型使用。 |
| **normal注入** | **rawnormal与XYZ Fourier合并，共享线性投影，normal进入FPS Q与K/V**：Hunyuan/TripoSG和当前；[H21-sample][T-encode]。 | **rawnormal只在点数据K/V内，Q由learned slots提供**：Michelangelo确证启用。[M-encoder][M-config]。它与query改变耦合，不能当作纯normal消融。 | **Fourier normal**：Hunyuan `normal_pe`和Dora `embed_point_feats`有代码但未在所查配置启用；Hunyuan `ShapeVAE`未暴露/传入normal_pe，且直接翻转会使input_proj尺寸不匹配；Dora true分支未构造随后调用的input_proj1。[H21-encoder][H21-sample][D-encoder]。不列为已验证可用的第三方案。 |
| **token聚合** | **单CA读全部点，然后SA**：原VecSet的CA、Hunyuan/TripoSG encoder都提供真实代码。[V-encode][H21-encoder][T-encoder]。1024×8192权重；当前路线最小。 | **uniform/salient两个CA分别归一化，结果相加，再SA**：Dora实际启用，[D-encoder][D-config]；两个分支不是把两组点串起来做一次softmax。 | 只2种已确证方案。Dora来源需要mesh SES/分组信息；若当前输入只有均匀8192点，伪造sharp标签不能复现它。把总点数8192/总query1024固定重分配是新适配，应与结构变量分开。 |
| **attention heads（encoder）** | **16heads×128**：当前。2048/16与Hunyuan实际DiT发布尺寸相同，但所查point encoders没有该尺寸；[H21-dit-config]与[H21-dit]支持宽/heads接线，不证明encoder最优。 | **32heads×64**：按Hunyuan点encoder1024/16、Michelangelo768/12、TripoSG512/8保留64/head的比例扩宽到2048。[H21-config][M-config][T-config]。32是适配值，并非这些仓库发布2048点encoder配置。 | **single-head CA(2048/head) + multihead SA**：原VecSet实际CA是1head×512，[V-ca]；可按2048扩宽但巨大head是适配。原VecSet SA是8×64位于decoder；不能把SA规格当encoder原配。此项要分别报CA和SA heads。 |
| **encoder norm** | **Pre-LN、无QK norm**：当前；原VecSet/Michelangelo/TripoSG点encoder实际先例。[V-preln][M-block][T-encoder]。 | **Pre-LN +逐head Q/K LayerNorm**：Hunyuan发布VAE qk_norm=true，`norm_layer=nn.LayerNorm`默认实际贯穿encoder，[H21-qknorm][H21-config]。保持encoder与DiT norm分项。 | **Pre-LN +逐head Q/K RMSNorm**：Hunyuan实际DiT [H21-dit]有先例，点encoder未查到启用；移到condition encoder是跨角色适配，不等于“Hunyuan点encoder也用RMS”。 |
| **FFN** | **GELU，H→4H→H**：当前/Hunyuan/Michelangelo/TripoSG点encoder。[H21-mlp][M-mlp][T-encoder]。 | **GEGLU，H→8H，分两支后有效4H→H**：原VecSet [V-preln]。有效中间4H不意味着与GELU4H参数相同（GEGLU FFN约12H²，普通GELU约8H²，不计bias）。 | 只2种相关且实际启用方案；SwiGLU不补作第三方案。 |
| **条件token接入** | **每层SA→CA→FFN，条件仅作为CA的K/V，独立K/V投影**：当前；原VecSet `BasicTransformerBlock`，[V-denoiser]，Hunyuan `HunYuanDiTPlain`同类，[H21-dit]。可承接2048条件宽与1536主干不同宽度。 | **CA的三分支分别受时间AdaLN调制**：原VecSet实际denoiser，[V-denoiser]；当前CA不受时间调制。仍是CA接入，需与主agent的时间调制项合并，不能算新的encoder。 | NEXUS已要求CA，prefix/把condition拼入SA序列不能作为可自由替换的补定项。本次未找到同等明确的“共享2048→1536adapter，再每层CA”发布先例；不虚构为第三方案。 |
| **末尾norm** | **encoder末LN**：当前/Hunyuan/Michelangelo/TripoSG实际开启；[H21-encoder][M-config][T-encoder]。 | **不加独立末LN，CA+FFN后直接输出**：原VecSet `encode`真实结构，[V-encode]，但随后有KL头；NEXUS不加KL头是任务适配。 | 原VecSet的KL mean/logvar不是NEXUS条件编码器要求，不应借末LN候选顺带加VAE。 |

**聚合与query的独立性**：原VecSet不是max/mean pooling，都是cross-attention。FPS是构造Q的方法；learnedQ并不排除CA，CA与FPS也不是互斥方案。Dora的dualCA与Hunyuan把分组点拼接后singleCA并非同一个归一化机制。Dora源码相加的是两个完整residual CA+FFN block的输出，含两份query残差；直接改成`Q+CA_uniform+CA_sharp`也不是同一实现。

**FPS细节仍是补定项**：当前从离质心最远点确定起始点；原VecSet调用`torch_cluster.fps`未显式指定`random_start`，Hunyuan wrapper默认true，TripoSG显式 `random_start=self.training`。[V-encode][H21-fps][T-sample]。FPS方法有引用不代表当前确定性起点由论文规定。当前query继承原normal等特征；“点queries”不等于“带query坐标的条件输出”，输出token没有必须保留坐标。

## 4. 明确缺项与结论边界

- 所查官方实现/配置没有提供NEXUS的8层2048/1024token conditioner细节。其引用足以支持VecSet方法家族，不能唯一恢复上述配置。
- 原VecSet论文completion的条件通过shape encoder得到集合，denoiser CA读取它；固定公开树主要训练入口为category-conditioned。`EDMPrecond.forward`可直接接float context [V-context]，但不足以确认completion conditioner是否冻结、是否另训、是否用KL头。NEXUS自己的联合训练要求优先。
- Hunyuan/TripoSG点encoder压缩GT shape供VAE重建和latent生成，DiT条件是DINO图像，不是点云。这些encoder可以提供结构先例；其latent大小、norm、VAE reconstruction指标不能用于宣布NEXUS occupancy生成改善。
- Michelangelo的learned query同时服务CLIP对齐；只取query结构即可，无需引入global语义token、KL压缩或额外对齐损失。Dora的分组与SES依赖mesh，不能以普通点云接口替代后声称继承同一数据条件。
- 本次是静态代码/配置核查，没有实例化模型、测参数总数、训练或评价。对FPS成本、head选择收敛、normal增益和NEXUS严格occupancy FP/FN没有实验结论。

## 固定代码链接

[V-factory]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L404-L427
[V-encode]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L338-L394
[V-embed]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L109-L139
[V-ca]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L308-L326
[V-preln]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L34-L106
[V-denoiser]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L118-L168
[V-context]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L491-L507
[M-config]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/configs/aligned_shape_latents/shapevae-256.yaml#L1-L19
[M-encoder]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/tsal/sal_perceiver.py#L20-L112
[M-global]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/tsal/sal_perceiver.py#L309-L368
[M-block]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/modules/transformer_blocks.py#L77-L115
[M-mlp]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/modules/transformer_blocks.py#L229-L244
[H2-config]: https://huggingface.co/tencent/Hunyuan3D-2/blob/9cd649ba6913f7a852e3286bad86bfa9a2d83dcf/hunyuan3d-vae-v2-0-withencoder/config.yaml#L1-L16
[H2-model]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/f8db63096c8282cb27354314d896feba5ba6ff8a/hy3dgen/shapegen/models/autoencoders/model.py#L198-L244
[H21-config]: https://huggingface.co/tencent/Hunyuan3D-2.1/blob/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-vae-v2-1/config.yaml#L1-L19
[H21-dit-config]: https://huggingface.co/tencent/Hunyuan3D-2.1/blob/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-dit-v2-1/config.yaml#L1-L19
[H21-encoder]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L521-L583
[H21-sample]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L585-L716
[H21-fps]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L507-L518
[H21-qknorm]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L195-L255
[H21-mlp]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L176-L192
[H21-data]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/data/dit_asl.py#L178-L208
[H21-dit]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L304-L405
[T-config]: https://huggingface.co/VAST-AI/TripoSG/blob/2c1c516d22d58db486a058d98d31bb6177344e06/vae/config.json#L1-L14
[T-encoder]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/autoencoders/autoencoder_kl_triposg.py#L26-L87
[T-sample]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/autoencoders/autoencoder_kl_triposg.py#L402-L437
[T-encode]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/autoencoders/autoencoder_kl_triposg.py#L439-L457
[D-config]: https://github.com/Seed3D/Dora/blob/a166e21e900cf1e1230a67db2efc8b5a2c022e7d/pytorch_lightning/configs/shape-autoencoder/Dora-VAE-test.yaml#L23-L41
[D-encoder]: https://github.com/Seed3D/Dora/blob/a166e21e900cf1e1230a67db2efc8b5a2c022e7d/pytorch_lightning/craftsman/models/autoencoders/michelangelo_autoencoder.py#L31-L160

快照文件保留原始字节与行号，供离线核查；其HTTP源URL和SHA256见 `sources/vecset_source_manifest.json`。这些只有公开代码/配置/树元数据，不含权重或凭据。
