# NEXUS Vertex 7 套结构暖启动筛查

日期：2026-10-06。设计依据为已固定源码的 [结构选项表](/Users/luthier/Documents/sophomore/nexus_fast_track/research/vertex_structure_options_20261005/NEXUS_VERTEX_STRUCTURE_OPTIONS.md)。本卡只规定结构、迁移和比较边界；真实执行状态以运行目录的 manifest、迁移审计与日志为准。

本轮比较 **同一 A24000 来源下，七套结构继续适配 10000 个 optimizer updates 的效果**。这是架构筛查，不是根因验证，也不是从零训练的架构能力排名。调度方已报告原结构 E2 继续训练明显改善；因此不能把新结构称作必要修复。该新结果在本卡中属于调度方提供的证据，本设计审查未独立连接服务器复核。

## 共同契约

- 原 50 个 NEXUS2K UID，按现有选择文件保持顺序和内容；D1–D9 事件；每父格 8 个 occupancy 值，预测 8 维 flow velocity。保留源时间方向、loss 与数据处理。
- DiT：36 blocks、hidden 1536、12 heads、head_dim 128；SA → CA → FFN；完整 VecSet tokens 始终走 CA；CA 无动态 gate；FFN 为 4H；无 dropout / DropPath。
- VecSet：hidden 2048、1024 输出 tokens、1 个聚合 CA + 7 个 SA，共 8 blocks；输入 XYZ + 原始 normals；4H FFN，最后 LN。任何变体均没有独立 KL 头。
- 输入 `Linear(8,1536,bias=True)`；父格中心 XYZ 的 8 组 π Fourier + 原 XYZ，共 51 维，`Linear(51,1536)` 加到 token；保留 `Embedding(9,1536)`，按目标 depth−1 加到 token 一次。
- SA 3D RoPE 保留现有每轴 21 对、剩余 2 通道不旋转、base=10000、位置 `256 × normalized_parent_center`。CA 不加 RoPE。
- 时间保留 256 维 cos/sin、1000t、base=10000、`Linear(256,1536) → SiLU → Linear(1536,1536)`。每 block 独立 `SiLU → Linear(1536,9216)`，向 SA 与 FFN 各提供 shift/scale/gate。
- 保留源 36 层及投影形状，使主干约 1.93B 参数；应由实际实现导出 DiT、VecSet、合计的准确参数量。不能把约 0.40B 的 VecSet 算进 DiT 后声称 DiT 约 2.33B。
- 各分支独立由 A24000 迁移，新增 10000 次 optimizer updates，到累计 step 34000；每 update 8 micro events。不得用 epoch、microstep 或评估步代替 optimizer updates。
- 每分支 80000 events；UID-major × D1–D9 共 450 个槽，`80000 = 177×450 + 350`。本轮明确采用新 sweep cursor=0，从 UID0/depth1 开始：顺序前 350 槽获 178 次、后 100 槽 177 次。源 cursor=16000 仅保留在 provenance，不决定新表起点。不能假称每 UID-depth 次数完全相同。七分支的事件表、noise、time 应逐数组一致。共同序列和每槽计数必须保存。

## 七套完整配置

以下表中的“基线”均指 S0 的全部共同配置；仅列出的覆盖项变化，未列项保持 S0。

| ID | 成套名称 | VecSet 完整选择 | DiT 完整选择与输出 |
|---|---|---|---|
| S0 | 同期原结构 | 16 heads；FPS 选点的 XYZ Fourier + normals 经共享点投影作为 Q；8 组 π×2^k Fourier；QK norm 关；所有 LN affine、eps=1e-6；tanh GELU；QKV bias 开 | SA/CA 均现有 RMS：`x/sqrt(mean(x²)+1e-6)`，gain [12,128]；QKV bias 开；SA/FFN 无 affine LN eps=1e-6；CA affine LN eps=1e-6；独立 time AdaLN；tanh GELU；final 无 affine LN eps=1e-6 → Linear(1536,8) |
| S1 | 仅 final time AdaLN | S0 | S0，仅 final 改为无 affine LN eps=1e-6 → time 生成 shift/scale → Linear8。独立 `SiLU → Linear(1536,3072)` |
| S2 | 仅关闭 CA QK norm | S0 | S0，仅 CA 的 QK RMS/gain 删除；SA 保留现有 RMS |
| S3 | 仅 learned 1024 queries | S0，仅聚合 Q 改为可学习 [1024,2048]；K/V 仍来自完整点特征；不再 FPS 产生 Q | S0 |
| S4 | TRELLIS1 block + DiT final 调制配套 | S0 | SA QK 为 TRELLIS 原式 `normalize(x.float(),dim=-1,eps=1e-12)×sqrt(128)×gain`，gain [12,128]；CA QK norm 关；独立 time AdaLN、SA/FFN 无 affine LN + CA affine LN、QKV bias 开、tanh GELU均保留；final 为 S1 的 time AdaLN |
| S5 | Hunyuan 点编码器与 attention 配套 | 32 heads / 每头64；FPS 点特征 Q；8 组 2^k、不含π；CA/SA QK LN：eps=1e-6，affine gamma/beta [64] 跨 heads 共享；所有 QKV bias 关，attention output bias 开；exact GELU；主干 LN eps=1e-6，末 affine LN eps=1e-5 | SA/CA QK RMS eps=1e-6，gain [128] 跨12头共享；QKV bias 关，attention output bias 开；exact GELU；final affine LN eps=1e-6 → Linear8；36层、现有独立 time AdaLN 与主干 LN 保留作为 NEXUS 适配 |
| S6 | 老师双路全局条件配套 | S0；额外摘要 `m=masked_mean(condition_tokens)`，`g=LN2048(eps=1e-5) → Linear(2048,1536) → SiLU → Linear(1536,1536)` | 入口 `x += g[:,None,:]`，每层 AdaLN 输入 `time_embedding + g`，CA仍读完整 tokens；SA/CA QK norm 均关；SA/FFN 无affine LN eps=1e-5；CA affine LN保留eps=1e-6；exact GELU；final affine LN eps=1e-5 → Linear8；时间编码与RoPE沿用共同契约 |

S4 是明确标记的 TRELLIS1 block 与原 DiT final 组合适配，并非 TRELLIS1 的整网复刻。S5 是 Hunyuan 的编码器/attention 配套移植，并非 Hunyuan 整网复刻：不复制 time token、U 型 skip、MoE、latent 表示或图像条件。S6 的老师代码是本地候选逆向复建，不能充当 NEXUS 作者实现；仅移植双路全局条件与相应 block/head 结构。源老师成功包含坐标 prior，本轮不借用其成功结论。

S6 必须保留两处 `g` 注入、LN+两层 MLP 与 block/head 配套；不能退化为已有 F 的“均值 → 单 Linear → AdaLN”后重跑。均值摘要本身是从 VecSet 到老师全局向量接口的本地适配。

## 新参数初始化与 A24000 迁移

新参数初始化必须使用隔离 RNG；完成构造和迁移后恢复并核验 A24000 的 Python、NumPy、CPU torch 与训练所用 CUDA RNG。不同 GPU 物理序号上的分支应映射源训练卡的逻辑 RNG 状态，而不是直接选另一张卡的空闲 RNG。本轮训练 noise/time 明确使用共同的新 sweep event-keyed 私有 CUDA generators；源 RNG 恢复属于状态迁移记录，不能据此声称实际噪声延续 A24000 的下一次抽样。新增层不得污染公共 generator；七分支共同固定随机流应看实际事件/noise/time 数组哈希，不能只比 seed。

| 分支 | 模型状态迁移 | Adam 状态迁移 | 起始函数边界 |
|---|---|---|---|
| S0 | 所有模型张量原样继承 | 所有 step/exp_avg/exp_avg_sq 原样继承 | 应通过源输出复放等价检查 |
| S1 | 原权重全继承；新增 final 2H 投影 weight/bias 全0 | 原 Adam 继承；新增层 Adam fresh step0 / 零矩 | final 新调制为0，应与 S0 起始输出一致 |
| S2 | 原共有张量继承；丢弃36层 CA Q/K gain | 对应 gain 的 Adam 丢弃，其余继承 | 去掉已训练 norm 后通常改变函数 |
| S3 | 原共有张量继承；新增 learned query 按 Michelangelo 源式 `randn([1024,2048])×0.02` 隔离初始化并记录seed | 新 query fresh；原共有 Adam 继承 | query来源改变，不等价于FPS起点 |
| S4 | 原共有张量继承；SA gain [12,128]原样复制但算式改变；丢弃CA Q/K gain；final新增2H投影全0 | SA gain与共有Adam保留；CA gain Adam丢弃；final fresh | RMS epsilon规则与CA norm均改变，不等价S0/S2 |
| S5 | 原同名同shape张量继承；旧DiT Q/K gain [12,128]按head求均值得新[128]；新增VecSet QK-LN gamma1/beta0；新增final LN gamma1/beta0；两侧QKV biases删除 | 变形shared gain Adam重置；新增norm fresh；已删除bias Adam丢弃；其余原样继承 | head重分组、Fourier频率、QK/LN、bias与GELU都会改变函数；保留共有权重/Adam只是暖启动 |
| S6 | 原共有张量继承；删除DiT全部QK gain；新global LN gamma1/beta0，MLP第一层Xavier weight/zero bias，最后Linear weight/bias全0；final LN gamma1/beta0 | 新global模块/final norm fresh；已删QK gain Adam丢弃；其余继承 | 初始g=0但QKnorm/LN epsilon/GELU仍改变函数；不能声称完全等价起点 |

共有参数即使同名同shape，也可能因 head 划分或输入 Fourier 改变而改变语义；这类保留是显式适配策略，不是证明数学等价。旧Adam状态应按参数名字和已验证的源顺序映射，不能只按目标 optimizer 的编号 zip。变形 RMS gain 的二阶矩不能简单均值后冒称正确继承。

每套输出逐参数 manifest：source name/shape、target name/shape、动作（copied/new/transformed/dropped）、初始化或变换规则、Adam状态是否继承、状态step。数值核验可以采用 source/target tensor hash，或在 source checkpoint 总SHA256已核验的前提下，对所有迁移模型张量与继承的Adam step/一阶矩/二阶矩执行复制后 `torch.equal` 并保存检查数量及结果；不能只检查keys和shape。源 checkpoint 不含param_names时，需由冻结源模型的 `named_parameters()` 与源 optimizer param_group 顺序建立映射，并审计数量、shape和顺序。

不能在 `load_state_dict` 后再次执行全模型初始化；尤其必须保留 A24000 非零的最终输出权重和已学 modulation。不能对所有分支重置Adam后称继续训练；不能因新增参数step0而把整个scheduler/累计step回到0。新增层与源层不同Adam年龄属于本筛查条件。

## 接线与发车前核验

1. 对七套实际模块导出配置和准确参数量；确认所有输出都是[B,N,8]、pad输出0、输入点mask和父格mask语义未变。
2. S4 的 `gain[heads,head_dim]` 应按实际tensor布局正确广播；S5 QK-LN/RMS gain 均沿head_dim广播，不能跨token归一化。
3. S3 learned queries在batch间共享，每个样本独立读自己的K/V；不能把点云特征误替换掉。S6 masked mean只统计有效条件token，g在parent padding归零前注入，并通过两条路径回传梯度。
4. S5未添加time token，严禁从Hunyuan输出层复制 `x[:,1:]`。不得误删首个父格。所有分支保留3D RoPE与learned depth。
5. 检查S0迁移等价、S1零final调制等价；其余仅要求finite forward/loss/grad、正确图和状态清单，报告真实初始偏移。S6末层0会使global MLP前层初次梯度为0，末层应有梯度；首次更新之后前层才可能获得梯度，不能误判为永久断图。
6. 每分支写入共同数据/选择/事件/source checkpoint/source code哈希；保存首个update中、optimizer.step之前的共同8事件loss，作为初始loss对照。S0/S1起始输出等价由独立非零小模型测试覆盖；本轮不要求新增0步完整GPU输出数组或额外评估。保持训练预算相同。显存调优若改变 microbatch聚合形式，必须保持每update同8个事件及相同loss权重。
7. 排队器按分支保存完整模型、Adam、RNG、cursor和累计update；10000新增update即停止，OOM/异常不得静默改结构或无限重试。约70GB/卡是吞吐调优目标，不能用无用途占显存作为达标。
8. 只在发车前完成启动与短程健康验证。发车后遵照用户指示不监督、不轮询进展；队列自行执行有界工作。

## 已核对的源码依据

- TRELLIS1 `442aa1e1afb9014e80681d3bf604e8d728a86ee7`：[每头 RMS 原式](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/attention/modules.py#L8-L15)、[独立调制/SA→CA→FFN/CA无gate](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/modulated.py#L76-L150)、[SA QKnorm启用配置](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L4-L17)。本地核对镜像在 `diagnostics/five_reference_audit_20260907/references/TRELLIS`。
- DiT `ed81ce2229091fd4ecc9a223645f95cf379d582b`：[final独立2H AdaLN](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L125-L142)。本地核对 `research/vertex_structure_options_20261005/source_snapshots/facebookresearch_DiT/models.py:125`。
- Michelangelo `6d83b0bacef92715dd5179d45647ed9a3d39bc95`：[learned query 初始化与聚合](https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/tsal/sal_perceiver.py#L20-L112)。本地核对 `research/vertex_structure_options_20261005/sources/michelangelo_sal_perceiver.py:42`；原宽和token数不同，2048/1024是NEXUS适配。
- Hunyuan3D-2.1 `82920d643c0dc2f7bfd7255f45f62d386edfe60c`：[encoder QK-LN](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L195-L255)、[encoder exact GELU](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L176-L192)、[末LN构造](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L559-L583)、[DiT exact GELU与QKnorm/out bias](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L126-L168)、[final affine LN](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L450-L464)。本地核对镜像在 `external_references/Hunyuan3D-2.1`。
- Hunyuan发布 VAE `[width1024,heads16,num_freqs8,include_pi=false,qkv_bias=false,qk_norm=true]`：[固定配置](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-vae-v2-1/config.yaml#L1-L19)。扩大到2048宽/32头并保留总8 blocks与1024tokens属于本地NEXUS适配。发布DiT选择RMS和无QKV bias：[固定配置](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml#L1-L40)。本地快照分别为 `sources/hunyuan21_vae_config.yaml` 与 `sources/hunyuan21_dit_config.yaml`。
- 老师候选：[/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:44](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:44) 为无QKnorm、exact GELU、eps1e-5的独立调制block；[同文件:98](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98) 为LN+两层MLP；[同文件:114](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:114) 为全局向量同时进入token和时间调制。源输入是2048维文本，原hidden144；改为VecSet均值与hidden1536是明确适配。

本卡依据固定的本地源码内容核对，没有对远程网页或服务器进行实时复核。结构优劣、物理生成成功与七分支训练完成均不由此卡证明。
