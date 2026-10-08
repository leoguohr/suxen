# Hunyuan3D-2.1：给 Nexus Vertex Diffusion 的结构参考

核查日期：2026-09-07。只读核查现有官方仓库与官方论文/权重配置，没有修改模型、拉取仓库或启动训练。

- 本地仓库：`/Users/luthier/Documents/sophomore/nexus_fast_track/external_references/Hunyuan3D-2.1`
- origin：`https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git`
- 本次源码 commit：`82920d643c0dc2f7bfd7255f45f62d386edfe60c`；`git status --short` 为空。
- [官方论文 v1 §3.1](https://arxiv.org/html/2506.15442v1#S3.SS1)
- [官方发布配置，固定上传 commit](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml)

## 最需要避免的三个误读

1. 官方发布模型调用的是 **`hunyuandit.HunYuanDiTPlain`**，不是同目录的 `hunyuan3ddit.Hunyuan3DDiT`。后者存在 Flux 式双流/单流与调制结构，但不能拿它当 2.1 发布模型的实际架构。
2. 实际 `HunYuanDiTPlain` **把时间嵌入作为额外 token 放在 latent 序列前面**，不是默认 AdaLN-Zero。`HunYuanDiTBlock.timested_modulate` 默认 false，模型构造也没有打开它。
3. Hunyuan ShapeVAE 的 `num_encoder_layers=8` 实际表示 **1 个 cross-attention 聚合块之后再接 8 个 self-attention 块**。不能用它支持我们之前暂定的“1+7”，更不能据此证明 Nexus 的8层也这样计数。

## 发布模型和 mini 教学配置

| 项目 | 官方发布配置 | 仓库 mini-overfitting 配置 |
|---|---|---|
| 生成对象 | ShapeVAE 的连续 latent，4096×64 | 同类 latent |
| 主干 | HunYuanDiTPlain | HunYuanDiTPlain |
| 层数/宽度/heads | 21 / 2048 / 16 | 16 / 2048 / 16 |
| 条件 | 1370×1024 的 DINOv2 图像 tokens | DINOv2-L，518像素图像 |
| QK norm | true，RMSNorm | 同左 |
| 显式序列位置编码 | use_pos_emb=false | 同左 |
| MoE | 末6层，8 experts，top-2 | 末3层，4 experts，top-2 |

仓库 README 把 Shape-v2-1 标为 3.3B；这里未实例化统计参数，不能把这个标签解释为已重新核实的总参数或激活参数数目。mini 是教学/过拟合配置，正式微调 YAML 使用 `from_pretrained` 加载发布结构。

证据：[mini YAML 78–96](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml#L78-L96)、[finetuning YAML 72–87](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml#L72-L87)。

## 实际 DiT block

```text
（后半段才有）来自前半段的 skip，与当前 token 在通道维拼接 → Linear → LayerNorm
  ↓
LayerNorm → Self-Attention → 残差
  ↓
LayerNorm → Cross-Attention（读取图像 tokens）→ 残差
  ↓
LayerNorm → GELU FFN / MoE → 残差
```

- 普通 FFN 为 H→4H→H，GELU。
- Self-attention 和 cross-attention 的 Q/K 均做逐 head 的 RMSNorm；block 主流归一化默认 LayerNorm。这两种 norm 不是同一处操作。
- 时间：正弦时间嵌入→两层 MLP→一个 `[B,1,H]` token，与 noisy latent tokens 拼接。该 token 在最后输出头被丢弃。
- 没有启用 3D RoPE；shape 主干的 `use_pos_emb=false`。论文里的 3D-aware RoPE 讨论属于 **Paint** 分支，不能移植成 Shape DiT 已用 RoPE 的证据。

源码：[HunYuanDiTBlock 304–405](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L304-L405)；[时间与 skip 路径 637–667](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L637-L667)；[TimestepEmbedder 92–123](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L92-L123)；[self-attention QK norm 265–297](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L265-L297)；[cross-attention QK norm 155–213](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L155-L213)。

## 点云编码器是什么、不是哪一部分

Hunyuan 的点云编码器在 **ShapeVAE encoder** 中，将已知 mesh 表面点压缩成扩散训练目标。它不是 DiT 的点云条件分支。DiT 的发布条件编码器是冻结的 DINOv2；冻结见 [conditioner.py 58–80](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/conditioner.py#L58-L80)。

可借的点云编码结构是：FPS 选 query，query/data 的 XYZ 做 Fourier encoding，与对应点特征合并后线性投影；1 个 residual cross-attention+FFN 聚合，再接8个 self-attention+FFN，末尾 LayerNorm。发布 VAE 隐藏宽度1024、latent宽度64，不能照搬成 Nexus 条件编码器的规格。

证据：[ShapeVAE 将 num_encoder_layers 传入 encoder 271–283](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/model.py#L271-L283)；[PointCrossAttentionEncoder 561–581](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L561-L581)；[FPS/位置/点特征 603–673](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L603-L673)；[forward 702–716](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L702-L716)。

另一个应明确的论文/发布差异：论文 §3.1.2 文本称 DINOv2 Giant，发布配置和微调配置实际为 DINOv2 Large（1024宽、24层）；本笔记表格按发布代码与权重配置填写。

## Flow 与采样

线性路径 `x_t=(1-t)ε+t z`，t=0为高斯噪声，t=1为真实latent，目标速度 `z-ε`，velocity MSE，训练t默认均匀。训练 YAML 设为 Linear/velocity，验证采样 Euler 50步。发布 pipeline 也用自定义 `FlowMatchEulerDiscreteScheduler`，不是 DPM-Solver；该 scheduler 时间递增，更新 `x_next=x+(t_next-t)*velocity`。名字叫 sigmas 的变量在这套接口里充当递增流时间，不能直接与反向扩散 scheduler 的命名对齐。

源码：[ICPlan 系数 43–54](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/diffusion/transport/path.py#L43-L54)；[路径及速度 139–162](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/diffusion/transport/path.py#L139-L162)；[采样与损失 138–182](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/diffusion/transport/transport.py#L138-L182)；[实际推理 736–772](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/pipelines.py#L736-L772)；[Euler update 301–310](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/schedulers.py#L301-L310)。

## 对 Nexus 的意义：工程判断

- 支持 `Self-Attention → Cross-Attention → FFN` 是有大模型实现先例的合理候选，但不能证明 Nexus 作者采用完全相同的 block。
- 说明 DiT 的时间注入没有唯一方案；“叫DiT所以必须AdaLN-Zero”不成立。
- FPS/Fourier/cross聚合后接self堆栈是可用的 VecSet 参考；“1+8”与 Nexus 自述8层的对应仍必须作为复现选择注明。
- 可以考察 QK norm，但不能把它与 Topology AE 中 space/time embedding 的 RMS 归一化混为同一个问题。
- 不建议为 Nexus 第一版引入 Hunyuan 的 MoE、长skip或图像编码器。这些不是 Nexus已知要求。
- **最根本差异：Hunyuan生成的是VAE latent→SDF→Marching Cubes mesh；Nexus Vertex生成的是每层八叉树父节点的8个占用量。** 因此不能沿用其64维latent接口、无空间位置的DiT、VAE或表面提取流程；Nexus仍需自己的3D RoPE、depth embedding、动态父节点集合与占用后处理。

## 有界补充：Hunyuan3D-Omni 的真实点云条件接口

仅检查 [Omni 论文 v1 §3.2.3/§3.2.5](https://arxiv.org/html/2509.21245v1#S3.SS2.SSS3) 和官方公开推理源码；没有克隆或下载权重。`git ls-remote` 核验官方 main commit 为 `4d47c0cc2bd0c4281963a7314ab330a5af36bfa8`。

**它确实有点云条件，但公开流程是图像加点云控制，不是已经验证的纯点云生成管线。** 论文点云输入为512/1024/2048个XYZ，允许完整/不完整点云，并描述drop/noise增强。它继续生成ShapeVAE latent，再解码SDF。

官方 `OmniEncoder` 的实际点云分支是：

```text
N×3 XYZ → 复制XYZ为N×6（不是附加法向）
       → Fourier embedding → Linear → RMSNorm → GELU
       → N个几何条件tokens
点云类型ID → Embedding(4,8) → Linear → 重复为10个类型tokens
DINO图像tokens + 几何tokens + 类型tokens → 沿序列维拼接
       → HunYuanDiTPlain各层的条件cross-attention
```

该轻量控制编码器没有VecSet式cross-attention压缩或8层self堆栈。image encoder冻结；控制tokens与图像tokens共用DiT条件接口。点云坐标在条件侧显式编码，提供了“条件空间信息怎样进入DiT”的另一种实现先例，但不证明Nexus可以删去论文规定的VecSet。它也没有给Nexus父节点query与点云key怎样做空间匹配提供完整答案。

固定源码证据：[OmniEncoder构造313–355](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8/hy3dshape/models/conditioners/omni_encoder.py#L313-L355)、[point分支462–468](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8/hy3dshape/models/conditioners/omni_encoder.py#L462-L468)、[DiT读取cond并送block 521–549](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8/hy3dshape/models/denoisers/hunyuandit.py#L521-L549)、[block cross-attention 344–370](https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/blob/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8/hy3dshape/models/denoisers/hunyuandit.py#L344-L370)。
