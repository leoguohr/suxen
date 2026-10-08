# VecSet 与点云条件的补充核查

核查日期：2026-09-07。只读官方源码；没有修改模型或启动训练。

## 3DShape2VecSet 原实现

本地已有官方仓库：`/Users/luthier/Documents/sophomore/nexus_fast_track/external_references/3DShape2VecSet`；固定 commit `8df9b7a55c42d4dcad152294755250a2ab1e34e5`。

`models_ae.py` 的 `encode` 做 FPS，分别对采样 query 点和完整点云做坐标编码，再执行一次 cross-attention 与一次 FFN，各有残差。其 `self.layers` 堆栈在 `decode` 中运行。因此，不能看见构造函数里的 `depth` 就把这些层全部算为点云 encoder，也不能据此证明 Nexus 的8层是1 cross + 7 self。

[官方 encode/decode，L229–266](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L229-L266)

原始 `PointEmbed` 使用 XYZ 的 Fourier 特征与 XYZ 本身，原始 `encode` 接收 `[B,N,3]`，没有 Nexus 要求的表面法线。cross-attention 在这里是单头，后续 self-attention 则使用配置的多头数。这些都说明“VecSet”描述的是一类集合表示/编码思路，不能代替具体的层配置。

[PointEmbed，L109–138](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L109-L138)；[cross block，L204–207](https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L204-L207)

Nexus 自己明确给出了8192点及法线、1024条件tokens、hidden2048、8层与联合训练。具体计数和法线注入应以 Nexus 为任务约束、参考其他实现作独立选择，不能声称由原 VecSet 唯一确定。

## BPT：点云编码器可训练的实际例子

论文：Scaling Mesh Generation via Compressive Tokenization，CVPR 2025。官方发布的是基于 Michelangelo 微调点云编码器的 lite 模型。

发布配置 `BPT-open-8k-8-16.yaml` 明确 `conditioned_on_pc: True`、`encoder_name: miche-256-feature`、`encoder_freeze: False`、`pc_num: 4096`；主干配置宽1024、24层。这里是自回归 mesh token 生成，不是 flow DiT。它能作为“点云编码器参与适配”的先例，不能用来确定 Nexus 的1024 tokens或8层。

[官方 README](https://github.com/Tencent-Hunyuan/bpt#download-pretrained-models)；[官方发布配置，L4–20](https://github.com/Tencent-Hunyuan/bpt/blob/main/config/BPT-open-8k-8-16.yaml#L4-L20)

## MeshAnything V2：另一种条件注入方式

官方输入支持 `[N,6]` 点坐标+法线；训练说明要求下载 Michelangelo 的点云编码器。推理源码使用257个条件tokens，读取点云编码特征并投影，再将这些tokens作为自回归模型的输入前缀。

这说明点云条件既可以做前缀，也可以做cross-attention；但 Nexus 已明确写cross-attention，所以不能为节省实现工作把它改为MeshAnything的前缀方案。

[官方点云输入说明](https://github.com/buaacyw/MeshAnythingV2#point-cloud-command-line-inference)；[257 tokens，L14–19](https://github.com/buaacyw/MeshAnythingV2/blob/main/MeshAnything/models/meshanything_v2.py#L14-L19)；[条件处理与生成，L93–121](https://github.com/buaacyw/MeshAnythingV2/blob/main/MeshAnything/models/meshanything_v2.py#L93-L121)

本节 BPT/MeshAnything 在线链接为 main 分支，核查时间如上；GitHub API 固定提交查询受公共速率限制，本次没有为这两项伪造固定 SHA。它们仅用于补充比较，主结构建议依据已有固定提交的 Hunyuan/TRELLIS/VecSet 源码。
