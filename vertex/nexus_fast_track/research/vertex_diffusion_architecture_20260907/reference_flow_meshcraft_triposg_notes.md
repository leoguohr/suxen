# Nexus 引用追踪：Flow Matching、DPM-Solver、MeshCraft 与 TripoSG

核查日期：2026-09-07。只读官方论文/仓库，本次没有实现或运行模型。

## 引用强度

- Lipman 2023 Flow Matching 与 Liu 2023 Rectified Flow：Nexus方法§3.1.1明确引用，PDF第4页右栏L008–010。
- Lu 2022 DPM-Solver：Nexus实现§4.1明确采用，PDF第6页右栏L023–024，每个八叉树深度20步。
- He 2025 MeshCraft：Nexus引言对已有latent diffusion mesh方法的讨论，PDF第2页左栏L044–046；没有声明继承其网络。
- Li 2025 TripoSG：Nexus引言3D场生成背景，PDF第2页左栏L018–020；没有声明继承其网络。

Nexus参考文献条目：第13页左栏L013–014(MeshCraft)、L030–033(TripoSG)、L037–038(FM)、L042–046(RF/DPM-Solver)。行号按现有本地PDF非空行文本编号。

## 1. Rectified Flow 可以直接为训练公式提供依据

[Flow Straight and Fast 原论文 Algorithm 1](https://arxiv.org/html/2209.03003v1#S2)明确写线性插值、端点差速度目标、均匀时间采样和平方损失。对于Nexus的0/1孩子占用目标Y，可以作如下实例化：

```text
epsilon ~ N(0,I)
t ~ Uniform(0,1)
x_t = (1-t) epsilon + t Y
target_velocity = Y - epsilon
loss = squared_error(model(x_t,t,parent_positions,depth,condition), target_velocity)
```

前四项是原RF算法的直接形式；将其用于Nexus占用表示、对padding做mask并按mesh聚合，是我们的任务适配。Nexus正文只说flow matching与velocity prediction，没有逐项公开这些训练超参数；所以这是一套有直接引用依据的候选，而不是已经找到了Nexus原训练代码。

Lipman的[Flow Matching §4.1 Eq20–23](https://arxiv.org/html/2210.02747v2#S4.SS1)还给出末端保留sigma_min的线性路径：`x_t=tY+[1-(1-sigma_min)t]epsilon`，目标`Y-(1-sigma_min)epsilon`。这说明Nexus引用FM/RF仍不足以唯一确定末端噪声或时间采样分布。也不需要因为文中出现OT一词就给Nexus父子节点增加集合匹配模型。

## 2. DPM-Solver：找到明确的接口差异，但不是不可适配

[官方支持模型表](https://github.com/LuChengTHU/dpm-solver#supported-models-and-algorithms)将`model_type="v"`定义为`alpha_t*epsilon - sigma_t*x0`；它不等于上述线性flow中的`Y-epsilon`。官方同时支持noise/data模型接口、多种阶数及DPM-Solver/++，所以仅凭Nexus一句DPM-Solver也无法确定具体变体。

对上面的noise→data线性路径，可以直接代数得到：

```text
estimated_data  = x_t + (1-t) * predicted_flow
estimated_noise = x_t - t * predicted_flow
```

这是本次结构推导，不是运行结果。它说明可以通过数据/噪声预测接口继续构造适配；仍须定义solver的alpha/sigma、模型输入缩放、时间映射和端点，不能只改一个`model_type`字符串。此处没有实现采样器，也没有声称这就是Nexus作者的转换。

## 3. MeshCraft：Nexus引用列表里与变长flow DiT最相关的补充

论文：[MeshCraft: Exploring Efficient and Controllable Mesh Generation with Flow-based DiTs](https://arxiv.org/html/2503.23022v1)。

[§4.2](https://arxiv.org/html/2503.23022v1#S4.SS2)明确给出：

- 基于SiT适配变长face tokens，同一batch补到最长序列；attention排除padding，loss也排除padding。
- 采用AdaLN-Zero，将面数embedding加到时间embedding；图像等外部条件另经cross-attention注入。
- 引入sandwich normalization、SwiGLU、QK-norm；时间采用logit-normal，监督速度MSE。

对Nexus的价值是：它提供了“变长token + 时间调制 + 外部CA + padding/loss mask”的明确论文先例。这比仅凭原DiT的引用猜测组合结构更有针对性。

边界：MeshCraft的token是排序后的面latent，Nexus Vertex是具有父节点空间位置的八个孩子占用。不能照搬它的面数量定义、序列RoPE或VAE；Nexus未声明采用其归一化或激活。论文§4.2的RoPE文字写keys/values，单靠该句不宜替我们确定Q/K/V旋转细节。

作者[官方仓库](https://github.com/XianglongHe/MeshCraft)本次仍仅有README并写“Codes coming soon.”，因此上述属于论文证据，不能称为已核验的运行实现。

## 4. TripoSG：时间token与点云/法线编码的相关先例

[TripoSG §3.1.1 Eq2–6](https://arxiv.org/html/2502.06608v1#S3.SS1.SSS1)明确写：时间经编码/MLP形成单个token，与latent序列拼接；block为Pre-Norm Self-Attention，再用两个CA分别读取全局和局部图像条件，最后FFN。其基础主干21层、hidden2048、16heads、约1.5B，包含长skip。

这和上一轮Hunyuan的时间token路径相呼应，进一步表明DiT＋CA并不强制采用AdaLN。TripoSG被Nexus列为背景引用，不能凭机构或作者重合认定Nexus复用了它。

[§3.1.2 Eq9](https://arxiv.org/html/2502.06608v1#S3.SS1.SSS2)使用`x_t=t*data+(1-t)*noise`，并描述logit-normal时间采样。这为与当前草案相同的时间方向提供了另一个3D flow先例，但Nexus的t分布仍需独立选择。

其生成目标为SDF-VAE latent，参考意义限于主干/训练方法；不应在Nexus Vertex前新增SDF-VAE、全局图像分支或MoE。[官方发布仓库](https://github.com/VAST-AI-Research/TripoSG)提供1.5B模型和推理源码；本轮未逐行审计它的全部调用配置，主干描述按上述论文公式。
