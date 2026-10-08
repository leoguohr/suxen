# NEXUS Vertex：位置、3D RoPE、时间与老师候选机制证据

核对日期：2026-10-05。只读当前 `d15_code/mini_nexus/vertex.py`、已有官方仓库和老师候选重建源码/验证记录；没有训练、GPU 推理、修改模型或重新逆向老师网络。本文列的是可选择的结构机制，不是已证实的故障原因，也不是 NEXUS 作者未公开设置的补全。

## 当前结构与证据级别

当前源码：[vertex.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py)。

| 项目 | 当前实际实现 | 可确认的边界 |
|---|---|---|
| 父格位置 | `parent_centers` L65–74 将 target child depth `d` 的 parent integer code `p` 转为 `c=-1+2(p+0.5)/2^(d-1)`；`_fourier_xyz` L26–32 为 XYZ + 8 bands `π·2^k` 的 sin/cos，51 维，经 Linear51→1536 加到 token | 额外 Fourier 加法路径是本地实现选择；NEXUS 已知规定 3D RoPE/depth embedding，不足以确认此额外分支 |
| RoPE | `apply_vertex_rope` L77–113；128/head 分 21 pairs/axis，126 维旋转 + 2 维保留；base10000，频率 `10000^(-k/21)`；L365 输入 `c·256` | 轴分配/频率形式与 TRELLIS.2 的官方实现有直接同机制依据；`c·256` 没有找到外部项目采用同一 NEXUS 数值设置的证据 |
| 时间 | `VertexDiT` L316–317、L371–378：256 维 cos→sin，`1000t`、base10000，Linear256→1536→SiLU→Linear1536→1536 | 和 DiT/TRELLIS 时间编码家族相同；NEXUS 论文不能单独确定 256、1000 或具体激活 |

## A. 额外父格位置路径的选择

这些候选直接关系到父格 token 如何知道自己在物体中的绝对位置，以及如何与点云条件交互。它们不改变八个 child occupancy 的表示。

| 方案 | 机制与输入单位 | 原项目实际选择 / 可选代码 | 对 NEXUS 的适配和取舍 |
|---|---|---|---|
| P1：连续 XYZ Fourier→学习投影，保留当前类别 | `φ(c)=[c,sin(ωc),cos(ωc)]` 后 Linear；NEXUS 用归一化物理中心 `c∈[-1,1]`。Hunyuan 的 FourierEmbedder 接受连续 XYZ，没有在该函数中强制单位 | Hunyuan3D 2.1 **ShapeVAE 点/查询编码器**采用 Fourier+Linear，论文明确说明；发布配置为 8 bands、`include_pi:false`。源码支持 `include_pi:true`，但不是该发布配置。见 H1–H3 | 当前 51 维结构有外部机制依据；当前 `π` 与 Hunyuan 发布值不相同。可保留 `π2^k` 或使用 `2^k`，但这是频率尺度候选，不能说 Hunyuan Shape DiT 也对父格加了这一分支；其用途是 VAE 编码，移入占用 DiT 是本地适配 |
| P2：TRELLIS 1 固定三轴 sincos APE 直接加 token | 对每轴整数 grid coordinate 使用 `ω_k=10000^(-k/m)`，拼接 sin/cos 为 hidden width；1536 时各轴 512 维，无当前 51→1536 的学习投影 | TRELLIS 1 图像条件 SLat 的公开配置明确 `pe_mode:"ape"`，在 sparse packing 后的 `h.coords[:,1:]` 上编码；不是 normalized XYZ。见 T1–T3 | 可用 parent integer `p`，或先推导统一参考坐标；使用哪一种必须写清单位。不能将 `[-1,1]` 直接喂给原整数频率后声称完全照搬。它改变绝对位置的频谱与可学习程度；不是纯删除分支对照 |
| P3：去掉额外加法位置，仅保留 self RoPE + depth | 位置通过 Q/K 旋转进入 self-attention；原始 source 用 sparse integer coords | TRELLIS.2 公开 shape-flow 配置 `pe_mode:"rope"`；`SLatFlowModel.forward` 仅在 `ape` 时加 PE。生成实际配置不是 APE+RoPE 同时开。见 T4–T6 | 是有官方实例的简化候选。NEXUS 的 parent-only self RoPE 是否足以与 VecSet 外部点云建立绝对空间对应仍未验证；不能推论删除位置投影必然更好 |
| P4：self RoPE + cross-attention query RoPE | 在 CA 的 Q 中显式编码查询坐标，K/V 条件仍由原输入生成；这种 Q-only 旋转可以使 query 的绝对位置影响条件读取 | Meta MeshFlow 的 `CrossAttention.forward` 实现了 Q-only RoPE，但 `build_meshflow_dit` 默认 `use_rope_in_cross_attention:false`。作者发布配置未随该源码 pin 提供，不确认权重实际打开此开关。见 M1–M4 | 可作为另一条位置进入条件交互的路径，而不等价于 P3。原 RoPE 类严格要求 head dim 能均分成三个偶数轴；128 不兼容，需明确本地适配（例如借用 TRELLIS.2 126+2 的分配）。不能把 VecSet 混合后的 tokens 当成保有明确 XYZ 的 keys，直接声称实现了空间相对 CA |

P1 和 P2 都携带加法绝对特征；P3 与 P4 的位置进入通路不同。Hunyuan Shape DiT 发布配置的 `use_pos_emb:false` 是 VAE latent 场景，不能单凭这一开关替 NEXUS 决定是否删除父格位置。

## B. 3D RoPE：先明确通道，再明确坐标单位

### 通道/频率的有据选择

| 方案 | 明确公式和布局 | 原实现状态与适配边界 |
|---|---|---|
| R1：TRELLIS.2 三轴等数 pairs + 剩余恒等通道 | `m=floor(head_dim/6)`；每轴 m 个 adjacent pairs；`ω_k=a/b^(k/m)`。默认 `(a,b)=(1,10000)`。128 时 21+21+21 pairs，最后 1 pair phase=0 | T4/T5 官方代码与公开生成配置确认实际使用；和当前的 126+2、base10000 同机制。FP32 做 complex rotation 后 cast 回输入 dtype。源位置直接是 integer sparse coords，**不含当前 c·256** |
| R2：MeshFlow 完整三轴均分 | `axis_dim=head_dim/3` 且 axis_dim 必须为偶数；每轴频率 `10000^(-2k/axis_dim)`，所有 head 维度旋转 | M1 官方代码机制明确，但不是128/head可直接实例化的类。若 hidden1536/head16→96，则三轴各32维/16pairs，满足源码约束；这是适配举例而非已核实权重 head 配置。更换 head 数会同时改变注意力分组，不能当作仅坐标倍率实验 |
| R3：老师候选的等分 + π·连续坐标 | 144 hidden、4 heads→36/head；每轴12维/6pairs，`ω_k=10000^(-k/6)`，相位 `π·xyz·ω_k` | 老师 `models.py` L16–41 与既有 RESULT forward contract 一致，但属于经有限前向核对选出的**逆向候选语义**，不是官方论文开源代码；不能单靠 state_dict 确认 π、heads 或 pair 排列。只能列作本地诊断参照，不能计作第三个官方复现来源 |

这里确有两种官方通道实现；没有为了凑三个来源而把 Hunyuan Shape DiT 宣称为 3D RoPE 使用者，或把原 RoFormer 一维序列实现宣称为 XYZ 分配依据。

### 坐标尺度的三种可比较定义

令 target child depth 为 d，`R_d=2^(d-1)` 是父格每轴格数，`p∈{0,...,R_d-1}`，`c=-1+2(p+0.5)/R_d`。

| 候选坐标 | 最高频 pair 的相邻父格相位增量 | 外部证据 / 推导标签 |
|---|---|---|
| C1：当前层 parent integer `r=p` | 1 rad；不同深度用各自父格单位 | 与 TRELLIS.2 直接使用整数 `coords` 的接口一致。把该接口用于跨深度 octree 是**本地适配**；原 SLAT 是固定 latent grid，不能从原项目推断该跨深度选择最优 |
| C2：统一参考 grid `r=(c+1)R_ref/2-0.5`，或只用于 SA 的 centered `r=cR_ref/2` | `R_ref/R_d` rad；物理距离统一，深度越粗跳得越大 | grid normalization 的逆变换是**适配推导**。R_ref=512 时 centered 形式正好是当前 c·256；未发现外部文献明确给 NEXUS 此倍率。它有单位解释，不能因此升级为作者设定 |
| C3：MeshFlow coarse 32-grid `r=clamp(floor((c+1)·16),0,31)` | 同 coarse voxel 为0，跨 voxel 为整数步 | MeshFlow `voxelize_pc` + `get_rope_cond` 的实际整数输出和论文 32³ 量化一致（M3、M5）。若转给 NEXUS parent center，是**本地迁移**：深层不同父格会共享位置编码，可能丢失细位置；MeshFlow 的目的在于缓解 GT vertices 与 surface-point 条件分布差异，不是证明 NEXUS 也需要这种量化 |

另一个低尺度参照是老师的 `r=πc`，相邻父格增量 `2π/R_d`；它使用连续 XYZ 噪声坐标，并不等于已知父格中心，且仅有候选级别证据。

**当前 c·256 的精确解释：** d=9 时 parent grid 为256，`c·256=2p-255`，相邻父格为2个相位单位，并不是 integer parent index p。512-grid 连续中心索引为 `g=(c+1)·256-0.5=2p+0.5`，因此当前 r=`g-255.5`。在仅对 self-attention 的 Q/K 同时使用 RoPE 时，共同平移在理想内积中抵消，这与512-grid世界单位一致；它仍不构成同数值外部来源。若只旋 CA 的 Q，这个原点平移不会被未旋的 K 抵消，必须重新写清原点。不得把归一化 c 或已经是整数 p 的输入再无条件乘256。

## C. 时间编码的三个官方候选

| 方案 | 编码、MLP、时间单位 | 实際源码状态和候选边界 |
|---|---|---|
| E1：DiT/TRELLIS 的256频率通道→SiLU MLP | `[cos(τ·10000^(-k/128)),sin(...)]`；Linear256→hidden→SiLU→Linearhidden→hidden | DiT 的 `TimestepEmbedder` 不在函数内乘1000；TRELLIS.2 的 flow trainer 传 `τ=1000t`（T7–T8），所以当前时间数值确有 flow 项目的直接先例。DiT 本身是图像 diffusion，时间索引语义仍不同；见 D1 |
| E2：Hunyuan3D 2.1 的 normalized t + hidden-wide sincos→GELU MLP | `τ=t∈[0,1]`、sin→cos、base10000；sinusoidal 维度=hidden；MLP hidden→4hidden→GELU→hidden。发布 hidden2048 故 2048→8192→2048 | `Timesteps` 默认 scale=1，发布 flow pipeline 将 scheduler timestep 除1000后传模型；time embedding 实际作为前置 token进入序列（H4–H6）。这不同于当前 `1000t` 的256通道。若只比较尺度，可保留现有 MLP并将τ换成t，但该拆分是**局部适配**，不是完整 Hunyuan结构 |
| E3：MeshFlow 的 hidden-wide sincos + 内部1000 + GELU MLP | sin→cos、base10000、内部 `τ=1000t`；MLP hidden→4hidden→GELU→hidden；time 输出供 block 调制 | `Timesteps`/`TimestepEmbedder` 与 `MeshFlowDiT.__init__` 明确构造该路径（M6）。这同时改变编码维度、激活与通道数，不能只归为“换时间倍率”。源码结构已核实；具体发布权重配置开关的确认范围不超出可见 build defaults |

不同项目中的 t 方向/flow target 不由时间编码决定。以上编码候选不能同时偷偷更改 noise→data / data→noise 定义、训练 t 分布或速度符号。cos/sin 顺序本身可由 Linear 输入权重排列吸收；复用权重时需要严格匹配，从头训练时它不是独立的表达能力证明。

## D. 老师最终点候选：结构来源与可借机制

老师包不是 NEXUS 作者源码。入口：[models.py](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:88)、[REPORT.md](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/REPORT.md:35)、[checkpoint_structure.json](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json)。本轮只读取既有记录，没有重新加载大权重、重跑采样或更新参数。

| 项目 | 既有源码/记录明确内容 | 可借鉴 / 不应迁入 |
|---|---|---|
| 最终配置依据 | `evidence/checkpoint_structure.json` 的 `results/point_diffusion/latest.pt.config` 和 candidate config 均为 text2048、width144、layers18、max_points274、prior_width512；shape 记录含 blocks.0…17，206 state entries/8,668,301 elements。`load_points` L149–153 强制读 checkpoint config | `teacher_point_INITIAL_config.json` / `teacher_early_config.json` 的6层不能覆盖最终18层；最终权重 SHA256记录 `e286272f2589defe17b72366d0cf88a44b839cf77b150cdf6257df27d2a5e4de`。这个 hash 来自已有结构审计记录，本轮没有再次 hash 原权重 |
| Global condition 的先后顺序 | PointFlow 接受一个已缓存的2048维文本全局向量；`text` = LN2048→Linear2048,144→SiLU→Linear144,144。denoise 中先得到此全局144维向量，再①加到每个 token 的 Fourier+slot hidden，②与时间144维向量相加后进入每块 AdaLN（models.py L98、L118–123）。REPORT L51 记录缓存来源为 raw prompt 最后隐层 masked mean；原文本编码器没有交付 | 可借“全局条件同时进入 token 内容与时间调制”的双路径。若将 VecSet `[B,1024,2048]`迁入，teacher-like局部方案是 **先 masked pool 得2048全局，再 LN/MLP 得1536，再加到 token/时间调制**；pool 是本地新增接口，老师 PointFlow 本身没实现点云 pooling。先对每个 token 做 LN/MLP 再 pool 是另一种新选择，非同一顺序；含 LN/SiLU 时一般不可交换 |
| 局部几何与 identity | noisy XYZ 做6-band `π2^k` Fourier39→144，加274×144 learned slot，再加 global text；self-attention用当前 noisy XYZ作3D RoPE | 只取明确 local geometry features 与 global conditioning 分工的机制。NEXUS token 是有父格坐标的8-child occupancy；不应引入按序号274个固定槽位、点数275类预测或把8通道改XYZ |
| 时间/调制 | 64维 `1000t` cos→sin；Linear64→144→SiLU→Linear144→144；与text全局相加；18个 self-attention→FFN block通过6组shift/scale/gate调制，无独立CA | “把条件也送入 modulation”可以是点云全局分支候选。heads4、π RoPE、epsilon1e-5、时间顺序、AdaLN排列属于已核对前向候选，不是 state形状唯一证明；不能宣称是NEXUS作者条件布局 |
| 最终输出 | `x1_hat=coordinate_prior(text)+α·denoise(x_t,t,text)`，α=`1.2198240256111603e-05`；coordinate_prior LN2048→Linear512→SiLU→Linear822→274×3；Euler使用 `(x1_hat-x)/(1-t)` | **不迁入 coordinate_prior**，也不把此结果当纯denoiser收敛。老师README明确结果主要来自50条训练集的文本坐标先验记忆；不是runtime GT查表，也不是泛化证明 |
| 状态/可续训边界 | 既有记录candidate的Adam仅有 coordinate_prior六项+α一项，slots step50000；metadata prior_steps160000，source_step267500与final frozen_denoiser_source_step258000有差别；final latest无optimizer/RNG | 这是先验局部优化状态，不是完整denoiser训练状态，不能拿其Adam slot计数当原network累计训练步，不能据此构造完整匹配恢复训练 |

既有候选纯噪声CPU点采样记录 `point_final/RESULT.json` 与 `point_final_seed98765/RESULT.json` 的两种seed均为点数50/50、坐标RMSE约1.1e-7；它们标注非原GPU RNG逐位重放。`cold_start_full_cascade/RESULT.json` 的 strict为41/50、Face FP51/FN67、F1≈0.9894。这些记录验证的是候选推理能力，不证明原训练源码恢复、原始denoiser独立成功或当前NEXUS占用模型架构应照搬。

## 固定版本、函数与行号链接

下面 GitHub 链接固定完整 commit；行号来自本轮本地 `nl -ba` 或固定 raw source 核对，不用 main 的浮动版本。论文正文与源码共同作为方案依据；源码可选开关不升级为作者实验实际选择。

- **H1** Hunyuan3D 2.1：论文[§3.1 ShapeVAE Fourier+Linear](https://arxiv.org/html/2506.15442v1#S3.SS1)。源码 commit `82920d643c0dc2f7bfd7255f45f62d386edfe60c`；[`FourierEmbedder.__init__/forward` L84–141](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L84-L141)。
- **H2** [`PointCrossAttentionEncoder.__init__` L559–581](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L559-L581)，Linear投影入口；[`ShapeVAE.__init__` L255–283](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/model.py#L255-L283)。
- **H3** 发布配置固定 HF revision `07d6dc9694e0ea942683bf6e3e374887d9f5b054`：[model `use_pos_emb:false` 与VAE `num_freqs:8,include_pi:false` L1–40](https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml#L1-L40)。
- **H4** [`Timesteps.forward` / `TimestepEmbedder` L63–123](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L63-L123)。
- **H5** [`HunYuanDiTPlain.__init__` time MLP构造 L583–590](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L583-L590)，[`forward` prepend time token L637–662](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L637-L662)。
- **H6** [`Hunyuan3DDiTFlowMatchingPipeline.__call__` normalized timestep L754–764](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/pipelines.py#L754-L764)，训练 [`Transport.training_losses` L173–175](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/diffusion/transport/transport.py#L173-L175)。
- **T1** TRELLIS 1 论文[§3.3](https://arxiv.org/html/2412.01506v1#S3.SS3)；源码commit `442aa1e1afb9014e80681d3bf604e8d728a86ee7`；[`AbsolutePositionEmbedder` L8–46](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/blocks.py#L8-L46)。
- **T2** [`SLatFlowModel.forward` sparse packing后加APE L240–257](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/structured_latent_flow.py#L240-L257)。
- **T3** [TRELLIS 1图像SLat配置 `pe_mode:ape` L4–19](https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json#L4-L19)。
- **T4** TRELLIS.2 论文[§3.3](https://arxiv.org/html/2512.14692v1#S3.SS3)、[附录A.2/Table5](https://arxiv.org/html/2512.14692v1#A1.SS2)；源码commit `75fbf0183001ed9876c8dbb35de6b68552ee08bd`；[`SparseRotaryPositionEmbedder` L7–58](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/sparse/attention/rope.py#L7-L58)。
- **T5** [TRELLIS.2 shape-flow配置1536/12heads/rope L4–18](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L4-L18)。
- **T6** [`SLatFlowModel.forward`仅APE分支加位置 L182–194](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L182-L194)，[`SparseMultiHeadAttention` 只支持self-RoPE L38–47](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/sparse/attention/modules.py#L38-L47)。
- **T7** [`TimestepEmbedder` L12–53](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/sparse_structure_flow.py#L12-L53)。
- **T8** [`SparseFlowMatchingTrainer.training_losses` 传 t·1000 L95–104](https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/trainers/flow_matching/sparse_flow_matching.py#L95-L104)。
- **M1** Meta MeshFlow 论文[§3.4](https://arxiv.org/html/2606.04621v1#S3.SS4)、[附录F](https://arxiv.org/html/2606.04621v1#S6)；源码commit `55f56f60e1bbf98d1c1991670ac998094d5f59ae`；[`RotaryPositionalEmbeddings` L92–197](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L92-L197)。
- **M2** [`CrossAttention.forward` Q-only RoPE L351–364](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L351-L364)。
- **M3** [`voxelize_pc` normalized XYZ→integer voxel indices L19–36](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L19-L36)，[`get_rope_cond` L293–296](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/pipelines/meshflow_pipeline.py#L293-L296)。
- **M4** [`build_meshflow_dit` `use_rope:true`, `rope_input_ndim:3`, `use_rope_in_cross_attention:false` defaults L859–873](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L859-L873)。类constructor自身的use_rope默认false，与build入口默认值不是同一个开关状态。
- **M5** [MeshFlow附录F的32³ coarse conditioning](https://arxiv.org/html/2606.04621v1#S6)；它不提供NEXUS的256 multiplier依据。
- **M6** [`Timesteps`/`TimestepEmbedder` L200–270](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L200-L270)，[`MeshFlowDiT.__init__` L725–730](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L725-L730)。
- **D1** DiT论文[§3.2](https://arxiv.org/html/2212.09748v2#S3.SS2)；源码commit `ed81ce2229091fd4ecc9a223645f95cf379d582b`；[`TimestepEmbedder` L27–64](https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27-L64)。

适配推导的公式可以检查，但该报告不宣称任何候选已提高当前 NEXUS 指标。优先的小范围实验应把额外位置分支、坐标尺度、时间尺度或全局条件注入分开改变，并保持同一份原始 condition tokens 与原有八叉树父格/occupancy接口。
