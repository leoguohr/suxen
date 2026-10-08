# NEXUS Vertex Diffusion 结构选型与实现依据

- 核查日期：2026-10-05。
- 对照版本：当前 C／N／T 使用的 R1-D9 冻结网络。
- 范围：网络结构、特征编码、归一化、初始化。优化器、loss、采样器、数据规模和运行策略不在本文范围内。
- 标记：**原实现**＝作者代码和实际配置可核实；**源码选项**＝代码支持，发布权重是否启用未确认；**适配**＝借用已有机制后为 NEXUS 调整；**老师候选**＝本地逆向复建代码。
- SA：自注意力；CA：跨注意力；FFN：逐 token 前馈网络；AdaLN：由条件生成归一化的缩放、偏移，部分实现还生成残差 gate。

## 0 保留的 NEXUS 接口

- 每个 token 表示一个父格的8个子格；网络输出8维 velocity。
- 同一网络处理不同八叉树深度；保留可学习 depth embedding 和3D RoPE。
- 点云与法向通过联合训练的 VecSet 编码；条件通过 CA 进入 DiT。
- 论文规格：8192个输入点、1024个条件 token、条件宽度2048、VecSet 8层、DiT约2B参数。
- 以下结构来源覆盖不同任务。移植到 NEXUS 后的效果尚需验证。
- 依据：[NEXUS §3.1.1][nexus31]、[§4.1][nexus41]。

## 1 DiT 的层数 宽度和注意力头数

- 当前：**36 blocks × hidden1536，12 heads，每头128维**。
- `36×1536×12`：本轮未找到原样采用这一组合的外部发布配置。宽1536、12 heads有直接先例；36层属于本地容量配置。

| 方案 | 可核查的结构 | 实现依据与适配范围 |
|---|---|---|
| 1 TRELLIS.2 型 | 30层、1536宽、12头 | **原实现**：[shape配置][t2cfg]。该模型同时采用共享调制和8192宽FFN；移植后重新统计参数量。 |
| 2 Hunyuan3D 2.1 型 | 21层、2048宽、16头 | **原实现**：[发布配置][hycfg]。原模型还有U型跳连及末6层MoE；只借层宽时不能沿用其参数量标签。 |
| 3 原 DiT XL 型 | 28层、1152宽、16头 | **原实现**：[DiT_XL_2][ditcfg]。原任务为图像latent，规模低于NEXUS约2B要求。 |
| 4 当前容量组合 | 36层、1536宽、12头 | **适配**：采用1536／12的已知搭配，按当前独立调制等部件配置36层。[当前构造][localdit] |

- 层宽配置依赖 block 组成、FFN宽度和调制是否共享；这些项目确定后再核算完整 DiT 参数量。

## 2 VecSet 的层数组合与聚合方式

- 当前：**1个CA聚合block＋7个SA block，总计8个**。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 聚合计入8层 | 1CA＋7SA | **适配**：聚合后接SA的路线有[Hunyuan][H21-encoder]、[TripoSG][T-encoder]依据；7层是按NEXUS总8层约束补定的数值。 |
| 2 聚合层单独计数 | 1CA作为tokenizer，之后8SA | **原实现**：[Hunyuan构造][H21-model]及[配置][H21-config]、[Michelangelo][M-encoder]及[配置][M-config]、[TripoSG][T-encoder]。物理上共9个blocks；只适用于把“8层”解释为聚合后的SA层数，不能标成总8层。 |
| 3 双分支聚合 | 两个CA分别读取uniform／salient点，输出相加后接SA | **适配**：[Dora双CA][D-encoder]、[配置][D-config]。若严格保留总8个blocks，可设计2CA＋6SA；该计数为本地适配，并需额外点分组依据。 |

- 原3DShape2VecSet的`encode`只有CA＋FFN；其SA堆栈位于`decode`。[源码][V-encode]
- Dora的两个CA各自做softmax，随后相加；与拼接所有点再做一次CA不同。源代码相加的是两个含残差的完整block输出。

### 2.1 VecSet 注意力头数

| 方案 | 2048宽条件编码器中的设定 | 实现依据与适配范围 |
|---|---|---|
| 1 保留当前分组 | CA／SA均16头，每头128维 | **本地选择**。所核点编码器中未找到原样的2048／16发布配置；128维/head在[Hunyuan DiT][hycfg]有跨角色先例。 |
| 2 保持64维每头 | CA／SA均32头，每头64维 | **适配**：[Hunyuan点encoder的1024／16][H21-config]、[Michelangelo的768／12][M-config]、[TripoSG的512／8][T-config]均为64维/head。扩宽到2048后为32头。 |
| 3 聚合用单头 | CA单头；后续SA单独设多头 | **适配**：[原VecSet单头CA][V-ca]。源宽512；扩到2048维单头需单独注明，不能当作已发布配置。 |

### 2.2 VecSet 末尾归一化

| 方案 | 结构 | 实现依据 |
|---|---|---|
| 1 末尾加 LN | 最后SA／FFN之后再做LayerNorm | **原实现**：[Hunyuan末LN开关的实际构造][H21-model]及[对应encoder][H21-encoder]、[Michelangelo配置][M-config]、[TripoSG][T-encoder]；当前采用。 |
| 2 不加独立末 LN | 残差block输出直接作为集合特征 | **适配**：[原VecSet encode][V-encode]没有独立末LN；其后另有KL头，NEXUS条件编码器不据此引入KL头。 |

## 3 VecSet 的 query 构造

- 当前：FPS选点，将这些点的XYZ Fourier与法向特征一起投影，用投影后的特征作query。

| 方案 | query 与条件数据的关系 | 实现依据与适配范围 |
|---|---|---|
| 1 FPS 坐标 query | 所选点仅用XYZ embedding作Q；完整点特征作K／V | **适配**：[原VecSet FPS＋PointEmbed][V-encode]。原模型没有法向，保留NEXUS法向并只送K／V是适配。 |
| 2 FPS 点特征 query | 所选XYZ与法向等特征共同作Q；与全点集共享投影 | **原实现**：[Hunyuan采样与编码][H21-sample]、[TripoSG encode][T-encode]。当前属于这一类。 |
| 3 可学习 query | 用固定数量可学习向量读取整个点云 | **原实现**：[Michelangelo CrossAttentionEncoder][M-encoder]、[实际配置][M-config]。在NEXUS中设1024个query、宽2048属于适配。 |

- Michelangelo对齐版本还有一个全局query；保留该机制时必须明确是否包含在1024个输出token中。[全局query构造][M-global]

## 4 点坐标与法向的特征编码

- 当前：8组`π×2^k`正余弦、原XYZ、原法向，共54维，线性投影到2048。

| 方案 | 频率与输入 | 实现依据与适配范围 |
|---|---|---|
| 1 含 π 的 Fourier | 8组`π×2^k`，拼接原XYZ与原法向 | **适配**：[原VecSet PointEmbed][vecembed]确证π频率；加入法向遵循NEXUS输入要求。 |
| 2 不含 π 的 Fourier | 8组`2^k`，拼接原XYZ与原法向 | **原实现依据**：[Michelangelo配置][M-config]、[TripoSG配置][T-config]及[编码][T-encode]。 |

- 两者输入维数相同，最高角频率分别为128π与128。频率改变不等于法向输入改变。
- 法向放置有两种实际路线：FPS点特征query中同时含法向；learned query本身不含法向，由数据K／V提供法向。来源分别见第3节方案2／3。
- **未列为可直接采用的方案**：法向也做Fourier编码。所核Hunyuan／Dora相关分支没有发布启用依据，并存在接口或维度接线问题。

### 4.1 条件编码器中的 QK 归一化

| 方案 | 结构 | 实现依据与适配范围 |
|---|---|---|
| 1 保留 Pre-LN 且不加 QK norm | 仅在attention／FFN之前做LN | **原实现**：[原VecSet][V-preln]、[Michelangelo][M-block]、[TripoSG][T-encoder]；当前采用。 |
| 2 增加逐头 QK LayerNorm | Q、K各自在head维度做LayerNorm | **原实现**：[Hunyuan点encoder][H21-qknorm]，由[发布配置qk_norm=true][H21-config]启用。 |
| 3 增加逐头 QK RMSNorm | Q、K各自做RMS类归一化 | **跨角色适配**：[Hunyuan发布配置][hycfg]启用RMS，由[norm选择][hynormchoice]传到[QK norm][hyqknorm]；所核Hunyuan点encoder使用的是LayerNorm。 |

## 5 DiT block 组织与 CA 分支

- 当前：**SA → CA → FFN**；CA前有独立LayerNorm。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 平铺且 CA 前归一化 | SA → LN＋CA → FFN | **原实现**：[TRELLIS.2 block][t2block]。当前代码与此类结构接近。 |
| 2 平铺且 CA 直接读取残差 | SA → CA → FFN；CA直接接收SA残差结果 | **原实现**：[PixArtBlock][pixblock]，没有额外的CA前LN。 |
| 3 U 型长跳连 | block内保持SA → CA → FFN；后半层融合前半层特征 | **原实现**：[Hunyuan block][hyblock]、[skip存取][hyskip]。需保持父格token顺序对齐。 |

- **排列证据范围**：以上兼容CA的实现均采用SA → CA → FFN。本轮未找到这些作者实现中支持CA → SA → FFN等其他排列的依据。

## 6 时间调制的参数共享

- 当前：每个block独立生成6组调制向量，对SA和FFN分别产生shift、scale、gate。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 逐块独立 AdaLN | 每层一个`SiLU → Linear(H,6H)` | **原实现**：[TRELLIS 1默认配置][t1defaults]、[对应block][t1block]。当前采用这一大类。 |
| 2 AdaLN single | 全网络共享一次6H投影；各block加自己的6H可学习偏置 | **原实现**：[TRELLIS.2][t2mod]、[PixArt][pixtime]。attention和FFN参数仍逐块独立。 |
| 3 时间 token | 时间嵌入作为额外token参与SA，最后移除 | **原实现**：[TripoSG forward][tripotime]、[Hunyuan forward][hytime]。移植时需单独处理该token的mask与RoPE。 |

## 7 全局条件是否参与调制

- 当前：VecSet tokens进入CA；AdaLN只接收时间特征。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 条件仅走 CA | 时间控制AdaLN；完整条件tokens供CA读取 | **原实现**：[TRELLIS.2 block][t2block]。保持当前条件入口。 |
| 2 CA 加全局条件 AdaLN | VecSet摘要经投影后与时间嵌入相加，送入各层AdaLN；保留CA | **适配**：[原DiT时间＋类别调制][ditcond]、[老师候选全局文本＋时间调制][teachercond]。从VecSet得到摘要的方式需要另选。 |
| 3 CA 加全局条件时间 token | 对条件进行attention pooling，与时间特征融合成时间token；CA仍读取全部tokens | **源码选项**：[Hunyuan pooling构造][hypool]、[forward][hytime]；[发布配置][hycfg]实际关闭此选项。迁移到VecSet需适配输入宽度。 |

- 方案2中的摘要可采用有效token均值，或单独学习的全局query。均值池化＋拟新增的共享线性投影这一完整组合属于本地适配。
- 老师候选输入为全局文本特征；其代码未证明VecSet均值池化的效果。

## 8 CA 残差的门控

- 当前：CA采用普通残差；SA和FFN具有时间gate。R1历史上还采用了CA输出零初始化，见第18节；普通残差和零初始化可以同时使用。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 无 CA gate | `x ← x + CA(LN(x),condition)` | **原实现**：[TRELLIS.2][t2block]。当前方式。 |
| 2 增加 CA 时间 gate | 调制输出由6H增至7H；新增H维向量乘CA输出 | **原实现**：[Meta MeshFlow 7H构造][metagateinit]、[forward][metagate]。该额外gate没有同时增加CA的shift／scale。 |
| 3 无动态 gate 且 CA 输出零初始化 | forward保持普通CA残差；初始化时CA输出投影置零 | **原实现**：[PixArt初始化][pixinit]。这控制初始状态，与方案2的动态gate是两个不同选择。 |

## 9 QK 归一化

- 当前：DiT的SA与CA都做QK RMS类归一化；VecSet不做。

| 方案 | 作用位置 | 实现依据与适配范围 |
|---|---|---|
| 1 全部关闭 | Q／K投影后直接进入attention | **原实现**：[PixArt SA][pixsa]、[CA][pixca]。主干LayerNorm仍保留。 |
| 2 仅 SA 开启 | SA开启QK RMS；CA关闭 | **原实现**：[TRELLIS 1配置][t1cfg]和[cross默认值][t1defaults]。 |
| 3 SA 与 CA 均开启 | 两处都做QK RMS类归一化 | **原实现**：[TRELLIS.2配置][t2cfg]、[Hunyuan发布配置][hycfg]。 |
| 4 QK LayerNorm | Q／K按每头特征维做LayerNorm | **源码选项**：[Hunyuan norm选择][hynormchoice]；点云编码器中的实际用例见第4节。 |

- RMS实现细分：

| 选择 | 公式和参数 | 依据 |
|---|---|---|
| 每头独立gain | gain为`[heads,head_dim]`；TRELLIS使用`normalize(q)×√head_dim` | [TRELLIS.2 MultiHeadRMSNorm][t2qknorm] |
| 各头共享gain | `nn.RMSNorm(head_dim, eps=1e-6)`；gain为`[head_dim]` | [Hunyuan QK norm][hyqknorm] |

- 当前采用每头独立gain，但分母为`√(mean(q²)+1e-6)`。它与TRELLIS的epsilon规则有差别，属于同类机制的本地数值实现。

## 10 QKV 线性层的 bias

- 当前：QKV投影带bias。

| 方案 | 结构 | 实现依据 |
|---|---|---|
| 1 保留 bias | Q、K、V投影包含可学习偏置 | **原实现**：[DiT配置调用][ditblock]。 |
| 2 去掉 bias | Q、K、V投影不含偏置 | **原实现**：[Hunyuan发布配置][hycfg]、[TripoSG实际构造][tripoconstruct]。输出投影是否有bias另行确定。 |

## 11 父格绝对位置如何进入网络

- 当前：归一化父格中心XYZ，经过8组`π×2^k` Fourier编码得到51维，再线性投影并加到token；同时保留SA的RoPE。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 Fourier 加学习投影 | `XYZ＋sin／cos → Linear → 加到token` | **适配**：[3DShape2VecSet PointEmbed][vecembed]、[Hunyuan点编码][hypoint]。源模块用于点／查询编码，移到父格DiT需要明确坐标单位。 |
| 2 固定三轴 sincos APE | 直接生成hidden宽的位置向量并加到token | **原实现**：[TRELLIS 1 APE][t1ape]、[实际配置][t1slatcfg]。原输入是整数grid坐标。 |
| 3 仅通过 SA RoPE 输入位置 | 去掉加法位置投影，保留RoPE和depth embedding | **原实现依据**：[TRELLIS.2 rope配置][t2cfg]、[forward][t2posforward]；用于NEXUS父格条件交互属于适配。 |
| 4 为 CA query 也加 RoPE | SA保留RoPE；CA的Q旋转，条件K／V保持原形式 | **源码选项**：[Meta CrossAttention][metacarope]；[builder默认关闭][metaropedefault]，发布启用状态未确认。 |

- 方案3在NEXUS中能否充分定位父格与外部点云的对应关系，尚未验证。
- 频率来源：原VecSet采用π频率；Hunyuan发布点编码配置采用`include_pi=false`。[发布配置][hycfg]
- VecSet聚合后的tokens没有自动保留一一对应的XYZ；方案4不能直接解释成Q、K空间相对位置匹配。

## 12 三维 RoPE 的通道分配

- 当前：每头128维；每轴21对，共126维旋转，余下2维保持不变；base=10000。

| 方案 | 通道规则 | 实现依据与适配范围 |
|---|---|---|
| 1 允许剩余通道 | 每轴`floor(head_dim/6)`对，余下通道不旋转 | **原实现**：[TRELLIS.2 SparseRotaryPositionEmbedder][t2rope]。直接支持当前128维/head。 |
| 2 严格三轴均分 | 每头维度须被6整除，全部通道参与旋转 | **原实现**：[Meta RotaryPositionalEmbeddings][metarope]。当前128不满足；例如1536宽配16头得到96维/head，是可行的本地适配组合。 |
| 3 严格均分加连续坐标的 π 相位 | 三轴等分，使用`π×xyz×frequency` | **老师候选**：[rope_3d][teacherrope]。候选为36维/head；不能用其小模型尺寸代替NEXUS容量规格。 |

- 两种官方布局均有源码依据；老师一项补充连续坐标相位的本地候选依据。
- 原RoFormer给出旋转机制；XYZ轴分配仍需上述三维实现补充。

## 13 RoPE 的坐标单位

- 记目标子层深度为`d`，父格分辨率`R_d=2^(d−1)`，父格整数坐标为`p`，归一化中心为`c=−1+2(p+0.5)/R_d`。
- 当前传入`r=256c`。在depth9时，`r=2p−255`；相邻父格的最高频相位差为2弧度。

| 方案 | 送入 RoPE 的坐标 | 依据与取舍 |
|---|---|---|
| 1 当前层整数格单位 | `r=p` | **适配**：[TRELLIS.2直接使用integer coords][t2rope]。每层相邻格间隔为1；跨深度的物理距离尺度不同。 |
| 2 统一参考格单位 | `r=c×R_ref/2`；当前`R_ref=512` | **适配推导**：统一物理尺度对应固定参考网格。当前`×256`具有这一单位解释；未找到外部模型原样使用此NEXUS设定。 |
| 3 归一化连续坐标单位 | `r=πc` | **老师候选机制＋适配**：[连续XYZ RoPE][teacherrope]。原候选用当前带噪XYZ；NEXUS改为已知父格中心。 |

- `p`、`c`、`256c`产生不同频谱，必须成套记录坐标单位与RoPE频率。
- SA中同时旋转Q／K时，共同原点平移在理想内积中抵消；仅旋转CA的Q时，没有这一抵消关系。

## 14 时间特征编码

- 当前：256维cos／sin，输入`1000t`，base=10000；`Linear(256,H) → SiLU → Linear(H,H)`。

| 方案 | 编码与 MLP | 实现依据与适配范围 |
|---|---|---|
| 1 固定256维编码 | cos在前、sin在后；256→H→H，SiLU | **原实现**：[DiT TimestepEmbedder][dittime]、[TRELLIS.2编码][t2time]；TRELLIS的[调用处传1000t][t2timescale]。 |
| 2 hidden宽编码与归一化时间 | sin在前、cos在后；输入`t∈[0,1]`；H→4H→H，GELU | **原实现**：[Hunyuan编码器][hytimestep]、[实际4H构造][hytimebuild]、[时间入口][hytimescale]。结果用作time token。 |
| 3 hidden宽编码与1000倍时间 | sin在前、cos在后；内部使用1000t；H→4H→H，GELU | **源码构造**：[Meta编码器][metatime]、[实际4H构造][metatimebuild]。编码结果用于block调制。 |
| 4 较短的64维编码 | cos／sin各32维；1000t；64→H→H，SiLU | **老师候选**：[time_embedding][teachertime]。原H=144，扩大到当前H属于适配。 |

- 时间编码维度、倍率、MLP宽度、激活和注入位置是不同选择；整套移植时需一起注明。
- 上述候选保持当前时间方向和velocity含义；它们不规定训练时间分布。

## 15 FFN 的激活和宽度

- 当前：`H → 4H → H`，GELU的tanh近似；VecSet与DiT均采用这一形式。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 标准 GELU FFN | `Linear(H,4H) → GELU → Linear(4H,H)` | **原实现**：[DiT][ditblock]、[PixArt][pixblock]。 |
| 2 更宽的 GELU FFN | 中间层约`5.3334H`；H=1536时为8192 | **原实现**：[TRELLIS.2配置][t2cfg]、[FeedForwardNet][t2ffn]。 |
| 3 GEGLU | 输入投影分为两路4H；一路乘另一条的GELU门，再投回H | **原实现**：[3DShape2VecSet FeedForward][vecffn]。来源为AE模块；迁移到DiT属于适配。 |

- 参数关系：忽略bias，4H标准FFN约`8H²`；两路4H的GEGLU约`12H²`。两者不能按相同参数量解释。
- GELU精确形式也有两种先例：DiT／PixArt使用`tanh`近似；[Hunyuan dense MLP][hyffn]使用`nn.GELU()`默认形式。

## 16 主干归一化

- 当前DiT：Pre-LN；SA／FFN的LN不带静态affine，CA的LN带affine；`eps=1e-6`。当前VecSet各LN均带affine。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 Pre-LN 配合 AdaLN | SA／FFN使用无affine LN，由条件生成缩放与偏移 | **原实现**：[DiT][ditblock]、[TRELLIS.2][t2block]。 |
| 2 带 affine 的 Pre-LN | LN保留自身的可学习缩放与偏移；时间另走time token | **原实现**：[Hunyuan block][hynorm]、[TripoSG block][triponorm]。 |
| 3 Pre-RMSNorm | 主干归一化按RMS处理，不减均值 | **源码选项**：[Meta MeshFlow norm分支][metanorm]。本轮未取得发布配置来确认启用。 |

- epsilon候选：`1e-6`有DiT／Hunyuan依据；`1e-5`有[TripoSG实际构造][tripoconstruct]和[老师候选block][teacherblock]依据。
- 主干RMSNorm和第9节的QK RMSNorm是两个独立位置。

## 17 输出头

- 当前：无affine LayerNorm后接`Linear(H,8)`。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 普通 LN 加线性投影 | 最后归一化，再逐token投影 | **原实现**：[TRELLIS.2输出][t2out]。当前属于这一类，epsilon需明确。 |
| 2 条件化 final AdaLN | 条件经独立2H投影生成最终shift／scale，再输出 | **原实现**：[DiT FinalLayer][ditout]。 |
| 3 共享风格 final 调制 | 可学习`[2,H]`表加时间向量，得到最终shift／scale | **原实现**：[PixArt T2IFinalLayer][pixout]。 |
| 4 带 affine 的 LN 加投影 | LN保留静态参数，随后输出 | **原实现**：[Hunyuan FinalLayer][hyout]、**老师候选**：[输出头][teacherout]。 |

- 所有候选在NEXUS中仍输出8维velocity。源模型的图像patch输出、3维XYZ和latent通道数均需替换。
- 仅采用time-token方案时才移除额外时间token；普通输出头不能删去首个父格。

## 18 初始化

- 当前R1的历史初始化：SA／FFN调制输出零初始化、最终输出零初始化；另将CA输出投影置零，depth embedding和time MLP采用`std=0.02`。[R1初始化入口][localr1]
- 当前C／N／T从已训练权重恢复；上述数值描述其初始化规则。

| 方案 | 初始化规则 | 实现依据与适配范围 |
|---|---|---|
| 1 DiT AdaLN Zero | 逐块调制输出、final调制及最终输出置零；time MLP std=0.02 | **原实现**：[DiT initialize_weights][ditinit]。若要求整块初始化为恒等映射，引入普通CA后需额外处理CA的初始残差。 |
| 2 PixArt 风格 | CA输出与最终输出置零；共享时间投影和各层调制表保留小随机值 | **原实现**：[PixArt初始化][pixinit]。 |
| 3 按深度缩小残差投影 | attention输出与FFN第二层的std按`1/√(5LH)`缩放 | **原实现**：[TRELLIS.2 scaled初始化][t2init]。共享调制投影和最终输出置零；各层调制偏置仍随机。 |
| 4 所有三条分支 gate 初始为零 | SA、CA、FFN共7H调制输出置零 | **原实现**：[Meta MeshFlow gate初始化][metazero]。U型skip融合另有作用，不能据此宣称整个网络恒等。 |

- CA输出零初始化、时间gate零初始化、残差投影缩放控制不同部件；组合时需要逐项说明。

## 19 Dropout 与残差随机丢弃

- 当前：attention dropout=0；FFN dropout=0；没有DropPath。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 不随机丢弃分支 | attention与FFN保持确定性的完整残差 | **原实现**：[DiT block][ditblock]、[PixArt默认block][pixblock]。 |
| 2 残差 DropPath | 按样本随机跳过SA／FFN残差；已有`p=0.1`实例 | **原实现**：[3DShape2VecSet decoder构造][vecdroppath]。用于条件encoder或DiT属于适配。 |
| 3 逐层增加 DropPath | 浅层概率低，深层概率高 | **源码选项**：[Meta逐层dpr][metadrop]。本轮未确认发布模型使用的最终概率。 |

- DropPath作用于残差分支；attention dropout作用于注意力概率，两者需分别命名。

## 20 组合约束

- 层数、宽度、head数、FFN宽度、调制共享方式共同决定参数量。
- RoPE轴分配必须与每头维度匹配；坐标单位和频率定义必须成套记录。
- time token、逐块AdaLN、共享AdaLN应先确定一种主时间入口，再选择输出头和初始化。
- 增加全局条件调制时，保留NEXUS规定的CA接口；VecSet摘要方式属于额外结构选择。
- 原3DShape2VecSet和Hunyuan ShapeVAE提供点云聚合模块依据；其AE／VAE训练职责与NEXUS条件编码器不同。
- 本文提供结构候选与来源，不按其他任务的成绩给NEXUS候选排序。
## 21 论文与源码来源索引

| 来源 | 论文 | 本文采用的实现依据 |
|---|---|---|
| NEXUS | [Native Mesh Generation with Diffusion](https://arxiv.org/abs/2607.13563v1) | 任务接口与公开规模规格 |
| DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | AdaLN、时间编码、输出头、初始化、层宽 |
| 3DShape2VecSet | [3D Shape Representation for Neural Fields and Generative Diffusion Models](https://arxiv.org/abs/2301.11445) | FPS查询、Fourier、GEGLU、点云聚合 |
| Michelangelo | [Conditional 3D Shape Generation based on Shape Image Text Aligned Latent Representation](https://arxiv.org/abs/2306.17115) | learned queries、点特征编码、全局query |
| Hunyuan3D 2.1 | [From Images to High Fidelity 3D Assets with Production Ready PBR Material](https://arxiv.org/abs/2506.15442) | 点编码器、time token、QK norm、U型skip |
| TRELLIS | [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506) | 独立调制、CA前LN、APE |
| TRELLIS.2 | [Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692) | 共享调制、RoPE、scaled初始化 |
| PixArt α | [Fast Training of Diffusion Transformer for Photorealistic Text to Image Synthesis](https://arxiv.org/abs/2310.00426) | AdaLN single、CA直接残差、输出调制 |
| TripoSG | [High Fidelity 3D Shape Synthesis using Large Scale Rectified Flow Models](https://arxiv.org/abs/2502.06608) | 点编码器、time token、norm与bias实际调用 |
| Meta MeshFlow | [Efficient Artistic Mesh Generation via MeshVAE and Flow based Diffusion Transformer](https://arxiv.org/abs/2606.04621) | CA gate、RoPE、时间编码及明确标注的源码选项 |
| Dora | [Sampling and Benchmarking for 3D Shape Variational Auto Encoders](https://arxiv.org/abs/2412.17808) | uniform／salient双CA聚合 |
| 老师候选 | [本地models.py](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:88) | 全局条件、时间调制、连续坐标RoPE；逆向候选级依据 |

### 21.1 老师候选的使用边界

- 最终点模型配置依据：[checkpoint_structure.json](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json)。最终为18层、hidden144；早期6层配置不代表最终模型。
- 输入：缓存的2048维全局文本特征。该向量经LN／MLP后，同时加入token内容与时间调制。[实际forward][teachercond]
- 完整输出包含文本坐标prior和缩放后的去噪分支。已有记录中分支系数约`1.22e-5`，整体结果主要来自先验记忆；本文仅引用局部结构。[候选记录](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/REPORT.md:35)
- VecSet先pool再LN／MLP是迁移接口。LN／MLP后再pool是另一种结构，二者一般不可交换。
- 点数分类、固定点槽位和坐标prior均超出本次结构选项范围。

### 21.2 当前代码定位

- 条件编码器：[VertexConditionEncoder](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:219)。
- DiT block：[\_VertexBlock](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:261)。
- 输入与位置／时间融合：[VertexDiT.forward](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:338)。
- 文件SHA256：`39491f94fadec84797b59a92d969dd923353abeea29a91a02e1a0f904ae1f386`，与E2 C冻结代码清单一致。
- 本文引用的GitHub源码均固定到commit。公开配置标为已启用；无法核实发布状态的分支保留“源码选项”标记。

[D-config]: https://github.com/Seed3D/Dora/blob/a166e21e900cf1e1230a67db2efc8b5a2c022e7d/pytorch_lightning/configs/shape-autoencoder/Dora-VAE-test.yaml#L23-L41
[D-encoder]: https://github.com/Seed3D/Dora/blob/a166e21e900cf1e1230a67db2efc8b5a2c022e7d/pytorch_lightning/craftsman/models/autoencoders/michelangelo_autoencoder.py#L31-L160
[H21-config]: https://huggingface.co/tencent/Hunyuan3D-2.1/blob/0b94677654c57bb9a6b6845cd7b704ccf551d327/hunyuan3d-vae-v2-1/config.yaml#L1-L19
[H21-encoder]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L521-L583
[H21-model]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/model.py#L238-L283
[H21-qknorm]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L195-L255
[H21-sample]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L585-L716
[M-block]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/modules/transformer_blocks.py#L77-L115
[M-config]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/configs/aligned_shape_latents/shapevae-256.yaml#L1-L19
[M-encoder]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/tsal/sal_perceiver.py#L20-L112
[M-global]: https://github.com/NeuralCarver/Michelangelo/blob/6d83b0bacef92715dd5179d45647ed9a3d39bc95/michelangelo/models/tsal/sal_perceiver.py#L309-L368
[T-config]: https://huggingface.co/VAST-AI/TripoSG/blob/2c1c516d22d58db486a058d98d31bb6177344e06/vae/config.json#L1-L14
[T-encode]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/autoencoders/autoencoder_kl_triposg.py#L439-L457
[T-encoder]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/autoencoders/autoencoder_kl_triposg.py#L26-L87
[V-ca]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L308-L326
[V-encode]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L338-L394
[V-preln]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L34-L106
[ditblock]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122
[ditcfg]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L328-L344
[ditcond]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L233-L246
[ditinit]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L182-L216
[ditout]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L125-L142
[dittime]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27-L64
[hyblock]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L378-L405
[hycfg]: https://huggingface.co/tencent/Hunyuan3D-2.1/blob/07d6dc9694e0ea942683bf6e3e374887d9f5b054/hunyuan3d-dit-v2-1/config.yaml#L1-L40
[hyffn]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L126-L135
[hynorm]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L331-L354
[hynormchoice]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L571-L574
[hyout]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L450-L465
[hypoint]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/autoencoders/attention_blocks.py#L559-L581
[hypool]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L595-L602
[hyqknorm]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L161-L168
[hyskip]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L657-L666
[hytime]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L637-L662
[hytimebuild]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L583-L590
[hytimescale]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/pipelines.py#L754-L764
[hytimestep]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L63-L123
[localdit]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:296
[localr1]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_b1_single_20260915/review_bundle/code/scripts/train_vertex_a100_b1.py:34
[metacarope]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L351-L364
[metadrop]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L737-L755
[metagate]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L578-L607
[metagateinit]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L538-L539
[metanorm]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L714-L716
[metarope]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L92-L197
[metaropedefault]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L859-L873
[metatime]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L200-L270
[metatimebuild]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L725-L730
[metazero]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L475-L477
[nexus31]: https://arxiv.org/html/2607.13563v1#S3.SS1.SSS1
[nexus41]: https://arxiv.org/html/2607.13563v1#S4.SS1
[pixblock]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L25-L54
[pixca]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L29-L57
[pixinit]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L176-L210
[pixout]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L172-L188
[pixsa]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L109-L125
[pixtime]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L121-L135
[t1ape]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/blocks.py#L8-L46
[t1block]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/modulated.py#L99-L149
[t1cfg]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L4-L17
[t1defaults]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/sparse_structure_flow.py#L68-L73
[t1slatcfg]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json#L4-L19
[t2block]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L104-L157
[t2cfg]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L4-L18
[t2ffn]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/blocks.py#L49-L59
[t2init]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L128-L167
[t2mod]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L132-L144
[t2out]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L193-L199
[t2posforward]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L182-L194
[t2qknorm]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/modules.py#L9-L16
[t2rope]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/sparse/attention/rope.py#L7-L58
[t2time]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/sparse_structure_flow.py#L12-L53
[t2timescale]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/trainers/flow_matching/sparse_flow_matching.py#L95-L104
[teacherblock]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:44
[teachercond]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:114
[teacherout]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:104
[teacherrope]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:30
[teachertime]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:24
[tripoconstruct]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L440-L462
[triponorm]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L173-L202
[tripotime]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L664-L718
[vecdroppath]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L204-L225
[vecembed]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L109-L139
[vecffn]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L51-L68
