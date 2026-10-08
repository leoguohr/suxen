# NEXUS Vertex Diffusion 结构依据与候选方案

- 分类对象：NEXUS v1原论文中可确定与不可确定的**具体结构细节**；同一模块可以分属两部分。目标保持为点云条件下的八叉树Vertex Diffusion。
- 当前实现：A24000与E2 C／N／T对应的本地R1-D9冻结代码。当前代码定位与来源索引见文末。
- 范围：网络结构、输入输出表示、特征编码与归一化。学习率、loss配方、采样器、初始化数值、训练预算、GPU与存储策略不列入选型。
- 论文定位：采用13页[原论文PDF][paper4]的页码；L编号来自[逐页非空文本行表][paperlines]，供复核使用，论文原版没有行号。左右栏分别标明。
- 依据等级：**原实现**＝官方源码与实际配置；**源码选项**＝代码支持但发布启用未确认；**适配**＝迁移机制后为NEXUS补定的结构；**老师候选**＝本地逆向复建，不能代替NEXUS论文证据。
- 老师对照口径：最终18层、hidden144的文本条件点模型；原生表示为连续XYZ，包含点数分类、点槽位及坐标prior。CAD50点生成成功有原报告和候选重放支持；原报告明确结果由学习的坐标prior主导。成功范围、完整路径与借鉴优先级见第2.4节。[原报告][teacher-original-result]、[PointFlow][teacherpoint]

## 一 结构细节未能通过原论文确定

- 原文引用DiT或VecSet，能够确定结构来源；引用本身不足以确定所用仓库的全部默认参数。
- 下列候选提供可核查的机制。原有A–G对照继续作为证据；本文不据老师整网成功宣布某个局部结构有效。借鉴优先级是面向当前八叉树任务的判断，不改变论文已确定／未确定的分类；适配范围见各节与第2.4节。

### 1.1 DiT 的层数 宽度与注意力头数

- **原论文边界**：§4.1明确Vertex DiT约2B参数；未给层数、hidden宽、head数和三者组合。[p6左栏L051–054][paper6]
- **老师候选**：最终点模型18 blocks、hidden144；候选forward为4 heads、每头36维。层数和宽度有[权重结构记录][teachershape]支持；head拆分属于[候选执行约定][teacherblock]。

- 当前：**36 blocks × hidden1536，12 heads，每头128维**。
- `36×1536×12`：本轮未找到原样采用这一组合的外部发布配置。宽1536、12 heads有直接先例；36层属于本地容量配置。

| 方案 | 可核查的结构 | 实现依据与适配范围 |
|---|---|---|
| 1 TRELLIS.2 型 | 30层、1536宽、12头 | **原实现**：[shape配置][t2cfg]。该模型同时采用共享调制和8192宽FFN；移植后重新统计参数量。 |
| 2 Hunyuan3D 2.1 型 | 21层、2048宽、16头 | **原实现**：[发布配置][hycfg]。原模型还有U型跳连及末6层MoE；只借层宽时不能沿用其参数量标签。 |
| 3 原 DiT XL 型 | 28层、1152宽、16头 | **原实现**：[DiT_XL_2][ditcfg]。原任务为图像latent，规模低于NEXUS约2B要求。 |
| 4 当前容量组合 | 36层、1536宽、12头 | **适配**：采用1536／12的已知搭配，按当前独立调制等部件配置36层。[当前构造][localdit] |

- 层宽配置依赖 block 组成、FFN宽度和调制是否共享；这些项目确定后再核算完整 DiT 参数量。
- 老师18层小模型的最终成绩包含独立坐标prior贡献，不能据此认定18×144去噪主干足以替代NEXUS约2B占据网络。[成功路径说明][teacher-original-prior]

### 1.2 VecSet 的层数组合与聚合方式

- **原论文边界**：§4.1明确VecSet为8层、hidden2048；未给CA／SA分配，也未说明聚合层是否计入8层。[p6左栏L055–059][paper6]
- **老师候选**：点模型没有VecSet。输入是2048维缓存文本向量，经LN和MLP投影到144维。[PointFlow][teacherpoint]

- 当前：**1个CA聚合block＋7个SA block，总计8个**。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 聚合计入8层 | 1CA＋7SA | **适配**：聚合后接SA的路线有[Hunyuan][H21-encoder]、[TripoSG][T-encoder]依据；7层是按NEXUS总8层约束补定的数值。 |
| 2 聚合层单独计数 | 1CA作为tokenizer，之后8SA | **原实现**：[Hunyuan构造][H21-model]及[配置][H21-config]、[Michelangelo][M-encoder]及[配置][M-config]、[TripoSG][T-encoder]。物理上共9个blocks；只适用于把“8层”解释为聚合后的SA层数，不能标成总8层。 |
| 3 双分支聚合 | 两个CA分别读取uniform／salient点，输出相加后接SA | **适配**：[Dora双CA][D-encoder]、[配置][D-config]。若严格保留总8个blocks，可设计2CA＋6SA；该计数为本地适配，并需额外点分组依据。 |

- 原3DShape2VecSet的`encode`只有CA＋FFN；其SA堆栈位于`decode`。[源码][V-encode]
- Dora的两个CA各自做softmax，随后相加；与拼接所有点再做一次CA不同。源代码相加的是两个含残差的完整block输出。

### 1.3 VecSet 注意力头数

- **原论文边界**：§4.1明确hidden2048，但没有head数。[p6左栏L055–059][paper6]
- **老师候选**：点模型没有VecSet。输入是2048维缓存文本向量，经LN和MLP投影到144维。[PointFlow][teacherpoint]

- 当前：16头，每头128维。[当前条件编码器][localencoder]


| 方案 | 2048宽条件编码器中的设定 | 实现依据与适配范围 |
|---|---|---|
| 1 保留当前分组 | CA／SA均16头，每头128维 | **本地选择**。所核点编码器中未找到原样的2048／16发布配置；128维/head在[Hunyuan DiT][hycfg]有跨角色先例。 |
| 2 保持64维每头 | CA／SA均32头，每头64维 | **适配**：[Hunyuan点encoder的1024／16][H21-config]、[Michelangelo的768／12][M-config]、[TripoSG的512／8][T-config]均为64维/head。扩宽到2048后为32头。 |
| 3 聚合用单头 | CA单头；后续SA单独设多头 | **适配**：[原VecSet单头CA][V-ca]。源宽512；扩到2048维单头需单独注明，不能当作已发布配置。 |

### 1.4 VecSet 末尾归一化

- **原论文边界**：§4.1没有encoder末尾Norm配置。[p6左栏L055–059][paper6]
- **老师候选**：点模型没有VecSet。输入是2048维缓存文本向量，经LN和MLP投影到144维。[PointFlow][teacherpoint]

- 当前：末尾加带affine的LayerNorm。[当前条件编码器][localencoder]


| 方案 | 结构 | 实现依据 |
|---|---|---|
| 1 末尾加 LN | 最后SA／FFN之后再做LayerNorm | **原实现**：[Hunyuan末LN开关的实际构造][H21-model]及[对应encoder][H21-encoder]、[Michelangelo配置][M-config]、[TripoSG][T-encoder]；当前采用。 |
| 2 不加独立末 LN | 残差block输出直接作为集合特征 | **适配**：[原VecSet encode][V-encode]没有独立末LN；其后另有KL头，NEXUS条件编码器不据此引入KL头。 |

### 1.5 VecSet 的 query 构造

- **原论文边界**：§3.1.1及§4.1只给VecSet与1024个输出token；未规定query来源。[p4右栏L013–016][paper4]、[p6左栏L055–059][paper6]
- **老师候选**：没有点云聚合query。`slot[274,144]`是输出点槽位embedding，其角色不同于VecSet query。[PointFlow][teacherpoint]

- 当前：FPS选点，将这些点的XYZ Fourier与法向特征一起投影，用投影后的特征作query。
- 当前接口限制：1024个query要求至少1024个有效输入索引；512点输入会被拒绝，未覆盖论文Table5的512点情形。[FPS检查][localfps]、[p7 Table5][paper7]

| 方案 | query 与条件数据的关系 | 实现依据与适配范围 |
|---|---|---|
| 1 FPS 坐标 query | 所选点仅用XYZ embedding作Q；完整点特征作K／V | **适配**：[原VecSet FPS＋PointEmbed][V-encode]。原模型没有法向，保留NEXUS法向并只送K／V是适配。 |
| 2 FPS 点特征 query | 所选XYZ与法向等特征共同作Q；与全点集共享投影 | **原实现**：[Hunyuan采样与编码][H21-sample]、[TripoSG encode][T-encode]。当前属于这一类。 |
| 3 可学习 query | 用固定数量可学习向量读取整个点云 | **原实现**：[Michelangelo CrossAttentionEncoder][M-encoder]、[实际配置][M-config]。在NEXUS中设1024个query、宽2048属于适配。 |

- Michelangelo对齐版本还有一个全局query；保留该机制时必须明确是否包含在1024个输出token中。[全局query构造][M-global]

### 1.6 点坐标与法向的特征编码

- **原论文边界**：§4.1明确XYZ点和法向输入；未给Fourier频率、编码维数和法向拼接位置。[p6左栏L057–059][paper6]
- **老师候选**：不读取点云法向。带噪XYZ以6组π频率编码，加原XYZ得39维，再投影至144维。[fourier_positions][teacherposition]、[denoise][teachercond]

- 当前：8组`π×2^k`正余弦、原XYZ、原法向，共54维，线性投影到2048。

| 方案 | 频率与输入 | 实现依据与适配范围 |
|---|---|---|
| 1 含 π 的 Fourier | 8组`π×2^k`，拼接原XYZ与原法向 | **适配**：[原VecSet PointEmbed][vecembed]确证π频率；加入法向遵循NEXUS输入要求。 |
| 2 不含 π 的 Fourier | 8组`2^k`，拼接原XYZ与原法向 | **原实现依据**：[Michelangelo配置][M-config]、[TripoSG配置][T-config]及[编码][T-encode]。 |

- 两者输入维数相同，最高角频率分别为128π与128。频率改变不等于法向输入改变。
- 法向放置有两种实际路线：FPS点特征query中同时含法向；learned query本身不含法向，由数据K／V提供法向。来源分别见第1.5节方案2／3。
- **未列为可直接采用的方案**：法向也做Fourier编码。所核Hunyuan／Dora相关分支没有发布启用依据，并存在接口或维度接线问题。

### 1.7 VecSet 中的 QK 归一化

- **原论文边界**：§4.1没有条件编码器QK norm配置。[p6左栏L055–059][paper6]
- **老师候选**：点模型没有VecSet。输入是2048维缓存文本向量，经LN和MLP投影到144维。[PointFlow][teacherpoint]

- 当前：VecSet不加QK norm。[当前条件编码器][localencoder]


| 方案 | 结构 | 实现依据与适配范围 |
|---|---|---|
| 1 保留 Pre-LN 且不加 QK norm | 仅在attention／FFN之前做LN | **原实现**：[原VecSet][V-preln]、[Michelangelo][M-block]、[TripoSG][T-encoder]；当前采用。 |
| 2 增加逐头 QK LayerNorm | Q、K各自在head维度做LayerNorm | **原实现**：[Hunyuan点encoder][H21-qknorm]，由[发布配置qk_norm=true][H21-config]启用。 |
| 3 增加逐头 QK RMSNorm | Q、K各自做RMS类归一化 | **跨角色适配**：[Hunyuan发布配置][hycfg]启用RMS，由[norm选择][hynormchoice]传到[QK norm][hyqknorm]；所核Hunyuan点encoder使用的是LayerNorm。 |

### 1.8 DiT block 组织与 CA 分支

- **原论文边界**：§3.1.1给DiT与CA；未画block内部的SA／CA／FFN顺序、CA前归一化及长跳连。[p4右栏L008–016][paper4]
- **老师候选**：SA → FFN；两条分支受AdaLN调制，没有CA与U型长跳连。[DiTBlock][teacherblock]

- 当前：**SA → CA → FFN**；CA前有独立LayerNorm。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 平铺且 CA 前归一化 | SA → LN＋CA → FFN | **原实现**：[TRELLIS.2 block][t2block]。当前代码与此类结构接近。 |
| 2 平铺且 CA 直接读取残差 | SA → CA → FFN；CA直接接收SA残差结果 | **原实现**：[PixArtBlock][pixblock]，没有额外的CA前LN。 |
| 3 U 型长跳连 | block内保持SA → CA → FFN；后半层融合前半层特征 | **原实现**：[Hunyuan block][hyblock]、[skip存取][hyskip]。需保持父格token顺序对齐。 |

- **排列证据范围**：以上兼容CA的实现均采用SA → CA → FFN。本轮未找到这些作者实现中支持CA → SA → FFN等其他排列的依据。

### 1.9 时间调制的参数共享

- **原论文边界**：§4.1引用DiT；未明确NEXUS使用哪一种时间调制、是否逐block独立。[p6左栏L051–054][paper6]
- **老师候选**：每个block独立6H调制；时间MLP输出加全局文本向量后传入所有blocks。[DiTBlock][teacherblock]、[denoise][teachercond]

- 当前：每个block独立生成6组调制向量，对SA和FFN分别产生shift、scale、gate。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 逐块独立 AdaLN | 每层一个`SiLU → Linear(H,6H)` | **原实现**：[TRELLIS 1默认配置][t1defaults]、[对应block][t1block]。当前采用这一大类。 |
| 2 AdaLN single | 全网络共享一次6H投影；各block加自己的6H可学习偏置 | **原实现**：[TRELLIS.2][t2mod]、[PixArt][pixtime]。attention和FFN参数仍逐块独立。 |
| 3 时间 token | 时间嵌入作为额外token参与SA，最后移除 | **原实现**：[TripoSG forward][tripotime]、[Hunyuan forward][hytime]。移植时需单独处理该token的mask与RoPE。 |

### 1.10 全局条件是否参与调制

- **原论文边界**：§3.1.1明确条件经CA进入网络；未说明是否另有全局条件调制支路。[p4右栏L013–016][paper4]
- **老师候选**：文本经LN → Linear → SiLU → Linear，既加到点token，又与时间embedding相加进入AdaLN；没有CA。[denoise][teachercond]

- 当前：VecSet tokens进入CA；AdaLN只接收时间特征。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 条件仅走 CA | 时间控制AdaLN；完整条件tokens供CA读取 | **原实现**：[TRELLIS.2 block][t2block]。保持当前条件入口。 |
| 2 CA 加全局条件 AdaLN | VecSet摘要经投影后与时间嵌入相加，送入各层AdaLN；保留CA | **适配**：[原DiT时间＋类别调制][ditcond]、[老师候选全局文本＋时间调制][teachercond]。从VecSet得到摘要的方式需要另选。 |
| 3 CA 加全局条件时间 token | 对条件进行attention pooling，与时间特征融合成时间token；CA仍读取全部tokens | **源码选项**：[Hunyuan pooling构造][hypool]、[forward][hytime]；[发布配置][hycfg]实际关闭此选项。迁移到VecSet需适配输入宽度。 |

- 方案2中的摘要可采用有效token均值，或单独学习的全局query。均值池化＋共享线性投影这一完整组合属于本地适配；结合已有F／G对照判断效果，本文不新增重复实验。
- **借鉴优先级：兼容的结构候选。** 保留CA、depth与8维velocity接口后比较全局调制。老师候选证明该通路存在；CAD50整体成功尚不能独立证明AdaLN改动有效，也不能证明VecSet均值池化有效。[条件注入][teachercond]、[成功路径说明][teacher-original-prior]

### 1.11 CA 残差的门控

- **原论文边界**：§3.1.1明确CA；未规定其残差是否有动态gate。[p4右栏L013–016][paper4]
- **老师候选**：没有CA；SA与FFN分别有条件gate。老师代码不能决定NEXUS的CA gate。[DiTBlock][teacherblock]

- 当前：CA采用普通残差；SA和FFN具有时间gate。[当前block][localblock]

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 无 CA gate | `x ← x + CA(LN(x),condition)` | **原实现**：[TRELLIS.2][t2block]。当前方式。 |
| 2 增加 CA 时间 gate | 调制输出由6H增至7H；新增H维向量乘CA输出 | **原实现**：[Meta MeshFlow 7H构造][metagateinit]、[forward][metagate]。该额外gate没有同时增加CA的shift／scale。 |

### 1.12 QK 归一化

- **原论文边界**：§3.1.1及§4.1没有QK norm的启用位置、公式和参数共享规则。[p4右栏][paper4]、[p6左栏Vertex Diffusion段][paper6]
- **老师候选**：候选block没有QK norm，仅在attention和FFN前做LayerNorm。[DiTBlock][teacherblock]

- 当前：DiT的SA与CA都做QK RMS类归一化；VecSet不做。

| 方案 | 作用位置 | 实现依据与适配范围 |
|---|---|---|
| 1 全部关闭 | Q／K投影后直接进入attention | **原实现**：[PixArt SA][pixsa]、[CA][pixca]。主干LayerNorm仍保留。 |
| 2 仅 SA 开启 | SA开启QK RMS；CA关闭 | **原实现**：[TRELLIS 1配置][t1cfg]和[cross默认值][t1defaults]。 |
| 3 SA 与 CA 均开启 | 两处都做QK RMS类归一化 | **原实现**：[TRELLIS.2配置][t2cfg]、[Hunyuan发布配置][hycfg]。 |
| 4 QK LayerNorm | Q／K按每头特征维做LayerNorm | **源码选项**：[Hunyuan norm选择][hynormchoice]；点云编码器中的实际用例见“VecSet 中的 QK 归一化”。 |

- RMS实现细分：

| 选择 | 公式和参数 | 依据 |
|---|---|---|
| 每头独立gain | gain为`[heads,head_dim]`；TRELLIS使用`normalize(q)×√head_dim` | [TRELLIS.2 MultiHeadRMSNorm][t2qknorm] |
| 各头共享gain | `nn.RMSNorm(head_dim, eps=1e-6)`；gain为`[head_dim]` | [Hunyuan QK norm][hyqknorm] |

- 当前采用每头独立gain，但分母为`√(mean(q²)+1e-6)`。它与TRELLIS的epsilon规则有差别，属于同类机制的本地数值实现。

### 1.13 QKV 线性层的 bias

- **原论文边界**：§3.1.1及§4.1没有QKV bias配置。[p4右栏][paper4]、[p6左栏Vertex Diffusion段][paper6]
- **老师候选**：QKV与attention输出投影均带bias；对应bias张量有[权重记录][teachershape]。[DiTBlock][teacherblock]

- 当前：QKV投影带bias。

| 方案 | 结构 | 实现依据 |
|---|---|---|
| 1 保留 bias | Q、K、V投影包含可学习偏置 | **原实现**：[DiT配置调用][ditblock]。 |
| 2 去掉 bias | Q、K、V投影不含偏置 | **原实现**：[Hunyuan发布配置][hycfg]、[TripoSG实际构造][tripoconstruct]。输出投影是否有bias另行确定。 |

### 1.14 父格绝对位置如何进入网络

- **原论文边界**：图2标出位置embedding，正文指定3D RoPE；未明确额外加法位置支路及其编码方式。[p4图2与右栏L011–016][paper4]
- **老师候选**：当前带噪XYZ的Fourier投影进入token；另有slot embedding和文本加法；没有父格坐标。[denoise][teachercond]

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

### 1.15 三维 RoPE 的通道分配

- **原论文边界**：§3.1.1明确3D RoPE；未给三轴通道分配、剩余维度处理和频率底数。[p4右栏L011–012][paper4]
- **老师候选**：4头，每头36维，三轴各12维全部旋转，底数10000。head拆分和频率公式属于候选执行逻辑。[rope_3d][teacherrope]

- 当前：每头128维；每轴21对，共126维旋转，余下2维保持不变；base=10000。

| 方案 | 通道规则 | 实现依据与适配范围 |
|---|---|---|
| 1 允许剩余通道 | 每轴`floor(head_dim/6)`对，余下通道不旋转 | **原实现**：[TRELLIS.2 SparseRotaryPositionEmbedder][t2rope]。直接支持当前128维/head。 |
| 2 严格三轴均分 | 每头维度须被6整除，全部通道参与旋转 | **原实现**：[Meta RotaryPositionalEmbeddings][metarope]。当前128不满足；例如1536宽配16头得到96维/head，是可行的本地适配组合。 |
| 3 严格均分加连续坐标的 π 相位 | 三轴等分，使用`π×xyz×frequency` | **老师候选**：[rope_3d][teacherrope]。候选为36维/head；不能用其小模型尺寸代替NEXUS容量规格。 |

- 两种官方布局均有源码依据；老师一项补充连续坐标相位的本地候选依据。
- 原RoFormer给出旋转机制；XYZ轴分配仍需上述三维实现补充。

### 1.16 RoPE 的坐标单位

- **原论文边界**：§3.1.1明确3D RoPE；未规定位置取父格中心还是其他坐标，以及归一化／整数单位。[p4右栏L011–012][paper4]
- **老师候选**：对当前带噪XYZ使用π相位；没有八叉树整数坐标或固定参考深度。[rope_3d][teacherrope]、[denoise][teachercond]

- 记目标子层深度为`d`，父格分辨率`R_d=2^(d−1)`，父格整数坐标为`p`，归一化中心为`c=−1+2(p+0.5)/R_d`。
- 当前传入`r=256c`。在depth9时，`r=2p−255`；相邻父格的最高频相位差为2弧度。
- 老师RoPE接收当前带噪XYZ，数值不保证在[-1,1]；老师目标数据的归一化范围不等于所有带噪状态的范围。[采样入口][teachersample]

| 方案 | 送入 RoPE 的坐标 | 依据与取舍 |
|---|---|---|
| 1 当前层整数格单位 | `r=p` | **适配**：[TRELLIS.2直接使用integer coords][t2rope]。每层相邻格间隔为1；跨深度的物理距离尺度不同。 |
| 2 统一参考格单位 | `r=c×R_ref/2`；当前`R_ref=512` | **适配推导**：统一物理尺度对应固定参考网格。当前`×256`具有这一单位解释；未找到外部模型原样使用此NEXUS设定。 |
| 3 归一化连续坐标单位 | `r=πc` | **老师候选机制＋适配**：[连续XYZ RoPE][teacherrope]。原候选用当前带噪XYZ；NEXUS改为已知父格中心。 |

- `p`、`c`、`256c`产生不同频谱，必须成套记录坐标单位与RoPE频率。
- SA中同时旋转Q／K时，共同原点平移在理想内积中抵消；仅旋转CA的Q时，没有这一抵消关系。

### 1.17 时间特征编码

- **原论文边界**：§3.1.1给出flow模型，§4.1引用DiT；未给时间embedding的频率、宽度、倍率、MLP和激活。[p4右栏L008–011][paper4]、[p6左栏L051–054][paper6]
- **老师候选**：64维cos／sin、1000t、64→144→144、SiLU；层宽由张量支持，频率及倍率由候选forward给出。[time_embedding][teachertime]、[PointFlow][teacherpoint]

- 当前：256维cos／sin，输入`1000t`，base=10000；`Linear(256,H) → SiLU → Linear(H,H)`。

| 方案 | 编码与 MLP | 实现依据与适配范围 |
|---|---|---|
| 1 固定256维编码 | cos在前、sin在后；256→H→H，SiLU | **原实现**：[DiT TimestepEmbedder][dittime]、[TRELLIS.2编码][t2time]；TRELLIS的[调用处传1000t][t2timescale]。 |
| 2 hidden宽编码与归一化时间 | sin在前、cos在后；输入`t∈[0,1]`；H→4H→H，GELU | **原实现**：[Hunyuan编码器][hytimestep]、[实际4H构造][hytimebuild]、[时间入口][hytimescale]。结果用作time token。 |
| 3 hidden宽编码与1000倍时间 | sin在前、cos在后；内部使用1000t；H→4H→H，GELU | **源码构造**：[Meta编码器][metatime]、[实际4H构造][metatimebuild]。编码结果用于block调制。 |
| 4 较短的64维编码 | cos／sin各32维；1000t；64→H→H，SiLU | **老师候选**：[time_embedding][teachertime]。原H=144，扩大到当前H属于适配。 |

- 时间编码维度、倍率、MLP宽度、激活和注入位置是不同选择；整套移植时需一起注明。
- 上述候选保持当前时间方向和velocity含义；它们不规定训练时间分布。

### 1.18 FFN 的激活和宽度

- **原论文边界**：§3.1.1及§4.1没有FFN扩张倍率、门控形式和激活配置。[p4右栏][paper4]、[p6左栏Vertex Diffusion段][paper6]
- **老师候选**：144→576→144、GELU；张量支持4倍扩张，激活由候选代码给出。[DiTBlock][teacherblock]、[权重记录][teachershape]

- 当前：`H → 4H → H`，GELU的tanh近似；VecSet与DiT均采用这一形式。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 标准 GELU FFN | `Linear(H,4H) → GELU → Linear(4H,H)` | **原实现**：[DiT][ditblock]、[PixArt][pixblock]。 |
| 2 更宽的 GELU FFN | 中间层约`5.3334H`；H=1536时为8192 | **原实现**：[TRELLIS.2配置][t2cfg]、[FeedForwardNet][t2ffn]。 |
| 3 GEGLU | 输入投影分为两路4H；一路乘另一条的GELU门，再投回H | **原实现**：[3DShape2VecSet FeedForward][vecffn]。来源为AE模块；迁移到DiT属于适配。 |

- 参数关系：忽略bias，4H标准FFN约`8H²`；两路4H的GEGLU约`12H²`。两者不能按相同参数量解释。
- GELU精确形式也有两种先例：DiT／PixArt使用`tanh`近似；[Hunyuan dense MLP][hyffn]使用`nn.GELU()`默认形式。

### 1.19 主干归一化

- **原论文边界**：§3.1.1及§4.1没有主干Norm类型、affine与epsilon配置。[p4右栏][paper4]、[p6左栏Vertex Diffusion段][paper6]
- **老师候选**：SA／FFN前使用无affine LayerNorm，epsilon=1e-5，再施加条件shift／scale；最终输出LN带affine。[DiTBlock][teacherblock]、[输出头][teacherout]

- 当前DiT：Pre-LN；SA／FFN的LN不带静态affine，CA的LN带affine；`eps=1e-6`。当前VecSet各LN均带affine。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 Pre-LN 配合 AdaLN | SA／FFN使用无affine LN，由条件生成缩放与偏移 | **原实现**：[DiT][ditblock]、[TRELLIS.2][t2block]。 |
| 2 带 affine 的 Pre-LN | LN保留自身的可学习缩放与偏移；时间另走time token | **原实现**：[Hunyuan block][hynorm]、[TripoSG block][triponorm]。 |
| 3 Pre-RMSNorm | 主干归一化按RMS处理，不减均值 | **源码选项**：[Meta MeshFlow norm分支][metanorm]。本轮未取得发布配置来确认启用。 |

- epsilon候选：`1e-6`有DiT／Hunyuan依据；`1e-5`有[TripoSG实际构造][tripoconstruct]和[老师候选block][teacherblock]依据。
- 主干RMSNorm和“QK 归一化”中的QK RMSNorm是两个独立位置。

### 1.20 输出头

- **原论文边界**：§3.1.1明确8值占据token及velocity输出语义；未给最终Norm／调制／投影组成。[p4左栏L039–046及右栏L008–011][paper4]
- **老师候选**：带affine LN → Linear(144,3)形成去噪分支输出；完整PointFlow再加文本坐标prior与残差缩放。不能把此输出直接当成8维occupancy velocity。[PointFlow][teacherpoint]

- 当前：无affine LayerNorm后接`Linear(H,8)`。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 普通 LN 加线性投影 | 最后归一化，再逐token投影 | **原实现**：[TRELLIS.2输出][t2out]。当前属于这一类，epsilon需明确。 |
| 2 条件化 final AdaLN | 条件经独立2H投影生成最终shift／scale，再输出 | **原实现**：[DiT FinalLayer][ditout]。 |
| 3 共享风格 final 调制 | 可学习`[2,H]`表加时间向量，得到最终shift／scale | **原实现**：[PixArt T2IFinalLayer][pixout]。 |
| 4 带 affine 的 LN 加投影 | LN保留静态参数，随后输出 | **原实现**：[Hunyuan FinalLayer][hyout]、**老师候选**：[输出头][teacherout]。 |

- 所有候选在NEXUS中仍输出8维velocity。源模型的图像patch输出、3维XYZ和latent通道数均需替换。
- 仅采用time-token方案时才移除额外时间token；普通输出头不能删去首个父格。

### 1.21 Dropout 与残差随机丢弃

- **原论文边界**：§3.1.1及§4.1没有Dropout／DropPath模块配置。[p4右栏][paper4]、[p6左栏Vertex Diffusion段][paper6]
- **老师候选**：候选执行代码attention dropout=0，无FFN dropout／DropPath；原训练是否曾启用随机丢弃，没有张量证据。[FORWARD_CONTRACT][teachercontract]、[block][teacherblock]

- 当前：attention dropout=0；FFN dropout=0；没有DropPath。

| 方案 | 具体结构 | 实现依据与适配范围 |
|---|---|---|
| 1 不随机丢弃分支 | attention与FFN保持确定性的完整残差 | **原实现**：[DiT block][ditblock]、[PixArt默认block][pixblock]。 |
| 2 残差 DropPath | 按样本随机跳过SA／FFN残差；已有`p=0.1`实例 | **原实现**：[3DShape2VecSet decoder构造][vecdroppath]。用于条件encoder或DiT属于适配。 |
| 3 逐层增加 DropPath | 浅层概率低，深层概率高 | **源码选项**：[Meta逐层dpr][metadrop]。本轮未确认发布模型使用的最终概率。 |

- DropPath作用于残差分支；attention dropout作用于注意力概率，两者需分别命名。

### 1.22 占据向量的输入投影

- **原论文边界**：每8个子格占据值组成一个token；未给从8维到hidden的投影结构与bias。[p4右栏L008–011][paper4]
- **当前**：`Linear(8,1536,bias=True)`，逐父格投影。[构造][localdit]
- **老师候选**：带噪XYZ先编码为39维Fourier，再Linear(39,144)；不直接读取8维占据。[denoise][teachercond]

| 方案 | 具体结构 | 实现依据与适配要求 |
|---|---|---|
| 1 带 bias 的 Linear | `Linear(8,H,bias=True)` | [Hunyuan输入x_embedder][hy-input]、[实际配置][hycfg]。原角色为连续latent输入；改为8维占据属于适配。 |
| 2 不带 bias 的 Linear | `Linear(8,H,bias=False)` | [原VecSet latent denoiser的proj_in][vec-input]、[实际forward][vec-input-forward]。输入通道改为8；此bias与attention的QKV bias独立。 |

- 两种方案均保留一父格一token。其他投影深度尚无本轮核实的同类输入先例；不把未核实的MLP方案列为作者实际配置。

### 1.23 深度 embedding 的融合方式

- **原论文边界**：图2和正文明确可学习depth embedding在Transformer blocks之前；未给表宽、共享规则和融合公式。[p4图2、右栏L011–013][paper4]
- **当前**：`Embedding(D,H)`，D9为9×1536；按`d−1`查表，在第一个block前加到每个父格token一次。[forward][localforward]
- **老师候选**：没有octree depth embedding；文本投影分别加入token与时间调制。[denoise][teachercond]

| 方案 | 具体结构 | 实现依据与适配要求 |
|---|---|---|
| 1 一张表在入口注入 | 全网共享H维depth表，进入blocks前加到token一次 | [OctGPT learned depth表][oct-depth]提供H维深度向量与特征相加的构件。入口只加一次是当前本地组合。 |
| 2 每个block独立注入 | 每块自己的H维depth表，在该block前相加 | [OctGPT block构造与forward][oct-block]、[模型调用][oct-caller]、[构造默认][oct-default]；发布权重启用状态未确认。原实现同时重复加入空间编码，迁移时只取depth部分是适配。 |
| 3 入口加法 加 AdaLN | 保留输入侧depth embedding，另把其投影加入全局调制向量 | [DiT类别加时间调制][ditcond]、[老师双路注入][teachercond]提供机制。替换为depth语义属于跨角色适配。 |

- OctGPT以自己的八叉树序列与表示训练；借用depth机制不等于采用其生成范式。
- 上述方案均保留可学习depth输入。本表优先保留输入侧可学习depth入口；这是一项选型约束。原文未给融合公式，仍无法判定其他融合方式是否等同作者实现。

### 1.24 面数条件的编码与注入

- **原论文边界**：§4.5明确在第一阶段加入face-count条件控制细节程度；未给编码函数、接入位置或与基础点云模型的启用关系。[p10右栏L052–057][paper10]
- **老师候选**：有文本预测顶点数量的275类输出头；没有外部目标面数条件入口。预测点数与输入面数预算是两个不同接口。[count_logits][teacherpoint]

- 当前：R1-D9的forward没有face-count输入；这是论文应用扩展中尚未实现的一项。[当前forward][localforward]

| 方案 | 具体结构 | 实现依据与适配要求 |
|---|---|---|
| 1 面数分桶 embedding | 预算离散分桶，查表后加到全局调制向量 | [DiT类别embedding][ditlabel]、[与时间相加][ditcond]。分桶规则及face-count语义是本地适配。 |
| 2 连续标量 embedding | 面数经预定标度变换、正余弦／MLP编码，再加入调制向量 | [DiT标量编码MLP][dittime]提供构件；原用途是时间，转作面数条件属于跨角色适配。 |
| 3 独立控制 token | 将预算编码成一个额外token，与父格token进入SA | [Hunyuan额外time token][hytime]、[TripoSG time token][tripotime]提供注入机制；改为预算token后需定义mask和RoPE。 |

- 三项均为有源码构件依据的候选；NEXUS原文不足以判定作者采用其中哪项。本文只登记缺项与方案，不启动实现。

### 1.25 自注意力的空间范围

- **原论文边界**：Vertex段只规定DiT，未给全局或空间窗口attention的具体配置。[p4右栏L009–016][paper4]、[p6左栏L051–054][paper6]。拓扑AE的全局attention消融不能作为Vertex配置证据。
- **当前**：每个父格读取同一样本全部有效父格；没有空间窗口或causal mask。[attention][localattention]
- **老师候选**：所有有效点之间做全局SA，以key mask排除padding。[DiTBlock][teacherblock]

| 方案 | 具体结构 | 实现依据与适配要求 |
|---|---|---|
| 1 全局 SA | 同一样本所有有效父格相互注意 | [TRELLIS发布flow配置][t1flow-release]、[实际full构造][t1flow-full]。当前采用。 |
| 2 交替偏移空间窗口 SA | 按父格坐标分窗，相邻层偏移窗口 | [TRELLIS发布mesh AE decoder][t1swin-release]用swin、窗口边长8；[构造][t1swin-build]交替三轴偏移0／4。迁移到Vertex DiT是跨角色适配。 |
| 3 固定空间窗口 SA | 每层使用同一空间分窗，偏移为0 | **源码选项**：[windowed分派][t1window-dispatch]、[空间窗口实现][t1window-code]。未核实发布生成配置启用。直接的跨窗口SA通信被移除。 |

- 边长8是整数空间距离，不能解释成每窗8个token。NEXUS不同depth采用何种窗口尺度仍需补定。
- 三项只改变SA范围，CA可继续读取全部VecSet tokens。[TRELLIS全局CA限制][t1ca-full]
- TRELLIS的参数名`shift_window`在部分构造中指序列化attention；本文空间窗口方案对应`swin／windowed`实际路线。[映射代码][t1swin-build]

## 二 结构细节能够通过原论文确定

- 下表“已确定”只覆盖该行写出的内容；未展开的内部参数仍归第一部分。
- “当前一致”是冻结代码层面的结构对照，不代表本轮执行了模型生成。
- 老师列按最终点模型候选填写；未提供对应模块时直接标记。

### 2.1 原生表示与层级结构

| 结构细节 | NEXUS原论文内容及定位 | 我们当前实现 | 老师最终点模型候选 |
|---|---|---|---|
| 阶段分解 | 先生成顶点集合，再以顶点与外部条件生成拓扑。[p3右栏式(1)][paper3]、[p4图2][paper4] | VertexStageSystem直接处理占据；独立VAE／AE属于另一个阶段。[构造][localsystem] | 点模型先出连续XYZ，再有独立拓扑路径。[模型定义][teacherpoint] |
| 八叉树占据表示 | 每坐标D-bit；每层记录节点占据。[p4左栏§3.1 L012–019][paper4] | [octree编码][localoctree]；结构一致 | 连续XYZ，每个token是一个点。[PointFlow][teacherpoint] |
| 占据取值 | 占据为1、空为0；进入连续生成模型。[p4左栏L031–046及右栏L008][paper4] | `[parents,8]`实数0／1目标。[octree编码][localoctree] | 输出三维连续坐标，无8位子格占据。 |
| 单token内容 | 同一父格的8个子格占据共同组成一个token；原文“single token”。[p4左栏L039–046、右栏L010–011][paper4] | noisy `[B,N,8]`；结构一致。[forward][localforward] | 带噪XYZ经Fourier变为一个点token。[denoise][teachercond] |
| 序列长度来源 | token数由上一层占据父格数决定。[p4左栏L035–043][paper4] | 父格列表决定N；完整展开不读GT点数。[generate_cells][localgeneration] | 先由文本预测点数；最多274个slot。[sample_points][teachersample] |
| 输出语义 | flow matching采用velocity参数化，作用于子格占据向量。[p4右栏L008–011][paper4] | 每父格输出8维velocity。[VertexDiT][localdit] | PointFlow预测clean XYZ；由调用入口换算速度。[sample_points][teachersample] |
| 跨层参数共享 | 同一网络处理不同depth。[p4左栏L031–032、L044–045][paper4] | 一个VertexDiT接收depth输入。[VertexStageSystem][localsystem] | 不建八叉树，没有跨depth共享这一接口。 |
| 粗到细依赖 | 从一个占据根节点开始；后续层依赖上一层生成的占据与相同条件。[p4右栏§3.1.2 L018–027][paper4] | 同一网络从depth1逐层展开。[generate_cells][localgeneration] | 一次处理整组点；没有逐层占据展开。 |
| 顶点解码位置 | 最细层占据格的中心作为最终顶点坐标。[p4左栏L015–019][paper4] | `decode_leaf_centers`恢复格中心。[实现][localdecode] | 输出连续XYZ，不做叶格中心解码。 |
| 论文实验分辨率 | D=9，整数坐标范围[0,511]。[p6右栏L018–020][paper6] | 本次对照是D9；D15属独立扩展。[当前构造][localdit]、[数据检查][localprepare] | 连续坐标，无D9／D15。 |
| 顶点序列建模 | 论文消除逐顶点序列化；层内生成占据集合。[p3右栏§Comparison][paper3]、[p4左栏L008–009][paper4] | 无causal mask、无父格序号embedding；坐标随token对齐。[attention][localattention] | 有可学习slot embedding；与论文sort-free八叉树token机制不同。 |

### 2.2 主干与条件接口

| 结构细节 | NEXUS原论文内容及定位 | 我们当前实现 | 老师最终点模型候选 |
|---|---|---|---|
| 主干家族与规模 | Vertex阶段使用DiT，约2B参数；原文短语“approximately 2 billion parameters”。[p6左栏L051–054][paper6] | 约1.93B DiT；36×1536／12这一实现组合见第一部分。[构造][localdit] | 最终18层hidden144；完整点模型记录约8.67M参数，包含prior等模块。[权重结构][teachershape] |
| 三维位置编码 | 明确使用3D RoPE；图2标有位置embedding。[p4图2、右栏L011–012][paper4] | SA Q／K使用三轴RoPE；单位及分轴细节见第一部分。[forward][localforward] | 使用连续XYZ的三轴RoPE；具体执行为候选复建。[rope_3d][teacherrope] |
| 深度条件 | 可学习depth embedding，放在Transformer blocks之前；原文“learnable depth embedding”。[p4图2、右栏L011–013][paper4] | 可学习表加到token，位于block堆栈前。[forward][localforward] | 无depth embedding。 |
| 点云输入类型 | mesh表面点及对应法向。[p6左栏L057–059及右栏L015][paper6] | 每点XYZ＋normal，共6维。[条件载入][localconditionload] | 2048维缓存文本；没有点云／法向encoder。 |
| 可变输入点数能力 | 论文报告512～16384点输入的密度测试。[p7 Table5][paper7] | 当前FPS要求有效点数≥1024；8192输入受支持，512输入未覆盖。[检查][localfps] | 无点云输入接口。 |
| 论文训练输入点数 | 8192个表面采样点。[p6左栏L057–059、L062及右栏L015][paper6] | 固定条件形状8192×6。[载入断言][localconditionload] | 不适用。 |
| 条件编码器家族 | 联合训练的VecSet点云encoder。[p4右栏L013–016][paper4] | VertexConditionEncoder与DiT均可训练。[构造][localsystem]、[实际入口][localjoint] | LN＋MLP处理缓存文本；没有对应VecSet。 |
| VecSet公开层宽规格 | 8层、hidden2048；原文“8 layers with a hidden dimension of 2048”。[p6左栏L055–056][paper6] | 1CA＋7SA、2048宽；宽度一致，8层计数解释见第一部分。[encoder][localencoder] | 不适用。 |
| 条件输出形状 | 1024个token，每个2048维。[p6左栏L058–059][paper6] | `[B,1024,2048]`。[encoder][localencoder] | 一个文本向量映射到144维；无1024-token接口。 |
| 条件接入机制 | 条件经cross-attention进入去噪网络。[p4右栏L013–016][paper4] | 每个block有CA。[block][localblock] | 无CA，采用全局文本加法与AdaLN。 |
| 面数控制能力 | 第一阶段加入目标面数条件，实现不同LOD；示例预算500～20000面。[p10右栏§4.5 L052–057][paper10] | 当前无面数条件；注入候选见第一部分。此项是论文应用扩展。 | 只有预测顶点数输出头，无外部面数预算输入。 |

- 图像分支的预训练DINOv3同样由原文明确给出，见[p4右栏L015][paper4]及[p6左栏L060–061][paper6]；本次选型范围为点云分支。
- 论文明确的是约2B的**Vertex DiT**；同页另有Topology Diffusion约2B描述，两段分别存在。“相同架构”不表示两阶段共用一套权重。[p6两阶段实现段][paper6]
- 8192是论文采用的训练输入点数；论文另有不同输入密度测试，不能据此把8192说成唯一可接受长度。[p7 Table5及§4.2.3][paper7]

### 2.3 已确定内容的边界

| 已确定内容 | 仍未由该证据确定的细节 |
|---|---|
| D9与整数[0,511] | 训练数据如何居中／缩放、round或floor、边界clamp细则。当前floor等规则是本地实现，归属编码约定，不扩展为结构选型菜单。 |
| 叶格中心解码 | 固定坐标变换的精确实现与量化碰撞处理。 |
| 论文评估时将mesh放入[-1,1]包围盒 | 该评估说明不能直接确定训练归一化映射或RoPE坐标单位。[p7左栏§4.2 L021–026][paper7] |
| 连续占据读出 | 阈值和空树处置没有具体规定；属于表示读出约定，本轮不展开采样／修补策略。 |
| 8值token与sort-free | 子格bit编号、内存排序、padding格式、mask布尔含义；这些实现约定不要求多个网络结构候选。 |
| 位置embedding与3D RoPE | 是否额外使用父格Fourier加法支路、RoPE作用于哪些attention、相位与分轴。 |
| Learnable depth embedding | 表宽、加法或其他融合公式，以及是否额外进入调制支路。 |
| 8层VecSet | CA／SA计数与排列、query构造、heads、内部FFN和Norm。 |
| 约2B DiT | blocks、hidden、heads、FFN、调制共享、输出头的具体组合。 |
| 条件通过CA注入 | CA前Norm、残差gate、每层是否都开CA，以及额外全局调制。 |
| 面数控制应用 | 条件编码、注入位置、是否在论文全部基础模型中始终开启。 |

### 2.4 老师成功证据与借鉴边界

- 本节是外部成功对照，不能补成NEXUS原论文的结构规定；不改变前述两类归属。

#### CAD50点生成成功证据

| 证据 | 点数正确 | 坐标RMSE | 成功范围 |
|---|---|---|---|
| 老师原报告的两个噪声种子 | 两组均50／50 | 6.08063513×10⁻⁸、6.05511849×10⁻⁸ | 每个样本通过点数与最小间距误差检查。[原报告][teacher-original-result] |
| 候选重放 seed34567 | 50／50 | 1.10722148×10⁻⁷ | 最大顶点误差／GT最小间距的最大值为4.15365×10⁻⁴。[记录][teacher-replay-a] |
| 候选重放 seed98765 | 50／50 | 1.10411672×10⁻⁷ | 上述比值最大值为4.17950×10⁻⁴。[记录][teacher-replay-b] |

- 目标为老师预处理后的CAD50缓存，坐标空间为包围盒最长边2；原始CAD与缓存的顶点数可能不同。[目标定义][teacher-target]
- 候选重放先做评价用Hungarian一对一匹配，再算全局坐标RMSE：`sqrt(Σ_i Σ_j ‖pred_ij − gt_iπ(j)‖² / (3Σ_i N_i))`。这是逐坐标RMSE；不对50个样本RMSE直接取平均。匹配不修改预测。[评价函数][teacher-metrics]
- 上表来自已有原报告与实际CPU候选重放记录。本次只读复核，不属于新模型重放或从头训练；点阶段成绩也不代表50个完整mesh拓扑全对。

#### 成功模型的完整点生成路径

```text
缓存文本特征 c ∈ R^2048
  ├─ text MLP → count head → 预测点数 N
  ├─ coordinate prior → P(c) ∈ R^(274×3)，取前 N 个槽位
  └─ text MLP → 点token加法 与 全局AdaLN条件

高斯XYZ噪声 x0 ∈ R^(N×3)
  → 18层去噪分支 D(x,t,c)
  → clean XYZ预测 x1_hat = P(c)[:N] + α D(x,t,c)
  → v = (x1_hat − x) / (1 − t)，100步Euler
  → 生成XYZ，点数来自count head
```

- `α=1.21982403×10⁻⁵`。原报告明确：最终阶段冻结原去噪器，学习文本坐标prior，结果由prior主导；属于50条训练提示的记忆，未证明未见文本泛化。[原报告][teacher-original-prior]
- 实际候选输入为缓存文本与噪声；不读取GT点数、GT坐标、GT面或UID选点。prior是学到的参数映射。[PointFlow][teacherpoint]、[sample_points][teachersample]
- 成功证据支持这套完整组合。尚无独立证据证明去掉prior后仍达标，或某个局部结构是成功的必要原因。

#### 向NEXUS八叉树借鉴的范围与顺序

| 老师机制或结构 | 对八叉树接口的影响 | 本文判断与下一步依据 |
|---|---|---|
| 条件直接恢复clean几何 | 可在原占据接口内建立能力诊断：点云＋父格＋depth→8个clean占据值 | **优先分析。** 复用E2-C：固定t=0、噪声=0，原velocity目标即y；不增加prior或改输出头。通过只说明该受控子问题可学。[执行卡][e2-card] |
| 全局条件加入时间AdaLN | 可保留CA、depth和8维velocity；需将VecSet tokens聚合并投影 | **兼容的结构候选。** 优先解释已有F／G对照；池化方式和新增投影属于本地适配。老师整网成功不提供该通路的独立收益结论。见第1.10节。[老师注入][teachercond] |
| 全局条件加到输入token | 可将VecSet摘要投影后广播到父格token，同时保留原CA | **后续兼容候选。** 老师代码有对应加法；需要独立于AdaLN变化比较，避免混合两条新通路。[老师注入][teachercond] |
| 层宽、head数、FFN、Norm、RoPE、输出头等局部配置 | 多数可在相同输入输出语义下适配；仍受约2B规模与3D位置接口约束 | **保留为结构备选。** 各节源码证明其实现存在；老师prior主导的成绩不足以提高某一局部配置的效果置信度。 |
| 文本→完整XYZ坐标prior | 直接移植会增加绕过逐层占据生成的坐标输出路径 | **不纳入本轮八叉树复现选择。** 可保留为独立老师基线；改成occupancy prior需另立扩展假设与实验。[完整输出][teacher-full-output] |
| 点数分类头与固定XYZ槽位 | 会替换“上一层生成父格决定长度、叶格占据决定点数”的机制 | **不直接移植。** 老师最多274个点槽位不能充当父格索引embedding。见第2.1节。[count与slot][teacherpoint] |
| 直接clean XYZ输出及X1速度换算 | 改变原8维occupancy velocity输出的语义 | **不随结构借鉴一起改。** 参数化属于独立对照；当前保留NEXUS八叉树velocity接口。[老师采样][teachersample] |

- 上表顺序是研究判断，不是已验证的性能排名。E2-C属于能力诊断；结构变更须由对应证据支持。本文不重新安排A–G或追加训练。
- 成功证据的层次：**完整组合能运行并拟合**；单独机制的贡献仍需受控对照；对NEXUS有效还需本任务的占据、完整树和整数顶点集合验收。两侧条件、对象与表示不同，不直接比较上述RMSE来排名。

#### 候选复建与恢复边界

- 张量记录支持：18层、hidden144、4H FFN、6H调制、274点槽位、文本2048维及输出3维等尺寸。[权重结构][teachershape]
- 候选执行逻辑：4头拆分、RoPE频率与相位、时间倍率、归一化epsilon、激活和条件融合顺序。[models.py][teacherpoint]
- 无对应模块：VecSet、点云法向输入、八叉树占据、depth embedding、逐层展开、叶格中心解码。
- mask边界：老师候选只遮蔽attention key，没有清零无效query／output；实际逐对象采样按预测N建张量，不传padding mask。[block][teacherblock]、[采样][teachersample]
- 已有权重重放不等于恢复原训练流程：原文本编码器、完整训练器和原RNG缺失；candidate Adam仅覆盖prior六项与residual_scale一项，不能当作完整去噪器续训状态。[恢复边界][teacher-recovery]

### 2.5 组合约束

- 保留每父格8值token、8维velocity、共享多depth网络、3D RoPE、可学习depth embedding及点云CA接口。
- 层数、宽度、heads、FFN与调制共享共同决定参数量；借用某一层宽配置后重新核算规模。
- RoPE通道分配与head宽匹配，坐标单位与频率成套定义。
- time token、AdaLN、输出调制与控制token需组合核验；额外token的mask与位置不能沿用父格索引。
- 点云AE源码提供聚合模块先例；迁移到NEXUS条件encoder后，保留条件角色，不据此增加VAE潜变量。

### 2.6 论文与官方源码索引

| 来源 | 论文 | 本文采用的实现依据 |
|---|---|---|
| NEXUS | [Native Mesh Generation with Diffusion](https://arxiv.org/abs/2607.13563v1) | 任务接口与公开规模规格 |
| DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | AdaLN、时间编码、输出头、层宽 |
| 3DShape2VecSet | [3D Shape Representation for Neural Fields and Generative Diffusion Models](https://arxiv.org/abs/2301.11445) | FPS查询、Fourier、GEGLU、点云聚合 |
| Michelangelo | [Conditional 3D Shape Generation based on Shape Image Text Aligned Latent Representation](https://arxiv.org/abs/2306.17115) | learned queries、点特征编码、全局query |
| Hunyuan3D 2.1 | [From Images to High Fidelity 3D Assets with Production Ready PBR Material](https://arxiv.org/abs/2506.15442) | 点编码器、time token、QK norm、U型skip |
| TRELLIS | [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506) | 独立调制、CA前LN、APE |
| TRELLIS.2 | [Native and Compact Structured Latents for 3D Generation](https://arxiv.org/abs/2512.14692) | 共享调制、RoPE、输出头 |
| PixArt α | [Fast Training of Diffusion Transformer for Photorealistic Text to Image Synthesis](https://arxiv.org/abs/2310.00426) | AdaLN single、CA直接残差、输出调制 |
| TripoSG | [High Fidelity 3D Shape Synthesis using Large Scale Rectified Flow Models](https://arxiv.org/abs/2502.06608) | 点编码器、time token、norm与bias实际调用 |
| Meta MeshFlow | [Efficient Artistic Mesh Generation via MeshVAE and Flow based Diffusion Transformer](https://arxiv.org/abs/2606.04621) | CA gate、RoPE、时间编码及明确标注的源码选项 |
| Dora | [Sampling and Benchmarking for 3D Shape Variational Auto Encoders](https://arxiv.org/abs/2412.17808) | uniform／salient双CA聚合 |
| OctGPT | [Octree based Autoregressive Models for 3D Shape Generation][oct-paper] | learned depth table与逐块注入，发布启用状态按正文标明 |
| 老师候选 | [本地models.py](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:88) | 局部结构为候选级依据；完整prior路径另有CAD50点生成成功记录，见第2.4节 |

### 2.7 当前代码与核查版本

- [VertexConditionEncoder][localencoder]：点云编码、query与条件token。
- [_VertexBlock][localblock]：SA／CA／FFN、Norm与调制。
- [VertexDiT.forward][localforward]：占据、父格位置、depth与time融合。
- [octree.py][localoctree]：8子格目标和树展开；[decode_leaf_centers][localdecode]：格中心解码。
- [冻结代码身份记录][codeidentity]：`vertex.py` SHA256为`39491f94fadec84797b59a92d969dd923353abeea29a91a02e1a0f904ae1f386`。
- 核查版本：NEXUS arXiv 2607.13563v1；GitHub代码链接均固定到commit；老师使用最终结构记录和对应候选源码。

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
[codeidentity]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/startup_evidence/e2/C/config.json
[ditblock]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L101-L122
[ditcfg]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L328-L344
[ditcond]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L233-L246
[ditlabel]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L67-L95
[ditout]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L125-L142
[dittime]: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L27-L64
[hy-input]: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/82920d643c0dc2f7bfd7255f45f62d386edfe60c/hy3dshape/hy3dshape/models/denoisers/hunyuandit.py#L578-L585
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
[localattention]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:136
[localblock]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:261
[localconditionload]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/train_ab.py:165
[localdecode]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/octree.py:41
[localdit]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:296
[localencoder]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:219
[localforward]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:338
[localfps]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex.py:35
[localgeneration]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/vertex_evaluation.py:135
[localjoint]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/train_e2.py:86
[localoctree]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/octree.py:65
[localprepare]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/prepare_data.py:45
[localsystem]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/source_runtime/d15_code/mini_nexus/training.py:15
[metacarope]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L351-L364
[metadrop]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L737-L755
[metagate]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L578-L607
[metagateinit]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L538-L539
[metanorm]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L714-L716
[metarope]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L92-L197
[metaropedefault]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L859-L873
[metatime]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L200-L270
[metatimebuild]: https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L725-L730
[oct-block]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octformer.py#L274-L304
[oct-caller]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octgpt.py#L97-L113
[oct-default]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/octgpt.py#L24-L34
[oct-depth]: https://github.com/octree-nn/octgpt/blob/9eb824296bb6888219f34eb78fe81d9d1c5912d4/models/positional_embedding.py#L205-L269
[oct-paper]: https://arxiv.org/abs/2504.09975
[paper10]: https://arxiv.org/pdf/2607.13563v1#page=10
[paper3]: https://arxiv.org/pdf/2607.13563v1#page=3
[paper4]: https://arxiv.org/pdf/2607.13563v1#page=4
[paper6]: https://arxiv.org/pdf/2607.13563v1#page=6
[paper7]: https://arxiv.org/pdf/2607.13563v1#page=7
[paperlines]: /Users/luthier/Documents/sophomore/nexus_fast_track/tmp/pdfs/sources/nexus_arxiv_2607.13563v1_paged_nonempty_lines.txt
[pixblock]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L25-L54
[pixca]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L29-L57
[pixout]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L172-L188
[pixsa]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt_blocks.py#L109-L125
[pixtime]: https://github.com/PixArt-alpha/PixArt-alpha/blob/cac2fd3b4544cf7620d8fbbf8b19d97ffcb85892/diffusion/model/nets/PixArt.py#L121-L135
[t1ape]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/blocks.py#L8-L46
[t1block]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/transformer/modulated.py#L99-L149
[t1ca-full]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/sparse/attention/modules.py#L34-L49
[t1cfg]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/ss_flow_img_dit_L_16l8_fp16.json#L4-L17
[t1defaults]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/sparse_structure_flow.py#L68-L73
[t1flow-full]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/structured_latent_flow.py#L149-L163
[t1flow-release]: https://huggingface.co/microsoft/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_flow_img_dit_L_64l8p2_fp16.json#L1-L17
[t1slatcfg]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json#L4-L19
[t1swin-build]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/models/structured_latent_vae/base.py#L14-L24
[t1swin-release]: https://huggingface.co/microsoft/TRELLIS-image-large/blob/25e0d31ffbebe4b5a97464dd851910efc3002d96/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.json#L1-L12
[t1window-code]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/sparse/attention/windowed_attn.py#L20-L66
[t1window-dispatch]: https://github.com/microsoft/TRELLIS/blob/442aa1e1afb9014e80681d3bf604e8d728a86ee7/trellis/modules/sparse/attention/modules.py#L122-L125
[t2block]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L104-L157
[t2cfg]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json#L4-L18
[t2ffn]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/blocks.py#L49-L59
[t2mod]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/transformer/modulated.py#L132-L144
[t2out]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L193-L199
[t2posforward]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/structured_latent_flow.py#L182-L194
[t2qknorm]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/attention/modules.py#L9-L16
[t2rope]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/modules/sparse/attention/rope.py#L7-L58
[t2time]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/models/sparse_structure_flow.py#L12-L53
[t2timescale]: https://github.com/microsoft/TRELLIS.2/blob/75fbf0183001ed9876c8dbb35de6b68552ee08bd/trellis2/trainers/flow_matching/sparse_flow_matching.py#L95-L104
[teacherblock]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:44
[teachercond]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:114
[teachercontract]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:16
[teacherout]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:104
[teacherpoint]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:88
[teacher-original-result]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/source_notes/teacher_cascade_REPORT.md:3
[teacher-original-prior]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/source_notes/teacher_cascade_REPORT.md:8
[teacher-target]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/source_notes/teacher_delivery_README.md:19
[teacher-replay-a]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/point_final/RESULT.json
[teacher-replay-b]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/point_final_seed98765/RESULT.json
[teacher-metrics]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/replay.py:25
[teacher-full-output]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:126
[teacher-recovery]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/REPORT.md:110
[e2-card]: /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/EXECUTION_CARD.md:11
[teacherposition]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/teacher_ae.py:21
[teacherrope]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:30
[teachersample]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:157
[teachershape]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json
[teachertime]: /Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:24
[tripoconstruct]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L440-L462
[triponorm]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L173-L202
[tripotime]: https://github.com/VAST-AI-Research/TripoSG/blob/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c/triposg/models/transformers/triposg_transformer.py#L664-L718
[vec-input]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L179-L200
[vec-input-forward]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_class_cond.py#L211-L225
[vecdroppath]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L204-L225
[vecembed]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L109-L139
[vecffn]: https://github.com/1zb/3DShape2VecSet/blob/8df9b7a55c42d4dcad152294755250a2ab1e34e5/models_ae.py#L51-L68
