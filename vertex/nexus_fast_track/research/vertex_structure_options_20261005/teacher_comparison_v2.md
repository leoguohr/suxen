# 最终老师点候选：逐项结构对照 v2

核对日期：2026-10-05。对照对象仅为 `/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py` 的 **PointFlow 最终候选**，不是其 TopologyFlow，不是早期6层版本，不是 NEXUS 官方作者源码。仅重读源码、JSON结构审计和既有记录；未训练、推理、连接服务器或修改主报告。

## 证据标记与最终配置

- **S：张量形状/已有检查点结构记录**。能确认维度、参数是否存在、独立 block 编号等；不能唯一确定 activation、head 数、RoPE/时间公式、forward 排列、初始化或训练 dropout。本轮读取的是已有 `checkpoint_structure.json`，没有重新打开/hash 原权重。
- **C：当前候选源码执行逻辑**。本轮确认当前代码这样执行；这些逻辑曾做有限前向和CPU采样核对，但不等于恢复老师原始源码。
- **N：没有对应模块/接口**。当前 PointFlow 中没有该模块；不能将相似名字或另一阶段模块强映射过来。
- **U：原训练设置未恢复**。当前推理代码或训练目标构件不证明老师原始训练器采用同样设置。

最终配置来自[checkpoint config](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:736)：`text_dim=2048,width=144,layers=18,max_points=274,prior_width=512`。结构记录的 block 编号完整为0…17，206个state条目、8,668,301个元素。[最后一个block](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:667)。[load_points](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:149)严格读取 checkpoint config 后加载 state_dict；`teacher_point_INITIAL_config.json` 或 `teacher_early_config.json` 中的早期6层不覆盖它。

最终输出不是单一 denoiser：`x1_hat=coordinate_prior(e)[:,:N]+α·denoise(x_t,t,e)`。既有记录α=`1.2198240256111603e-05`；老师交付说明承认主要精度来自文本先验对50条训练集的记忆。[实际组合](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:126)、[结果边界](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/README.md:66)。因此以下局部机制可以比较，整体成绩不能作为原去噪分支独立有效的证明。

## 第一部分：主报告原第1–19项的老师对照

下面保留主报告的原编号，方便合并；是否由 NEXUS 论文确定应由主报告的论文证据列判断。老师对照只说明老师候选自己的做法。

| 原编号 / 项目 | 最终老师点候选实际做法 | 证据类型与边界 | 当前代码/记录定位 |
|---|---|---|---|
| 1 DiT层数、宽度、heads | 18个独立DiTBlock，hidden144；候选设4 heads，每头36 | **S** config和block0…17/qkv432×144支持层数与宽度。**C** `self.heads=4`；张量形状不唯一确定4 heads。不能用该小模型替代NEXUS约2B容量规格 | [PointFlow构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:95)、[heads与reshape](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:51)、[最终config](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:736) |
| 2 VecSet层数组合/聚合 | 没有VecSet或点云条件编码器。输入直接是缓存2048维文本全局向量，由text LN/MLP投影到144 | **N** 无“1 CA+7 SA”或任何VecSet层。**S** text映射2048→144→144；**C** 只有全局向量投影，不能当作8层VecSet对照 | [text构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98)、[缓存输入读取](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/replay.py:67) |
| 3 VecSet query构造 | 没有FPS、随机采样点query、可学习CA query或全局VecSet query。`slot.weight[:N]`只是按点序号加到生成hidden的identity embedding | **N** 无VecSet query。**S** slot274×144；**C** 加slot前缀，不做点云聚合。slot不能改名为FPS query或全局query | [slot参数](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:100)、[slot使用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:119)、[slot shape](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:47) |
| 4 点XYZ/normal特征编码 | 对**当前带噪XYZ**做6 bands `π2^k` Fourier，拼raw XYZ+sin+cos为39维，Linear39→144；不输入normal | **S** input144×39确认接口维度。**C** 6 bands、π和排列来自`fourier_positions`，39维形状不唯一证明此公式。**N** 无法向通路，也不是条件点云编码 | [Fourier函数](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/teacher_ae.py:21)、[输入Linear](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:101)、[调用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:119) |
| 5 DiT block组织/CA | pre-LN＋调制→self-attention＋time/text gate residual→pre-LN＋调制→GELU FFN＋gate residual。没有CA，18块顺序执行，无UNet式skip stack | **C** SA→FFN顺序/两个残差。**N** 无SA→CA→FFN的中间CA；text通过加法和AdaLN而非K/V注入 | [DiTBlock.forward](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:53)、[18块执行](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:122) |
| 6 时间调制参数共享 | 各block有各自`SiLU→Linear144,864`，输出shift1/scale1/gate1/shift2/scale2/gate2；所有block接收同一condition向量，但modulation参数独立 | **S** 每个blocks.i.ada.1独立参数条目。**C** condition共享、modulation不共享；不是AdaLN-single的共享大投影＋各层表 | [ada构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:50)、[六组拆分](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:56)、[condition传入各块](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:121) |
| 7 全局条件参与调制 | 缓存e2048→LN2048→Linear2048,144→SiLU→Linear144,144得z；z同时加到每个token hidden，并与time144相加作为AdaLN condition。count head也使用z | **S** text维度与参数。**C** 全局条件双路径、先后顺序。没有条件tokens上的pool函数；raw text masked-mean来源只是已有报告记录，原encoder未交付 | [text路径](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98)、[count使用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:108)、[双注入forward](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:118)、[原encoder边界](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/REPORT.md:51) |
| 8 CA残差gate | 没有CA，不存在CA gate。仅有SA gate1和FFN gate2，均由time+text condition生成 | **N** 不能写“老师CA关闭gate”或“老师CA ungated”：老师没有CA残差。**C** SA/FFN gate存在 | [block两个gate残差](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:56) |
| 9 QK归一化 | hidden做functional LayerNorm后生成Q/K/V；Q/K经RoPE即进入SDPA，没有另做head RMSNorm、head LayerNorm或L2 normalize | **C** 当前候选无显式QK norm；hidden LN不是QK norm。**S** 无Q/K norm参数名只能辅助，不能证明所有可能的无参归一化；**N** 无CA QK norm | [QKV→RoPE→SDPA](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:57) |
| 10 QKV Linear bias | self qkv144→432、attention output144→144均有bias；所有列出的Linear构造未关闭bias。没有CA query/KV Linear | **S** qkv.bias432、out.bias144等参数记录直接支持。**C** 默认带bias构造。**N** 无CA bias对照 | [Linear构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:47)、[qkv bias shape](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:76) |
| 11 父格绝对位置输入 | 无父格和单独parent-position projection。当前noisy XYZ经Fourier输入Linear；另加learned slot和text。XYZ还驱动self RoPE | **N** 无parent position路径。**C** “带噪坐标内容特征＋slot＋坐标RoPE”可作局部机制比较，但不是已知父格center加法APE。不要借TopologyFlow的`position`字段给PointFlow补模块 | [PointFlow输入](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:119)、[RoPE坐标传递](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:123) |
| 12 3D RoPE轴分配 | 每头36→XYZ各12维/6 adjacent pairs，所有维旋转；strict head_dim%6==0，频率`10000^(-k/6)` | **C** heads4、轴划分、pair排列/频率为候选执行语义，非state shape唯一证明。没有当前NEXUS128/head的余2维方案 | [FORWARD_CONTRACT](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:16)、[rope_3d](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:30) |
| 13 RoPE坐标单位 | 相位为`π·x_t·frequency`，位置输入是当前连续带噪XYZ，无父格整数code、depth或×256。目标点在老师数据说明中按bbox最长边2归一化，初始Gaussian噪声不被函数限制在[-1,1] | **C** noisy coordinate相位和无clamp。数据归一化为已有来源记录，不是`rope_3d`强制的范围；不能写“老师始终对[-1,1]坐标做RoPE” | [相位公式](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:38)、[Gaussian初始化](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:166)、[数据说明](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/source_notes/teacher_delivery_README.md:19) |
| 14 时间特征编码 | t→64维cos32+sin32，`1000t`、base10000、分母32；Linear64→144→SiLU→Linear144→144；与text z相加，不prepend time token | **S** time.0[144,64]、time.2[144,144]。**C** 倍率、顺序、频率、激活和融合；shape仅证明MLP尺寸 | [time_embedding](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:24)、[time MLP](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:102)、[与text融合](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:121)、[time shapes](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:58) |
| 15 FFN激活和宽度 | Linear144→576→GELU→Linear576→144，4×；text/time/prior和AdaLN MLP用SiLU；没有SwiGLU/MoE | **S** 两层FFN shape确认4×尺寸。**C** GELU/SiLU由当前源码确定；checkpoint形状不能证明原activation | [FFN/ada构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:49)、[text/time/prior](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98) |
| 16 主干归一化 | 两个block residual入口均functional `F.layer_norm(...eps=1e-5)`，没有传weight/bias，随后time/text affine modulation；最终head先nn.LayerNorm144（有affine参数）；text/prior入口各有nn.LayerNorm2048 | **C** pre-LN位置、epsilon、functional无learnable affine；**S** output/text/prior norm参数shape。不能从shape单独区分LayerNorm与其他同参数量的norm，更不能恢复原eps | [block LN](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:57)、[FFN入口LN](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:63)、[输出Norm](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:104) |
| 17 输出头 | denoise head为LN144→Linear144,3；随后加text coordinate_prior及α缩放。最终forward输出[B,N,3] clean-point x1_hat，而非8-child velocity；采样器再换算v=(x1_hat−x_t)/(1−t) | **S** output weight[3,144]、bias[3]与prior维度。**C** clean含义、prior合成/速度转换；输出张量3维本身不证明预测x1还是v | [head](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:104)、[forward合成](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:126)、[采样换算](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:167)、[head shapes](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:702) |
| 18 初始化 | 候选constructor创建nn.Linear/Embedding/LayerNorm，没有自定义Xavier、AdaLN-zero或head-zero初始化；α在构造时为1，load_points后被权重覆盖为极小最终值 | **C** 当前constructor没有显式zero/init步骤，α构造值明确。**U** 原训练初始化未恢复；训练后权重或本候选constructor不能证明原始初始化，不能据此声称老师原训练用默认随机初始化 | [block构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:45)、[PointFlow构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:95)、[strict加载覆盖](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:149) |
| 19 Dropout/残差随机丢弃 | 候选SDPA显式dropout_p=0；PointFlow/DiTBlock没有nn.Dropout、DropPath或层随机丢弃；load_points返回eval模型 | **C** 当前推理代码无随机dropout。**U** 原训练dropout未证明，源码FORWARD_CONTRACT明示此边界；不能把候选推理0当原训练0 | [SDPA参数](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:60)、[eval加载](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:154)、[训练dropout边界](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:22) |

主报告子项2.1“VecSet heads”、2.2“VecSet末尾norm”、4.1“条件编码器QK norm”也都是 **N：无对应VecSet模块**。不能把PointFlow的4 heads、text入口LN或生成主干归一化填进这些条件编码器项目。源码入口是[PointFlow.__init__](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:95)；缓存文本而非VecSet的读取见[replay.py](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/replay.py:67)。

## 第二部分：NEXUS原生接口/层级/条件要求的老师对照

| NEXUS接口或约束项目 | 最终老师候选实际做法 | 类型/可比较范围 | 定位 |
|---|---|---|---|
| 原生token输入：每occupied parent的8 child occupancy | PointFlow接受x_t[B,N,3]当前连续带噪XYZ；经Fourier39投影为hidden[B,N,144] | **S/C** 老师输入投影39维、输出3维；**N** 无8-child occupancy token。不能将老师坐标输入称为NEXUS原生输入 | [denoise形状检查与投影](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:114)、[input shape record](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:51) |
| 原生输出：8 child occupancy的flow velocity | 老师最终forward直接返回[B,N,3] x1_hat；sampler转为velocity再Euler更新 | **C** X1参数化，不是直接velocity输出；**N** 无8-channel occupancy输出、阈值化孩子占用或octree重建 | [forward](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:126)、[Euler](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:167) |
| 动态点数/父格数量 | 老师text z144→SiLU→Linear275 logits，argmax类索引作为N；要求1≤N≤274，类0预测会报错；slot和prior仅取前N项；采样逐对象运行 | **S** logits275类、slot274、prior822=274×3；**C** argmax/count检查/前缀截取。不是通过octree 8-child decisions累积点数，也不是读GT count | [count构造/调用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:99)、[N预测与验证](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:158)、[prior截取](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:128) |
| padding/有效token mask | forward/denoise可传mask[B,N]，SDPA广播为[B,1,1,N]；候选只有key维attention mask，没有显式清零无效query或输出。实际sample_points按预测N创建无padding张量，调用model时不传mask | **C** Mask能力与实际单对象采样用法不同。不能写“老师mask同NEXUS端到端清零一致”。重建训练loss用mask做误差归约，但那不是原训练loop证明 | [attention mask](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:60)、[sampler无mask调用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:169)、[重建loss mask](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/objectives.py:44) |
| octree层级和grid坐标 | 没有octree、parent_codes、整数[0,511] raster或按深度level的数据接口；输入连续XYZ | **N** 没有可映射模块。老师目标点bbox最长边2只是数据归一化，不能等价为D9占据表示 | [完整PointFlow入口](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:114)、[数据说明](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/source_notes/teacher_delivery_README.md:19) |
| 多depth共享生成网络 | 同一18块PointFlow用于不同N，Euler每步重复同一网络；没有depth参数、depth调度或跨octree层次执行 | **C** 在时间步/点数上复用网络。**N** 不构成NEXUS跨depth共享的实现证据，不能把18个block当18个octree层 | [PointFlow块循环](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:122)、[Euler复用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:167) |
| learned depth embedding | 老师只有274×144 slot embedding，表示按点序号的身份，没有depth embedding | **N** 无对应depth字段/模块。slot与depth的语义和粒度不同，不做强映射 | [slot构造](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:100)、[slot使用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:119) |
| 点云XYZ+normal→VecSet 1024×2048条件tokens | 老师接收单个缓存文本向量e[B,2048]；text模型与tokenizer没有交付。没有8192点输入、normal、1024个condition tokens、VecSet或者点云条件mask | **S/C** text width2048是维度碰巧相同，不是同表示。**N** 无点云条件编码器或CA；不能宣称同NEXUS conditioning接口 | [text shape/调用](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98)、[采样特征检查](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:161)、[缓存读取](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/replay.py:67) |
| conditioning的注入方式 | 全局text z同时①加到每点hidden，②加到time向量后生成SA/FFN的AdaLN shift/scale/gates，③供count head；prior有另一条独立text2048→坐标输出路径 | **C** global condition双路径可作结构机制参照。**N** 没有condition-as-K/V cross-attention。要借双注入可单独适配，不引入coordinate_prior、count或slot | [denoise双注入](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:118)、[prior独立分支](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:105) |
| VecSet与DiT联合训练/条件池化 | PointFlow当前代码只有text LN/MLP，没有VecSet训练状态。已有文本缓存由原prompt最后隐层masked mean生成的说法来自REPORT；无法在交付原encoder源码中核验该流程 | **N/U** 无VecSet联合训练对应。若新增VecSet全局摘要，先pool到2048再LN/MLP是迁移方案；逐token LN/MLP后再pool是另一个方案，二者一般不交换 | [原encoder缺失说明](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/REPORT.md:51)、[text投影](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98) |
| Flow matching路径/训练目标 | sampler t=i/100，i=0…99，v=(x1_hat−x_t)/(1−t)。`point_velocity_and_count_loss`重建构件使用x_t=(1−t)noise+t target，velocity MSE+count CE；count coefficient显式要求传入，声明不代表最终prior训练目标 | **C** 当前sampler/重建loss公式。**U** 原训练time分布、loss权重与完整loop未恢复；不能用该helper证明最终候选按该loop训练。与TopologyFlow直接velocity路径必须区分 | [sample_points](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:157)、[重建点loss与声明](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/objectives.py:44)、[prior loss边界](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/objectives.py:63) |
| 多层采样器/20步DPM-Solver | 老师点候选是全局XYZ单阶段100步Euler；不同TopologyFlow另有50步Euler，但不是点模型的octree递归 | **C/N** 无逐octree层DPM-Solver对照。不能将拓扑阶段sampling次数借给点模型 | [点Euler](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:157)、[独立拓扑Euler](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:174) |

### 全局条件路径的可读公式

当前候选执行逻辑（**C**）：

```text
e: cached global text [B,2048]
z = Linear144(SiLU(Linear2048→144(LN2048(e))))
tau = Linear144(SiLU(Linear64→144(cos/sin(1000t))))
h = Linear39→144(Fourier6(x_t)) + slot[:N] + z[:,None,:]
condition = tau + z
h = Block17(...Block1(Block0(h, condition, x_t), condition, x_t)...)
d = Linear144→3(LN144(h))
p = reshape274×3(Linear512→822(SiLU(Linear2048→512(LN2048(e)))))
x1_hat = p[:,:N,:] + alpha*d
```

z的text LN/MLP和p的prior LN/MLP是两套参数；z与tau是两个独立MLP输出相加。**没有条件tokens池化、没有CA、没有depth，也没有把text先验当作训练目标GT读取。** [实际forward](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:98)。

## 状态与验证边界

1. 既有`checkpoint_structure.json`记录point final的hash为`e286272f2589defe17b72366d0cf88a44b839cf77b150cdf6257df27d2a5e4de`。这是历史结构审计的来源标识，本轮没有对原checkpoint重新hash。
2. 该记录candidate Adam只有prior LN/Linear的六个参数槽+α一个标量槽；slot step50000，metadata prior_steps160000。这不是完整18层denoiser的optimizer/RNG。[optimizer记录](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:1485)、[七个slot](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/checkpoint_structure.json:1528)。final latest没有optimizer/RNG。[README界定](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/README.md:75)。
3. `source_step=267500`与`frozen_denoiser_source_step=258000`不是同一条一致计数，保留差别；state_dict不编码requires_grad，冻结规则来自训练器。[PointFlow注释](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/models.py:89)、[原记录差别](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/REPORT.md:118)。
4. 既有CPU纯噪声点采样记录验证的是候选函数；本轮没有再次执行。两seed点数50/50、坐标RMSE约1.1e-7，不证明原denoiser独立成功、独立重训成功或泛化。[point_final记录](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/point_final/RESULT.json:1)、[第二seed](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/evidence/point_final_seed98765/RESULT.json:1)。

本轮读取文件的当前SHA256：

| 文件 | SHA256 |
|---|---|
| models.py | `679c6482a47ca40ed3bcb5e86eb6f407db105a5fb6cc41a6c8708b6e64127a0a` |
| teacher_ae.py | `740de899b2d7ea47ec56edf61101d45dee6251956505426e16c1f5cec9f6416c` |
| objectives.py | `882af74504ccf714c1a5f9267e3adecf8d1db62fc6301de28eaa5dd038f58296` |
| evidence/checkpoint_structure.json | `7b0295c2d45bb80e0565c0271bbe9b28ae7b13e471d6f80ba5bc8abd3fa8a40c` |
