# Topology Flow：结构、接口与实验范围

本包记录用户固定50条 mesh 上的 Topology Flow overfit 实验及其结构候选。它是采用自有冻结 OwnAE-v2 的本地实现与实验记录；论文规定、源码选择和实验观察分别陈述。打包阶段只读取已有文件，没有补训练或补评价。

## 1. 任务和固定数据

给定原始GT顶点坐标与真实点云XYZ/法向，一个共享模型从独立高斯噪声生成逐顶点 topology latent，再由冻结 Decoder 输出Edge与实际Face。本实验只覆盖GT顶点上的Topology；没有验证 Vertex 预测顶点与Topology的完整串联，也没有加入UID embedding或独立mesh head。

| 项目 | 实际范围与证据 |
|---|---|
| 固定名单 | 用户指定50个唯一UID；[selection50.json](../snapshot/initial/configs/selection50.json)。不是另行挑选的CAD50。 |
| 数据规模 | 52,820个顶点、154,978条无向Edge、102,890个无向Face；单mesh 18–2547顶点。 |
| 计数差异 | `nexus_2k_001825` 沿用用户批准的1277点原拓扑，原表1279仅记录为差异。 |
| 拓扑与编号 | 保留已有局部顶点顺序、原坐标和GT标签；不重新量化、焊接或简化。逐UID数组和来源绑定见[数据身份](../snapshot/initial/inputs/data_identity.json)、[缓存清单](../snapshot/initial/cache/manifest.json)。 |
| 点云条件 | 已有Stage2的8192个XYZ+法向，输入 `[8192,6]`；不从GT拓扑临时构造条件。 |
| 坐标系 | 现有Stage2归一化点云与Stage3 D9中心顶点处于同一坐标系；校验了来源、变换、点到来源面的残差及单位法向。[坐标核验](../snapshot/initial/inputs/coordinate_audit.json)。 |
| RoPE单位 | 原归一化GT XYZ乘256，作为本地D9参考网格单位；三组固定相同scale，没有做尺度扫描。Fourier输入使用乘256之前的归一化XYZ。 |

来源面信息只用于条件来源核验，不作为Flow的推理输入。训练会用GT拓扑通过冻结Encoder得到目标后验；生成路径只接收顶点、真实点云与高斯噪声，不调用Encoder读取GT拓扑。

## 2. 冻结AE与latent接口

使用 `NativeTopologyAE / B_v2_teacher_blocks / latent512` 的 OwnAE-v2 step36220。整个AE/VAE冻结；Flow和独立点云条件编码器参与训练。源码里的 `B_v2_teacher_blocks` 是既有自有AE的结构标签，不代表本实验加载了老师权重。

| 接口 | 实际约定 |
|---|---|
| token语义 | 每个原局部顶点对应一个512维latent，模型输入/输出为 `[B,N,512]`。 |
| 后验缓存 | 固定50条的μ/logvar；每次参与训练时重新采样后验。logvar采用既有 `[-20,10]` clamp。 |
| 固定归一化 | 512维逐通道统计，先对每mesh顶点求均值、再对mesh等权平均；均值为 `E_mesh E_vertex μ`，方差为 `E_mesh E_vertex(μ²+exp(logvar))−mean²`。FP64累积、FP32保存，std下限1e-6；本次没有通道触及下限。 |
| 逆变换 | 生成标准化latent后使用同一组mean/std逆变换，直接进入同一冻结Decoder。续训不重新估计统计。 |
| 一致性检查 | 50条导出—回读—Decoder回解的hidden/Edge/Face差异为0。这证明缓存接口一致；VAE重建本身仍有错误。 |
| Decoder | 既有16层、1024宽Decoder，独立32维Edge/Face特征头；保留原评分、去均值和阈值。 |

实现依据：[vae_codec.py](../snapshot/c0/code/vae_codec.py)、[latent_data.py](../snapshot/c0/code/latent_data.py)、[export_cache.py](../snapshot/c0/code/export_cache.py)、[缓存清单](../snapshot/initial/cache/manifest.json)。

| 身份 | SHA256 |
|---|---|
| OwnAE checkpoint | `e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15` |
| selection50 | `ee2588bcd212e750893904973b4aee4e3a40d3295f35b2d923f6d92076fa0ac4` |
| cache manifest | `f14f796c421cab6b9d5206047f927ea7c3490ff3c2435b2ebd38904a749c27e3` |
| 固定标准化语义哈希 | `a144fa69f836d15282acd1173bfee4ca3027fd832828d4ab9303a74f8ca44f01` |

## 3. C0、C1和C2实际启用的结构

三组共享主干36层、宽1536、12头、FFN扩展4倍；独立点云条件编码器8层、宽2048、16头，将8192点编码为1024个条件token。其局部实现为FPS查询、1个cross-attention块加7个self-attention块。Topology块顺序为self-attention→cross-attention→FFN。

| 模块 | C0 | C1 `c1_fourier` | C2 `c2_teacher` |
|---|---|---|---|
| 绝对坐标输入 | 无额外投影 | 39维Fourier→1536，加入latent输入 | 与C1相同的39维特征和注入位置 |
| Fourier内容 | — | 原XYZ及各轴6个频带的sin/cos，频率 `π·2^k, k=0…5` | 同C1 |
| Fourier投影初始化 | — | 权重与bias为0；保留C0共享张量的seed0初始化及后续CPU RNG | 权重Xavier、bias为0 |
| 主干QK norm | SA/CA均为每头RMSNorm | 同C0 | 主干SA/CA的QK norm均删除；条件编码器不随之修改 |
| 主干LayerNorm | eps=1e-6；SA/FFN非affine，CA affine | 同C0 | eps=1e-5；各分支affine设置同C0 |
| 最终LayerNorm | 非affine，eps=1e-6 | 同C0 | affine，eps=1e-5 |
| FFN激活 | GELU tanh近似 | 同C0 | 标准GELU |
| RoPE | self-attention Q/K上的3D RoPE，scale256 | 同C0 | 同C0 |
| 时间与门控 | 256维正弦时间特征→MLP；每块六路time AdaLN调制SA/FFN的shift、scale、gate；CA无独立gate | 同C0 | 同C0 |
| 初始化与输出 | 线性层Xavier、bias0；time MLP权重Normal(0,.02)；AdaLN末层、CA输出层、最终velocity输出层为0 | 除新增Fourier外同C0 | 除列明的模块变化外沿用本地初始化；不声称还原了老师原trainer的初始化 |
| 输出头 | LayerNorm→Linear512；无最终time AdaLN；预测velocity | 同C0 | 同C0，最终LN差异见上 |
| Flow参数 | 1,930,898,432 | 1,930,959,872 | 1,930,741,760 |
| 条件编码器参数 | 402,987,008 | 402,987,008 | 402,987,008 |
| 总参数 | 2,333,885,440 | 2,333,946,880 | 2,333,728,768 |

C1借鉴已有候选中的Fourier+RoPE机制；C2是更完整的微结构组合迁移，同时保留真实点云CA、约2B规模和OwnAE512接口。因此C2是本地混合适配，不能称为老师模型原样复现，也不能把其比较解释为单一QK norm消融。老师自己的latent、Decoder、权重和训练后验选择没有直接迁移。

三组保留局部编号、变长mask和逐顶点输出；没有Topology用的octree depth embedding、顶点index embedding或固定mesh slot。联合置换latent、顶点与mask的等变性和padding隔离有小尺寸CPU检查；该检查不证明真实生成成功。

有效配置与实现：[C0配置](../snapshot/c0/configs/user50_recipe.json)、[C1配置](../snapshot/candidates/configs/c1_fourier.json)、[C2配置](../snapshot/candidates/configs/c2_teacher.json)、[候选TopologyDiT](../snapshot/candidates/code/topology_flow.py)、[共同block/条件编码器](../snapshot/candidates/code/_vertex_reference.py)。候选参数差额可直接由增加的39→1536投影、最终LN与删除的QK norm计算；它们不表示训练结果优劣。

## 4. 统一训练与恢复约定

1. 对本次mesh从冻结μ/logvar重新抽取后验z，然后按固定统计标准化。
2. 独立采样标准高斯epsilon和均匀 `t∈[0,1]`；使用 `x_t=(1−t)epsilon+t z`，velocity目标 `z−epsilon`。
3. 先在每个完整mesh内部平均velocity MSE，再对参与更新的5个mesh等权平均。microbatch=1，累积5条后更新；50条每epoch遍历一次，即每epoch10次有效更新。
4. 后验、Flow初噪声和时间使用独立CPU随机流。同一次重计算使用已采样的相同张量。
5. seed0；fresh AdamW，lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=.01、global clip=1；前100次有效更新按 `lr*min((completed+1)/100,1)` warmup，随后固定。
6. C0后500次属于完整状态续训；C1、C2各自fresh起训。恢复点包含model、AdamW、独立随机流、CPU/CUDA RNG、有效更新数及数据游标；配方身份与可追加阶段预算分开。

实现依据：[flow_training.py](../snapshot/candidates/code/flow_training.py)、[train_flow.py](../snapshot/candidates/code/train_flow.py)、[C0恢复检查](../snapshot/c0/resume500_cpu.json)。1000次五mesh更新意味着每条mesh参与100次。C1实际904次对应90整轮加20条参与第91次；C2实际1049次对应104整轮加45条参与第105次。

精度固定FP32、TF32关闭、SDPA MATH、确定性算法。候选执行前校准选择全部block直接执行而不做activation重计算，未改变上述结构、精度或训练目标；校准报告记录优化器更新0、Adam steps `[0]`及所比较梯度bitwise一致。[校准报告](../snapshot/candidates/perf/report.json)、[实际执行策略](../snapshot/candidates/perf/execution_policy.json)。这一数值门只覆盖已比较的校准样本，不应扩展为所有训练轨迹必然逐位相同的结论。

## 5. 真实生成与解码边界

每个checkpoint独立评价相同50UID，seed0按UID派生固定初始高斯 `[1,N,512]`。以本地显式Euler50从t=0积分到t=1；没有近目标latent起步、clean投影或生成中调用Encoder。每mesh内可复用一次点云条件编码，生成latent逆标准化后进入冻结Decoder。

所有无序顶点对按原Edge logit>0判边；从当前预测Edge图完整枚举闭合三角形，再用冻结Face评分判面。候选外GT Face计FN，不补GT边、不截断候选、不修复结果。Edge/Face按无向集合计TP、FP、FN和micro-F1；严格Edge要求该mesh Edge FP/FN均0，联合严格要求Edge/Face四项FP/FN均0。只有同checkpoint、同噪声条件完整50条均联合严格才记严格50/50。

评价身份、代码哈希、采样设置及完整性由各 `identity.json`、`summary.json` 和逐UID `metrics.json` 绑定。部分样本的结果可以分析已评子集，不能生成全50成绩或与另一checkpoint拼接。依据：[evaluate_flow.py](../snapshot/candidates/code/evaluate_flow.py)、[sampling.py](../snapshot/candidates/code/sampling.py)、[Face枚举](../snapshot/candidates/code/_faces_reference.py)。

## 6. 论文规定与本地选择

下表压缩已有结构调研结论；论文页码以arXiv v1 PDF从封面起计。历史结构研究不等于本次新实验的结果。

| 检查项 | 论文依据 | 本任务边界 |
|---|---|---|
| 两阶段与逐顶点latent | NEXUS §3、Eq.(1)、§3.2.1、Eq.(6)，PDF p.3、5–6 | 保留逐顶点语义；本次只给GT顶点做Topology。 |
| 随机AE瓶颈 | Eq.(6)，PDF p.5 | AE随机后验有依据；Flow每次取sample还是mean的训练细节仍作为本地选择。 |
| latent维度 | §4.1，PDF p.6明确64 | 本任务冻结OwnAE512，明确存在维度差异。 |
| DiT、条件、位置 | §3.2.1及§4.1，PDF p.6；Vertex条件注入见§3.1.1，p.4 | 约2B DiT、点云CA与3D RoPE有依据；具体层数/head、六路AdaLN、norm、初始化、Fourier和scale256是当前源码选择。 |
| 生成目标 | Topology Generative Model，PDF p.6 | velocity flow matching有依据；线性路径、时间均匀采样和固定latent统计属于本地明确约定。 |
| Edge/Face | Eq.(3)–(5)、§3.2.2，PDF p.5–6 | 保留spacetime评分与预测图完整闭合三角候选。 |
| 采样器 | §4.1 Topology Diffusion，PDF p.6写DPM-Solver20步 | 当前Euler50必须标为本地采样设置。 |
| 微结构证据 | “采用DiT”及引用其他论文不确定所有实现细节 | 不从Hunyuan/TRELLIS/DiT或候选源码倒推NEXUS已规定这些细节。 |

论文入口：[NEXUS arXiv v1](https://arxiv.org/pdf/2607.13563v1)。既有调研未取得可核实的NEXUS正式源码和独立补充材料；因此不宣称完成逐行官方复现。老师候选源材料、权重及数据不属于本包发布范围。
