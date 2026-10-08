# 老师CAD50：点Flow、拓扑Flow、三种拓扑指标及串联的实际复建

## 一、结论与验证范围

本轮不再只恢复AE。已复建：
1. 最终拓扑Flow与其早期三方法共用的DiT结构；
2. 最终点网络的文本编码投影、点数分类、学习点槽位、18层去噪器和文本坐标先验；
3. 三种拓扑指标及Edge图完整三角候选→Face分类；
4. 从缓存文本向量和随机噪声出发的点→拓扑串联。

这些是本地依据老师权重/配置/参考输出构造的候选源代码，不是老师原始源码。原包明确“不含代码”。**本轮模型optimizer更新次数=0**；加载原权重达到高F1不是从随机初始化训练成功。已实测CPU前向、完整采样以及少量可微目标的梯度冒烟测试。

来源仍为`69e9a5d3-44dc-4773-9713-f4ab473b475d.zip`，SHA256 `982686feb58c207932d9311786cabb3ffd7a1397627b83c44df3d5c8ab2189e5`。没有生成或改写老师权重。

## 二、老师说的“两个diffusion、三种loss”在包内具体对应什么

早期`results/overfit/{cosine,euclidean,spacetime}`每个都有自己的AE、Flow和latent_normalization。共同使用平衡BCE训练拓扑AE，Flow记录MSE；差异是从embedding计算边/面logit的拓扑指标。不能把它们说成三个点diffusion损失，或把三个Flow的latent统计混用。

早期三种方法同预算：AE3000步、Flow4000步，AE两层、Flow四层。后期只继续spacetime，最终AE8层/130000累计步，Flow10层/128000累计步。点网络是另一路，最终含新加的学习先验；不与早期三方法表混算。

来源：`source_notes/teacher_three_methods_REPORT.md`、`teacher_early_config.json`、`teacher_topology_final_status.json`、`teacher_delivery_README.md`。这些是附件原文记录，不是本次训练。

## 三、拓扑Flow的复建

- 输入：每顶点64维噪声latent，以及给定/生成的XYZ；生成时不调用Encoder，也不读GT Face。
- `input`: Linear(64,144)；`position`: Linear(39,144)。39维是XYZ和六频带Fourier特征。
- 时间：64维cos/sin嵌入→Linear(64,144)→SiLU→Linear(144,144)。
- 最终10个DiT block；qkv144→432，output144→144，FFN144→576→144，AdaLN144→864（六组shift/scale/gate）。早期相同结构仅4个block。
- 输出：LayerNorm144→Linear144,64，作为rectified-flow速度。
- 模型最终state有112项、3,809,008个元素；早期52项、1,556,560个元素。
- 采样：z0~N(0,I)，t=0,.02,...,.98，50步Euler；用该AE自己的mean/std反归一化，交给Decoder和对应拓扑评分器。

权重直接确定形状；4个head、3D RoPE的相邻pair布局/π尺度/频率、时间嵌入顺序及AdaLN排列是经过有限前向假设测试后固定的全局候选，**不是由state_dict唯一证明**。在一条训练样本上用教师加噪目标筛选候选后，再用全部50条纯噪声采样验证；没有按UID改变规则，也没有优化任何权重。

## 四、点网络的复建：不是只有原去噪分支

最终`point_diffusion/latest.pt`有206个state条目、8,668,301个元素，配置为18层、width144、max_points274；外部config.json写6层是早期配置，不能覆盖checkpoint。

- 原始文本条件：缓存的2048维向量。
- 文本MLP：LayerNorm2048→Linear2048,144→SiLU→Linear144,144。
- 点数：SiLU→Linear144,275，argmax决定点数。推理不读取GT点数。
- `slot.weight`: 274×144；输入噪声坐标的39维Fourier投影，加slot与文本条件。
- 18个DiT block，时间和文本共同调制AdaLN；输出3维clean-point候选。
- 新增先验：LayerNorm2048→Linear2048,512→SiLU→Linear512,822，reshape成274×3。
- 最终候选：x1_hat = coordinate_prior(text) + alpha * denoiser(x_t,t,text)。
- alpha = 1.2198240256111603e-05，来自原权重；不是本次调参。
- 对X1参数化速度v=(x1_hat-x_t)/(1-t)，100步Euler从随机点出发。

老师README/串联报告明确承认：最终精度主要来自学习的文本坐标先验对50条训练集的记忆，原去噪残差已被大幅缩小。本次以权重和前向核对了这一结构，**不能把最终点结果当作原始纯diffusion单独收敛的证明**；也不能把学习先验称为运行时读取GT坐标的查表——实际生成函数只有文本向量与噪声输入。

原始Text-to-CAD预训练模型、tokenizer及CadQuery生成脚本没有随包交付。已知文本条件来自raw prompt最后隐层masked mean；本轮用缓存向量，没有重新编码新文本，没有重新生成那50条训练CAD。

与论文的区别：Nexus原点生成是多层octree占据扩散；老师此缩小实验是有序点槽位、点数预测及XYZ生成，最终又加文本坐标先验。论文仅作为架构/指标依据，不拿它覆盖老师包内具体事实。

## 五、三种指标

由三套AE各自输出Edge/Face32维表示：

- Cosine：Edge为10*(cos(zi,zj)-edge_threshold)；Face为10*(三对平均cos-face_threshold)。早期config给出cosine_scale10。阈值从该模型state读取。
- Euclidean：Edge为||zi-zj||²-edge_threshold；Face为Gram行列式面积量-face_threshold。这里是“大于阈值为正”，不擅自改成直觉上的“越近越连边”。
- Spacetime：32维分16维space/16维time。Edge为两种平方距离之差；Face为两种Gram面积量之差。不是论文式9的其它Minkowski消融。

三个分支均logit>0判正，实际Face只在预测Edge图完整三角候选中判别。

**前向离散输出验证不能单独恢复所有训练温度/梯度尺度。** 已知的cosine_scale来自配置；其它正比例缩放及所有训练实现细节不由预测集合唯一决定。AE阈值沿用buffer重放；从头训练cosine/euclidean时阈值如何参数化及初始化需另作明确选择，不能声称已恢复。

## 六、实际重放结果

### 早期三方法AE

| 方法 | 老师AE Face F1 | 本次Face F1 | 50条中预测Face集合完全同老师的条数 |
|---|---:|---:|---:|
| cosine | 0.1006504684 | 0.1006459094 | 39 |
| euclidean | 0.0920402250 | 0.0920402250 | 50 |
| spacetime | 0.4946212953 | 0.4945127305 | 48 |

欧氏变体50条保存面集合完全一致；另两种接近但没有宣称逐logit完全一致。最终spacetime AE本轮又运行全50条，仍是Face FP0/FN10、F1约0.999102495、49条Face集合与老师相同，老师是FP0/FN9。

### 早期拓扑Flow（三套权重分别采样）

本次CPU使用seed12345；早期原始噪声张量/完整RNG布局不可用，所以以下是候选采样质量，不是逐数组精确复现。

| 方法 | 老师早期Flow Face F1 | 本次候选Face F1 |
|---|---:|---:|
| cosine | 0.0813371821 | 0.0840114568 |
| euclidean | 0.0559707978 | 0.0640359986 |
| spacetime | 0.3112000696 | 0.3303067954 |

### 最终模型

本次生成的随机数在CPU产生；同名seed不保证与老师Windows/GPU产生同一张噪声。

| 阶段 | 本次条件/seed | 本次结果 | 老师原报告参照 |
|---|---|---:|---:|
| 给定GT顶点、生成拓扑 | 12345 | Face F1=0.9905482042 | 0.9916329285 |
| 给定GT顶点、生成拓扑 | 23456 | Face F1=0.9919006479 | 0.9901934323 |
| 文本条件→点 | 34567 | RMSE=1.107221479e-07, 50/50 count | 6.08063513e-8, 50/50 count |
| 文本条件→点 | 98765 | RMSE=1.104116722e-07, 50/50 count | 6.05511849e-8, 50/50 count |
| 新生成点→拓扑 | 点34567/拓扑12345 | Face F1=0.9894037356 | 原串联0.9910931174 |
| 同一新生成点→拓扑 | 点34567/拓扑23456 | Face F1=0.9928912085 | 原串联0.9910931174 |

两组点采样的最大顶点误差/最小间距均低于0.001，满足老师给出的0.1间距比例门槛，RMSE也低于1e-4。实际点F1不适用，报告点数/坐标指标。

串联第一次点模型生成的NPY是本次真实点网络输出，不是老师`verified_points`或训练GT。第二段读取这些新生成点，再从独立高斯噪声运行拓扑Flow；GT仅用于评价对应关系，不修改输出。

**串联seed12345低于0.99，seed23456高于0.99；不能只保留较好的种子，宣称所有确认种子均已通过。** 它表明剩余网络已形成可运行、接近老师质量的候选，但不是原始噪声/逐层数值/完整训练程序精确恢复。

另以交付的清理后代码从冷启动一次性执行完整50条串联（不复用点目录）：点RMSE=1.106974689e-07，Face F1=0.9894037356，TP/FP/FN=5509/51/67。结果见`evidence/cold_start_full_cascade/RESULT.json`。

## 七、没有恢复的部分

1. 原始Text-to-CAD文本生成模型与CAD代码生成脚本；缓存特征不是新文本编码器本体。
2. 各阶段完整训练器、动态负例精确采样顺序、计数CE权重、所有dropout/KL归约/后验clamp。
3. 原始CPU/CUDA随机数流和每次生成的起始噪声张量；不能要求现有seed跨设备逐位一致。
4. 从随机初始化训练达到老师指标；本轮全部是已有权重重放，optimizer更新0。
5. 论文原octree点生成器；老师这个包并无其权重，不用普通点Flow冒充。

`point_prior_candidate/candidate.pt`和最终点`latest.pt`的state_dict逐项相同。candidate中的Adam只含coordinate_prior6项+scale1项，Adam step50000不是整个网络的160000累计训练步；没有RNG。final点latest无optimizer/RNG。
`minkowski_target_099/latest.pt`中的optimizer属于Flow阶段，不能用于AE；其Adam step与阶段累计步数也不是同一计数。
source_step=267500与frozen_denoiser_source_step=258000元数据不一致，保留此差别，不臆造唯一历史。

## 八、交付代码的使用与后续

先按README在独立目录冷启动重放，用户原100条和CAD512分支不改。现在已有真实baseline代码，不必再凭“老师能overfit”猜网络。

`objectives.py`给出可微训练组件并通过小规模梯度测试，但不冒称原训练loop已经恢复。后续可在明确预算内实现独立重训：AE→固定AE后拓扑Flow→点网络/先验→生成点串联。加载老师权重的成功与独立重训成功必须分别报告。

## 来源

- 用户老师ZIP中的README、results/overfit/REPORT/config/CSV与三套权重。
- results/minkowski_target_099/{best_ae,best_flow,latest,latent_normalization}及预测。
- results/point_diffusion/latest.pt/text_conditions.pt、point_prior_candidate及cascade报告。
- 原论文Nexus: Native Mesh Generation with Diffusion, arXiv:2607.13563v1，§3.1/§3.2/§4.4：https://arxiv.org/html/2607.13563v1 。原文提供octree/拓扑两阶段与指标公式；本次不将其默认当老师缩小实验的全部实现。
