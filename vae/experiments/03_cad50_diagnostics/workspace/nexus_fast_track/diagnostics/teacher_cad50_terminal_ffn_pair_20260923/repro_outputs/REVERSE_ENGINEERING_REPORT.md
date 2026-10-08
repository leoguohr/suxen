# 原始老师 ZIP 逆向工程：恢复范围、实测结果与原512差别

2026-09-23，GPT-6 Astra / xhigh。本报告由本次实际恢复实现的执行者整理，只读取已有原件核验、代码、训练记录及评价；本报告编写没有执行训练或新增网络计算。末端FFN实验的最终结果由另份实验报告填写，本文不预写其成败。

**三个老师交付网络的可运行推理实现已经恢复并实测；老师完整训练流程尚未恢复。** 老师AE原权重在本次实现上达到Face F1 `0.9991024951`，独立随机初始化重训20,000步达到 `0.9905379832`。这两个结果分别回答“能否执行老师权重所代表的函数”和“我们的训练能否重新找到同样好的权重”，不能混为一个成功结论。

## 1. 原始输入与本次实际做了什么

唯一原件是用户指定的 `nexus_overfit_data_results_no_code.zip`，大小 **241,939,514 bytes**，SHA256：

```text
982686feb58c207932d9311786cabb3ffd7a1397627b83c44df3d5c8ab2189e5
```

它与昨天所用原包字节相同，没有新增老师文件。此次重新读取全部 **1,330个成员**，逐项CRC通过，无路径穿越、符号链接或重复名；全部 **22个PyTorch文件**重新解析，包含各阶段权重、优化器、RNG、配置与历史记录。解析使用严格白名单，将序列化张量恢复为惰性描述及存储hash，没有执行未知原包源码。

随后重新编写三网络、评分、采样器、loss组件和评价入口，实际运行本次代码。我们此前已经读过昨天的候选；权重不能唯一确定的前向约定显式承接其已验证选择。因此这是有来源与数值证据的再次恢复，**不能称盲法独立恢复，也不能称找回老师原源码**。

直接证据：[原件再核验总报告](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/ORIGINAL_ZIP_REVERSE_AUDIT.md)、[原ZIP成员清单](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/FILE_MANIFEST.json)、[全部checkpoint元数据](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/ORIGINAL_CHECKPOINT_METADATA.json)、[原训练记录](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/ORIGINAL_TRAINING_RECORDS.json)。清单文件本身SHA256为 `44ba53e1ab53ff95c7b03d14b9316e0a032c90d901ad9c39d71b221d28826c87`。

## 2. 三个网络分别恢复到了哪里

| 网络 | 原件直接支持的结构 | 本次已完成 | 尚不能宣称 |
|---|---|---|---|
| 拓扑AE/VAE | 8层Graph和Encoder Transformer，宽128；每顶点latent64；8层Decoder；输出两个32维表示；3,476,032参数 | 原件244个状态项全部同名strict加载；真实Encoder→μ→Decoder全50重建 | 原训练时dropout、后验采样/clamp、KL归约、负例顺序和完整阶段训练器已恢复 |
| 顶点条件拓扑Flow | 10层DiT、宽144，输入/输出latent64，顶点位置条件；3,809,008参数 | 112项全部strict加载；固定输入速度场、50步Euler和全50串联已执行 | 精确原采样噪声跨平台一致；已用独立重训AE训练出匹配的新Flow |
| 文本条件点网络 | 18层DiT、宽144，2048维文本特征，274个有序点槽位，275类点数头；额外文本坐标先验2048→512→822；8,668,301参数 | 206项全部strict加载；点数预测、100步采样及全50点到拓扑生成已执行 | 原始raw-text编码器/Tokenizer、论文octree实现、完整原点训练器已恢复；未见文本泛化已验证 |

三套strict加载均 **missing keys=0、unexpected keys=0、原件覆盖率100%**。没有忽略键，也没有为通过加载而偷偷重命名或删层；[完整双向state映射](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/reconstructed_from_original/STATE_KEY_MAPPING.json)均为identity。

老师最终点输出是学习的文本坐标先验加去噪残差，原件残差系数 **`1.2198240256111603e-5`**。精度主要来自先验对50条训练提示的记忆。原报告也承认这一点；不能把它解释成原始纯点diffusion已单独收敛。生成函数用缓存文本向量和噪声，GT只用于评价；学习先验也不等于推理时直接读取GT坐标。

证据等级保留四类：A为原张量形状/字段，B为原文字记录，C为采用并数值验证的前向假设，U为仍未知。4个attention head、Fourier具体排列/π尺度、RoPE/time/AdaLN次序等属于C，而不是单凭state_dict唯一推出的事实。完整表见 [CONFIG_PROVENANCE.json](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/reconstructed_from_original/CONFIG_PROVENANCE.json)。

## 3. 本次新实现与昨天候选有什么实际区别

本次主要实现文件为 [recovered_networks.py](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/reconstructed_from_original/recovered_networks.py)，SHA256：

```text
ba699f37e62c206dd248cd0d48608538f702a9f0d9d92713077b4f7b89e9d29f
```

昨天AE候选 [teacher_ae.py](/Users/luthier/Downloads/Nexus_teacher_full_reconstruction/teacher_ae.py) 的SHA256为 `740de899b2d7ea47ec56edf61101d45dee6251956505426e16c1f5cec9f6416c`。两份实现写法不同：昨天手写Transformer前向，本次用标准 `nn.TransformerEncoderLayer`；DiT/位置旋转/采样器也重新编写，保留已验证数学约定。本次KL组件返回每元素值，让未确定的归约保持显式；取mean时与昨天组件一致。

同运行时、同原件、同输入的实测表明：

- 全50 AE的Edge和Face embedding **逐元素最大差均为0**，50/50预测Face集合相同。
- CAD02、CAD00两个配对采样中，点坐标最大差 `1.8626e-9`、拓扑latent最大差 `9.5367e-7`，两条预测面集合相同。它们分别重置种子，不混作全50连续噪声流。
- 完整全50串联使用点seed34567、拓扑seed12345，各generator仅初始化一次，按00→49执行；全部50个预测面集合与昨天固定第一组CPU记录相同。
- 七项固定输入loss比较差值均0，BCE/KL梯度有限；这验证可微公式，不能验证缺失的训练loop。

恢复同一函数的合理结果可以是行为相同，而不是必须制造结构差异。以上实测既说明本次代码确实可运行，也说明没有借重新逆向悄悄更换评分、阈值或权重。逐函数对照见 [COMPARISON_WITH_YESTERDAY.md](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/reconstructed_from_original/COMPARISON_WITH_YESTERDAY.md)。

## 4. 原权重重放、独立重训、原512父点是三个不同结果

以下均指CAD50的真实网络评价；joint strict要求每条Edge/Face的FP/FN均为0。表中的训练起点和预算不同，不能按F1直接归因某一结构。

| 对象 | Edge FP/FN | 实际Face FP/FN | Face micro-F1 | joint strict |
|---|---:|---:|---:|---:|
| 本次新逆向AE＋老师原权重 | 0/2 | 0/10 | **0.9991024951** | 49/50 |
| 老师式小AE，随机初始化独立重训20,000步，冷加载 | 13/39 | 25/80 | **0.9905379832** | 37/50 |
| 原512 CAD50共同父B2500，原保存且核验的全50评价 | 5159/1656 | 5359/3255 | **0.3501810501** | 32/50 |

老师原权重重放达到AE门槛0.997，但仍不是50/50严格成功；相对老师原保存的Face F1 `0.9991923180`，当前CPU重放在CAD00还差一个Face。独立重训未达0.997：它的最佳Face F1在20,000步，严格覆盖峰值38/50在19,000步；最佳点80个Face FN中，74个缺预测Edge三角候选，6个是已有候选判负。这是错误发生位置，不能据此认定唯一训练根因。

独立重训每步抽5次mesh，共100,000次抽样；固定8层从随机初始化开始。老师原记录则是累计AE130,000步、2→4→6→8层、阶段fresh AdamW、后期困难样本抽样，日志还记录LR从1e-4降至5e-5、2.5e-5。本次没有重演这一完整路径；原“uniform groups”、精确负例函数、初始化和RNG次序也没有补齐。

因此，原权重高分说明候选前向能表达已获得的高质量解；20k重训较低说明这次独立训练尚未到达同质量。**既不能由前者宣称训练完全复现，也不能由后者证明结构表达上限或保证再加步数必过。** AE重建不经过点网络和拓扑Flow，不能用另外两个模型尚未重训来解释这个AE差距。

本次完整点→拓扑串联的Face F1为 **`0.9894037356`**，TP/FP/FN=5509/51/67，joint strict41/50，未达串联0.99。点数50/50正确，坐标RMSE `9.3614062e-8`；总耗时114.46秒。本次CPU torch2.3.0a0与昨天CPU torch2.10.0不同，不能保证所有浮点数逐位一致。先前A100两组高于0.99属于已存的另一运行环境结果，本次没有重跑GPU，也不据平台选择把所有验收概括成通过。

对应原始评价和身份：

- [本次AE验证JSON](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/reconstructed_from_original/VERIFICATION_RESULT.json)，SHA256 `acb1177c4d6daa509b2a65efef97f32b10d55ac37b8586a8d53faf13b870660d`。老师AE权重成员SHA256 `2d960c533d878f82280bb5f950c015fc5641c45c9b9bebdcd81a91560333037d`。
- [本次全50串联JSON](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/reconstructed_from_original/cascade_cpu_34567_12345/CASCADE_RESULT.json)，SHA256 `c33cf536b081c66a4b39480343bbc116bd2f422d291a8f9948d267b7560a82de`。
- [独立重训冷加载JSON](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_style_ae_scratch_20260922/repro_outputs/COLD_VERIFY.json)，SHA256 `33935f856384c4b7ebeb869582759bf9155b3f0f5479d79457a8217bdfbab15c`；最佳权重为 `/guohaoran/nexus_fast_track/diagnostics/teacher_style_ae_scratch_20260922/train_outputs/checkpoint_020000.pt`，SHA256 `248f62e2e4a2f6366b0550e99eb0a5fbc2703cfa1ed2ce8134913895104bcacd`。完整训练口径见 [重训报告](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_style_ae_scratch_20260922/repro_outputs/REPORT.md)。
- [B2500评价JSON](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_reverse_audit_20260922/repro_outputs/OUR_B2500_EVAL.json)，SHA256 `9914b03433869ce4c6ab58f45384e8c412b1f2b306b6483678d015bfb8b2461c`；共同父权重SHA256 `4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`。它是本次FFN实验的父点，不是H实验终点。

## 5. 已恢复的训练线索与仍缺的状态

| 原件中的内容 | 直接核实 | 对训练复现的限制 |
|---|---|---|
| 最终AE三份权重 | `best_ae.pt`、`vae.pt`、`latest.pt:vae`的244项逐张量形状/存储hash相同 | 最终AE optimizer不在这些文件中 |
| 拓扑`latest.pt` | stage=flow；112个Adam槽与Flow完整有序形状匹配；当前槽step14000；含Python/NumPy/torch/CUDA RNG | 是最终Flow阶段状态，不能冒充AE结束时Adam/RNG；累计Flow计数128000与槽计数不同本身不是错误 |
| 点`latest.pt` | 206项推理state和配置，无optimizer/RNG | 不能恢复完整点训练现场 |
| 点`candidate.pt` | 与最终点state逐项相同；优化器只有6个prior张量＋1个scale；当前槽step50000，prior累计160000；无RNG | 不是18层去噪器完整优化器；缺原cached-frozen-output及其构造 |
| 点来源步数 | candidate的267500与推理说明的258000不一致 | 保留为未解释元数据差异，不猜补唯一历史 |

有了权重、可微函数甚至某阶段Adam/RNG，仍缺原训练过程如何产生每一个batch、负例、随机latent和阶段切换。最终权重不能唯一决定这些过程。可以实现一个合理独立训练器，但应逐条标明选择，不能倒称它就是老师原训练器。

昨天README已指出上述主要缺口；本次新增的是全部22个checkpoint的原件级再次取证、完整状态对应、日志出处，以及重新编写代码的实际全50验收。详见 [状态关系与优化器映射](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923/DERIVED_RECOVERY_EVIDENCE.json)。

## 6. 与原512有效实现相比，哪些结构差异确实存在

| 部分 | 已验证老师小AE候选 | 原512有效骨干 |
|---|---|---|
| 输入 | XYZ＋6频带sin/cos，39→128共享投影 | vertex XYZ、face重心分别3→512投影 |
| Graph | 原h聚合，经self/neighbor线性后LN、GELU残差 | h先LN，再self/neighbor线性、SiLU残差 |
| Encoder | 8组Graph＋Transformer，宽128 | 12组Graph＋Transformer，宽512 |
| latent | 每顶点64 | 每顶点512；当前CAD50协议用μ、sampling off、KL0、logvar冻结 |
| Decoder | 8块宽128，每块有attention和逐点GELU FFN | 16块宽1024，attention残差，无FFN |
| 读出 | LN＋Linear(128,64)，拆成两个32维表示 | LN＋两个Linear(1024,32)，逐mesh中心化 |
| 评分 | 16space＋16time，零符号阈值 | 同一评分族，保留固定正scale及Face面积因子0.25 |

老师侧的运算次序等仍按A/C证据等级区分，不能把数值兼容的候选自动升级为唯一原始源码。原512结构来自实际源码及动态前向：

- [原512骨干快照](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_reverse_audit_20260922/original_512_topology.py)，SHA256 `d8006b84f335b16c80042dd492ddbd769156309acac3a35979cfe124b791ba7d`。
- [原512有效前向](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_reverse_audit_20260922/original_512_effective_forward.py)，SHA256 `767e951d74772436a8a8a53a5557f965dfa853d5e91d4c499e51dfc93e09c892`。
- [固定100结构及失败诊断](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/fixed100_teacher_comparison_20260923/FIXED100_FAILURE_ANALYSIS.md)，本次读取SHA256 `0bcbde61df9edfbf849478eb52d4d34bab4d57f412bc8d2fcb78c0737d64cb68`；详细源文件行号、历史模型身份和42份证据见其同目录 `EVIDENCE_INDEX.json`。

一个64维Linear拆两份与两个32维Linear在函数形式上可以等价，不能用“head分开”直接解释失败。Fourier、FFN、Graph归一化确有差异，但架构、数据规模、目标、Adam/AdamW、采样和预算同时不同，现有比较无法唯一归因。

此外，“原512 CAD50”和“原512固定100”共享骨干不等于同一实验：本次CAD50继续训练全部Encoder/μ/16层Decoder/两head；固定100最新last2/last3分支只开放末两/三块、LN和head。固定100最大2547点，CAD50最大274点；固定100最新两支都是73/100，其Edge FN与GT面缺候选均0，不能把小AE的漏边机制直接套给它。固定100的历史诊断支持大mesh共享表示/读出及Edge/Face优化取舍存在困难，尚未确认某个单一结构缺陷。

## 7. 末端FFN实验现在检验的只是一个有界假设

待检假设是：**在原512 CAD50这个B2500父点，保留原目标和完整旧状态，仅在Decoder末端加入一个可学习的逐点非线性残差，能否在相同100次全50更新预算内改善实际重建。**

新模块为pre-LN(1024)→Linear(1024,4096)→GELU→Linear(4096,1024)残差，位置在最后attention残差后、原输出LN前；末Linear权重/bias为0、dropout0。初始化保持旧函数，新模块随后可以学习。原Soft4、pool、LR、μ、旧Adam/RNG、更新口径等保持既定协议。它增加的是一个FFN，不是第17个attention块，也没有把每个原Decoder块都改成老师结构。

该实验可能给出这个父点、这段预算下的受控证据；即使有改善，也不能自动证明FFN是历史所有失败的唯一原因，或证明原100条/全模型必然通关。若无改善，同样不能证明所有FFN设计或该网络容量无效。新增梯度参与原global clip，因而首步旧参数位移无需与Control相同；真正检查的是零步函数/预测及旧参数裁剪前梯度等价。

本报告不引用H任何训练终点或中途优劣。FFN验收应在实验报告中分别列Face F1门槛、严格UID、Edge/Face FP/FN、Face缺候选/有候选判负以及对照净变化；不能以loss下降代替实际重建。

## 8. 交付定位

本报告是本轮FFN实验交付中的逆向工程部分。完整可运行恢复代码、原件证据、命令、50条预测与新旧数值对照保留在 [teacher_original_zip_reaudit_20260923](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_original_zip_reaudit_20260923)；原训练器缺口也在那里逐项标明。训练效果与FFN假设是否获得支持，应继续引用各自明确的checkpoint和独立实验报告，不能把函数恢复、加载旧权重和从零学习成功合并成一个“已复现”。
