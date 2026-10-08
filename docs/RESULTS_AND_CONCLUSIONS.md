# Vertex Diffusion：结果与结论索引

更新：2026-10-08。原任务是点云／法向条件下的 Vertex Diffusion；产物为顶点集合。VAE／OwnAE-v2 已完成，属于独立拓扑工程，本页不重新评价或收集其训练结果。老师材料是外部对照，不与自己的 octree 成绩合并。

**当前已核实：原结构继续训练的 S0、仅新增 final time AdaLN 的 S1，均为 7/100 完整 D9 树精确恢复；仍未通过固定50对象的完整生成目标。E2-T 的 39/100 是 D9、GT parents 下的单层采样成绩，不能换称完整树成功。七结构筛查尚未全部完成，不能给出七组排名或唯一根因。** 最新数值见下方 E2 与结构筛查表及原始证据路径。

本次工作读取已有报告、JSON、日志及服务器终点评估；没有重新训练、重跑模型或新增生成。历史复算报告证明的是当时执行的检查，本次归档文件哈希校验不等于重新验证模型能力。原始数组、JSON、训练日志及哈希映射在 Release 资产，仓库主要保留代码、报告与索引。

## 证据与计数口径

| 资产组 | 内容与定位 |
|---|---|
| `Nexus_Vertex_Historical_Evidence_20261008` | 自己的历史 Vertex 证据。归档内 `nexus_fast_track/...`；仓库可读材料在 [`vertex/nexus_fast_track`](../vertex/nexus_fast_track)。去重数组的原路径到实际保存成员映射见资产内 `ORIGIN_PATH_MAP.jsonl`。 |
| `Nexus_Teacher_Reconstruction_20261008` | 老师候选复建及分析。仓库入口为 [`teacher/downloads/Nexus_teacher_full_reconstruction`](../teacher/downloads/Nexus_teacher_full_reconstruction)。 |
| 老师原始 ZIP | 独立原始资产，SHA256 `982686feb58c207932d9311786cabb3ffd7a1397627b83c44df3d5c8ab2189e5`。原包没有原训练源码；复建候选另有代码，不得把候选代码称为老师原源码。见[候选说明](../teacher/downloads/Nexus_teacher_full_reconstruction/README.md)。 |
| `Release: 20261008 server evidence` | 最新 E0/E1/E2 与七结构筛查结果／中断快照。以下保留服务器原路径，具体资产名以根目录 Release 索引为准。 |

完整树必须从根开始、后续 parents 来自模型预测；GT-parent 单层、插值回归、训练 loss 与完整树分别报告。50对象×2种子是100次尝试，不是100个独立物体。`pooled/micro F1`由合计 TP/FP/FN计算，`macro F1`按样本等权；二者不可混用。容量中止保留在分母中。原始重复浮点顶点、显式去重顶点、量化唯一格子是不同目标；点数不符时不能伪造等点数匹配 RMSE。

## 全阶段历史

下表的相对证据路径均位于上述 Vertex 历史资产组；可点击报告为仓库副本。较新的终点评估优先于原目录中仍写着 `running`、`evaluation_pending`、`未启动` 的历史快照。

| 阶段 | 实际结果与结论边界 | 关键证据 |
|---|---|---|
| 2026-09-08 overfit20 | 1000更新完成；20/20在D7–8超过节点上限，没有最终D9顶点集。t=.9模型F1接近1，但零速度基线同样接近1。 | [最终报告](../vertex/nexus_fast_track/mini_nexus/VERTEX_OVERFIT20_RESULT_20260908.md)；原始来源 `nexus_fast_track/server_snapshots/vertex_overfit20_20260908_31548_v1/`，历史包亦保留 `vertex_staged_20260914/previous_overfit20/`。 |
| 早期 A/B 固定回归与随机噪声 | A200步通过；B2000步后随机检查5/20精确，纯噪声采样0/4。后续fixed/random-time诊断各500更新均未通过新噪声检查；早期交付写“诊断运行中”已被终点覆盖。 | [A/B报告](../vertex/nexus_fast_track/diagnostics/vertex_A_B_review_20260914_221614/RESULT.md)；`diagnostics/vertex_staged_20260914/noise_time_controls/{fixed_time,random_time}/evaluation-000500.json`。 |
| 旧 A100 B1 尝试 | 本地仅到R0/A step595，最近step500评估坐标精确0，B1尚未启动；没有该尝试最终完成证据。 | `diagnostics/vertex_a100_b1_20260914/local_status.json`，仅2026-09-14快照；与下一行独立B1区分。 |
| 独立 B1 R0/R1 | R0 500更新0/16；R1 1000更新16/16。R1一个局部响应误差仍超原门槛，因此原严格`passed=false`保留。 | [B1结果](../vertex/nexus_fast_track/diagnostics/vertex_b1_single_20260915/review_bundle/RESULTS_结果说明.md)。 |
| R1 冻结64 | 同一物体、depth9、GT parents、t=.5：64/64几何／回归通过；局部响应诊断仍失败。 | [R1 frozen64](../vertex/nexus_fast_track/diagnostics/vertex_r1_frozen64_20260915/review_bundle/RESULT_结果说明.md)。 |
| B2 随机时间与冻结64 | 累计step2000；112/112时间切片坐标正确，8/8新噪声单层采样；后续独立64噪声20步Euler为64/64。原B2全时间MSE加严门槛失败不推翻几何采样成绩。 | [B2训练终点](../vertex/nexus_fast_track/diagnostics/vertex_b2_20260915/restart1/RESULT.md)、[B2冻结64](../vertex/nexus_fast_track/diagnostics/vertex_b2_frozen64_20260915/RESULT.md)。 |
| C 单物体完整D9 | 累计3800；恢复复核4/4开发＋8/8原定终验完整树正确。714–1800训练日志缺失，旧实例是否消费终验种子不明，称恢复复核。 | [C最终恢复报告](../vertex/nexus_fast_track/diagnostics/vertex_c_recovery_20260916/RESULT.md)。 |
| D2 双物体完整D9 | 累计7400；32/32完整树，16/16同parents／实际同噪声条件切换对正确。是两个已训练对象过拟合，不是泛化。 | [D2最终总包](../vertex/nexus_fast_track/diagnostics/Vertex_D2_final3600_20260916/README.md)，含有效3600更新日志；注意目录大写`Vertex_D2`。 |
| D4 四物体 | 本地最后6816/7200；最近完整开发6800为14/16，历史最好6400为15/16，共同parents24/24。没有7200终点／64树终验证据。 | [阶段边界](../vertex/nexus_fast_track/diagnostics/point_native_xyz_gate_20260925/evidence/source_docs/STATUS_AND_EVIDENCE.md)；`diagnostics/vertex_d4_resume6800_dualgpu_20260917/run/status.json`。 |
| D9原生XYZ与老师比较 | D2历史32/32整数树正确不等于连续XYZ达老师标准。CAD50往返中48例等点数RMSE约0.000880–0.001953，另2例量化碰撞；自己的CAD50点云条件生成未运行。 | [POINT_NATIVE_XYZ_GATE](../vertex/nexus_fast_track/diagnostics/point_native_xyz_gate_20260925/POINT_NATIVE_XYZ_GATE.md)。 |
| R1 D15 双物体 | 6000更新，32/32完整树，15层各32/32。对显式去重浮点GT，RMSE约1.812e-5／1.681e-5；原始重复顶点点数门禁0/32，不能称老师原标准通过。 | [D15最终核验](../vertex/nexus_fast_track/diagnostics/r1_d15_compat_20260925/NEW_SERVER_FINAL_RECHECK_CN.md)。 |
| Phase2 D15 十物体 | 累计12000；seen16/40、unseen0/8；共同parents条件对21/45。seen的24次失败中19次首错D2–5；unseen8次均首错D2。 | [Phase2最终报告](../vertex/nexus_fast_track/diagnostics/nexus_phase2_d15_10_20260926/PHASE2_FINAL_REPORT_20260928.md)。 |
| Phase3 粗层分析／加权续训 | 冻结诊断后做2000更新粗层2:1加权续训；seen由16/40到7/40，unseen仍0/8。没有等预算未加权续训对照，不能单独归因于权重。 | [分析](../vertex/nexus_fast_track/diagnostics/phase3_coarse_analysis_20260928/PHASE3_COARSE_ANALYSIS.md)、[最终比较](../vertex/nexus_fast_track/diagnostics/phase3_coarse_weight_20260928/final_evidence_20260928/comparison/FINAL_REPORT_CN.md)。 |
| D15 overfit50 | 新增10000、累计22000；完整树0/100；逐层exact为84、9、4、4、3、1、0，D8–15均0；GT-parent单层57/400。loss下降没有形成完整生成通过。 | `diagnostics/nexus_overfit50_resume7000_20260930/result_verification.json`；[长期分析第1–4节](../vertex/nexus_fast_track/research/NEXUS_VERTEX_EXPERIMENT_LOG.md)。 |
| D9 A/B/C/D | 各新增2000、累计24000；四组×Euler/DPM完整树全0/100。Euler叶层pooled F1为A .07289030、B 0、C .04759827、D .00001780。D后450更新切双卡，多步等价门禁未通过，严格终点归因受限。 | [ABCD最终报告](../vertex/nexus_fast_track/diagnostics/nexus_d9_cd_20261001/FINAL_RESULTS_CN.md)，原数组在`verified_delivery/`。 |
| D9 E/F/G | E/F/G各新增2000，A复用；两采样器仍全0/100。Euler叶F1为A .07289030、E .00003683、F .06393773、G .00007835。F局部条件响应改善未变成总体收益；高LR在本轮两种结构下均更差。 | [AEFG终点](../vertex/nexus_fast_track/diagnostics/nexus_d9_efg_20261002/collected_final/README_CN.md)、[探针复核](../vertex/nexus_fast_track/diagnostics/nexus_d9_efg_20261002/audit/astra_final_probe_review.md)。 |
| A24000 GT-parent诊断 | D9 pooled F1由自由展开.07289到GT-parent .30356，D9exact由0到9/100；九层独立诊断全对仍0/100。D9 TP/FP/FN=32304/74886/73340，上游传播与本层失配同时存在。 | [保存轨迹分析](../vertex/nexus_fast_track/diagnostics/nexus_d9_A_gtparents_20261002/audit/TRAJECTORY_REVIEW.md)；`collected_final/summary.json`。 |
| 插值与实际轨迹 | D9 t=.5宏F1为插值.63867／轨迹.47324；FP/FN分别12194/62820与28057/80436。t=.95零速度基线在D6–9也各100/100，不能将晚期插值精确当作生成成功。 | **完整证据在** `nexus_fast_track/tmp/vertex_interp_review_20261003/evidence/summary.json`，同目录上层`cpu_verification_20261003.json`、`review_statistics.json`；不只在同名diagnostics目录。 |
| E0/E1 | E0真实2400次forward重现旧数组；E1 7200次完成，恢复正确条件完全复原。t=0错配条件使D6/7/8/9宏F1从.6841/.5230/.4627/.4366降为.5813/.4463/.3814/.3715；支持条件影响输出，不能证明充分利用。 | [启动与门禁说明](../vertex/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/STARTED.md)；`startup_evidence/run/{e0_global_gate,e1_global_gate,e1_summary}.json`，完整数组属最新服务器证据。 |

## E2：2000更新已完成，保持单层范围

2026-10-08直接读取服务器三支`status.json`，均为`training_complete=true`、`evaluation_complete=true`、`e2_update=2000`，累计step26000。共同起点是A24000，原结构、同50训练对象，只训练D9与GT parents；每支2000更新、每次8个micro、每对象320次曝光。[完整协议](../vertex/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/EXECUTION_CARD.md)。

- C：zero noise、t=0；目标y，是确定性条件映射。
- N：Gaussian noise、t=0；目标y−epsilon。
- T：同一Gaussian noise、预存uniform时间；目标y−epsilon。

共同评价使用与训练分离的保存噪声、GT parents及20步Euler。以下`exact/100`均为**单层**，不是完整树；共同起点与较早GT-parent诊断采用的探针不同，不把11/100与旧9/100混为同一组重放。

| 分支 | 新增更新 | 共同Euler单层exact/100 | 宏F1 | pooled F1 | TP / FP / FN | 两seed均exact的UID/50 |
|---|---:|---:|---:|---:|---|---:|
| C/N/T共同起点 | 0 | 11 | .487355 | .306321 | 32534 / 74240 / 73110 | 5 |
| C | 1000 | 1 | .519171 | .406945 | 43293 / 63834 / 62351 | 0 |
| C | 2000 | 0 | .416762 | .330406 | 40552 / 99272 / 65092 | 0 |
| N | 1000 | 12 | .735910 | .596417 | 61991 / 40243 / 43653 | 5 |
| N | 2000 | 12 | .770297 | .654219 | 66558 / 31271 / 39086 | 5 |
| T | 1000 | 30 | .959732 | .922338 | 97220 / 7948 / 8424 | 12 |
| T | 2000 | 39 | .994201 | .990016 | 104515 / 979 / 1129 | 17 |

**逐行原始来源：** `Release: 20261008 server evidence`；服务器根目录
`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_e0_e2_capability_20261003/e2/`，每行对应`{C,N,T}/evaluation/update-{000000,001000,002000}/summary.json`的`final_euler`与`final_euler_uid_pairs`。三支完成状态分别在`{C,N,T}/status.json`。

任务内回归应另读，三行的输入难度不同，不能用其高分直接排列生成能力：

| 任务内评价 | 起点exact | 1000更新exact | 2000更新exact | 2000更新宏F1 / pooled F1 | 2000更新FP / FN |
|---|---:|---:|---:|---|---|
| C，确定性zero-input/t0 | 7/50 | 14/50 | 15/50 | .994477 / .993215 | 302 / 1126（原100槽合计，重复两次） |
| N，Gaussian/t0 | 9/100 | 28/100 | 38/100 | .977917 / .960067 | 529 / 7625 |
| T，Gaussian/random-t | 26/100 | 49/100 | 58/100 | .998024 / .996694 | 285 / 413 |

来源为同九份`summary.json`的`task`、`task_uid_pairs`。C只有50个独立输入；原JSON的14/28/30个exact是100个重复seed槽，已在此换回7/14/15个UID，不能称100次独立测试。T任务内58/100包含GT插值信息，不等于其共同纯噪声Euler的39/100。

结论：现有结构在该D9受控任务中可以明显改善；T在本轮共同采样探针上优于C/N，C的clean映射改善没有迁移到共同带噪采样。这支持继续区分条件映射、噪声鲁棒性和跨时间去噪能力；不支持“网络完全无条件信息”“必须改结构”或“已找出唯一根因”。三支都未在这里验收完整树或未见对象泛化。

## 七结构筛查：两支终点完成，其余中断／未启动

各支计划独立从A24000暖启动、10000更新到累计34000；S0原结构，S1仅final time AdaLN，其余配置与迁移见[结构卡](../vertex/nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006/ARCHITECTURE_CARD.md)。这是暖启动筛查，不是从零训练能力排名。

| 分支 | 已核实状态 | 完整D9树exact/100 | D9 pooled F1 | D9 TP / FP / FN |
|---|---|---:|---:|---|
| S0 原结构 | 新增10000、累计34000；补评估完成，100/100尝试均有完整轨迹，无容量中止 | 7 | .521241 | 52827 / 44226 / 52817 |
| S1 final time AdaLN | 新增10000、累计34000；补评估完成，100/100尝试均有完整轨迹，无容量中止 | 7 | .511243 | 52451 / 47095 / 53193 |
| S2 关闭CA QK norm | 历史日志到6098；可恢复checkpoint为6000新增更新／累计30000；没有10000终点评估 | 未完成 | — | — |
| S3 learned queries | 历史日志到5688；可恢复checkpoint为5600新增更新／累计29600；没有10000终点评估 | 未完成 | — | — |
| S4/S5/S6 | 正式分支目录未出现，本轮尚未启动；smoke测试不算正式训练 | 未执行 | — | — |

**原始来源：** `Release: 20261008 server evidence`；服务器根目录
`/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007b/runs/`。
S0/S1为`{S0,S1}/evaluation/summary.json`中的`full_trees_exact`、`per_depth[8]`；S2/S3为各自`train.jsonl`、`status.json`、`checkpoint_identity.json`。S0/S1最终权重身份引用前一恢复目录`/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs/`，不可用它们的旧评估失败记录覆盖后续成功补评估。

归档时主任务另行只读确认当前实例`eusk5t3tad7qs-0`只有一张不同UUID的空闲GPU，未发现本任务进程；旧队列`running`与S2/S3的`state=training`是失效快照，**不能写成仍在训练**。本轮没有恢复、重启或追加预算。对应现场核验记录归入同一服务器证据资产。

S0在已用配对种子上由A24000的0/100到7/100，说明原结构继续适配后出现完整恢复；不能据此宣布50对象任务通过或唯一归因于训练时长。S1同为7/100且pooled F1略低，本轮没有显示总体收益；其余五组未完成，不作七组最终排名。

## 老师prior与自己octree的边界

| 项目 | 老师最终点模型候选 | 自己的Vertex |
|---|---|---|
| 来源 | 从原ZIP权重形状、配置和输出逆向复建；候选有`models.py`、`objectives.py`、`replay.py`，不是原训练源码 | 独立NEXUS式实现，冻结代码随各阶段保存 |
| 条件／表示 | 缓存2048维文本；连续XYZ槽位、count head、coordinate prior | 8192×XYZ＋normal，经VecSet；每parent八维octree occupancy |
| 成功证据 | 原prior保留的候选重放，两种seed共100/100达到既有点门槛；来源为[POINT_NATIVE_XYZ_GATE](../vertex/nexus_fast_track/diagnostics/point_native_xyz_gate_20260925/POINT_NATIVE_XYZ_GATE.md) | 单物体／双物体小规模已通过；50对象最新完整树S0/S1各7/100 |
| 解释限制 | 原报告称高精度主要来自学到的坐标prior，residual scale约1.219824e-5；不是纯去噪主干单独成功证明 | 无老师坐标prior；不能把老师CAD50、文本条件或重复顶点标准静默当作自己的任务成绩 |

老师候选的具体结果、原包身份与续训缺口见[README](../teacher/downloads/Nexus_teacher_full_reconstruction/README.md)和[REPORT](../teacher/downloads/Nexus_teacher_full_reconstruction/REPORT.md)。`point_prior_candidate/candidate.pt`的optimizer只覆盖prior六个张量与scale，缺RNG；不是整个18层去噪器的完整可续训状态。材料中已有拓扑结果仅作为老师包原貌保存，不并入本页Vertex成果，也不触发自己的VAE重新训练。

## 当前结论与版本优先级

1. 已证明的是若干固定对象、固定协议下的能力与失败边界；没有证明未见对象泛化或完整论文复现成功。早期单物体／双物体通过与后续50对象困难并不矛盾。
2. 条件会影响输出；GT-parent、E1条件置换与恢复、E2-T提升都不能被简化为“条件已完全学好”。上游树错误与本层去噪错误同时存在，尚无唯一根因。
3. A–G的DPM没有挽救该固定模型／预算下的完整恢复；这不证明DPM普遍无效。高t插值含GT、bit accuracy受空格比例影响、loss下降不能替代整数集合与完整树验收。
4. [长期研究入口](../vertex/nexus_fast_track/research/NEXUS_VERTEX_EXPERIMENT_LOG.md)最后更新2026-10-01，其中E/F/G“未实施”与E2启动快照已被本页列出的终点证据覆盖；原文保留以追溯当时决策。
5. 最新[结构依据与候选主文](../vertex/nexus_fast_track/research/vertex_structure_options_20261005/NEXUS_VERTEX_STRUCTURE_OPTIONS.md)区分论文确定项、未定项、外部源码、适配和老师候选。主文SHA256为`04e9c7d93e6a78f1fd0196fbd61d466408f73cf621ce974d218c480d6d43d621`；V1归档／V2以及`document_validation_v2.json`是旧版证据，不能把旧V2的校验当作最新主文校验。
6. 大Vertex checkpoint／Adam张量不在轻量结果包内；身份与源路径不能替代权重本体。历史源码、重复快照与原数组的去重位置以Release清单和映射为准；恢复训练前还需重新检查现存文件及完整状态。本次只归档，没有作恢复授权或执行。
