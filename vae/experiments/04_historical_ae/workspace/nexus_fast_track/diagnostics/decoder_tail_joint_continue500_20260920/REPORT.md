# Joint分支新增500次全100条联合更新

已完成并按预算停止。本轮是从Joint step500 / Tail与Face Adam step1000续接的新500次更新，末尾Joint累计1000次，三组Adam均到1500。不是500个epoch，也不是原主线mini-batch数。
同一个原100条集合，每次遍历全部100条、按mesh等权累积联合梯度后统一clip与Adam更新。目标mean100(Edge fully-diff Soft4 + Face fully-diff Soft4)，各项Soft4内部固定除4；没有额外诊断系数。
仅Decoder末块15、最终LayerNorm、Edge head、Face head可训练；末端LR=1e-5、两head各1e-4，三组Adam连同CPU/CUDA RNG从同一父checkpoint整体恢复。没有重置、重新warmup、刷新两轮Face pool、增加其他目标或解冻上游。math00、mu路径、KL=0、clip=1、wd=0均保持。
更新前全100条真实网络与缓存路径输出、联合loss/梯度逐位一致；重复完整梯度逐位一致。末尾重新加载endpoint并由真实输入复核全部100条；Face由每个检查点的预测Edge图重新枚举。

| 本轮新增更新 | Joint累计 | Edge成功/100 | 联合成功/100 | Edge FP/FN | 实际Face FP/FN | 缺失GT Face候选 |
|---:|---:|---:|---:|---|---|---:|
| 0 | 500 | 71 | 71 | 117054/1 | 7537/336 | 1 |
| 100 | 600 | 71 | 71 | 113762/1 | 7225/312 | 1 |
| 200 | 700 | 72 | 72 | 110226/1 | 6851/296 | 1 |
| 300 | 800 | 71 | 71 | 112917/1 | 7990/349 | 1 |
| 400 | 900 | 71 | 71 | 108805/1 | 7159/274 | 1 |
| 500 | 1000 | 72 | 72 | 106725/1 | 6873/273 | 1 |

末尾联合严格成功72/100；起点71条保留71、丢失0、新增1。后期300/400/500三次共同成功71条。
新增成功UID：nexus_2k_000520。丢失UID：无。

| 预先固定组 | 条数 | 起点Edge FP/FN | 末尾Edge FP/FN | 起点Face FP/FN | 末尾Face FP/FN | 末尾联合成功 |
|---|---:|---|---|---|---|---:|
| all100 | 100 | 117054/1 | 106725/1 | 7537/336 | 6873/273 | 72 |
| initial_failed29 | 29 | 117054/1 | 106725/1 | 7537/336 | 6873/273 | 1 |
| N_gt1500 | 33 | 113846/1 | 104399/1 | 7410/332 | 6761/271 | 8 |
| initial_success71 | 71 | 0/0 | 0/0 | 0/0 | 0/0 | 71 |

原失败29条：Edge FP减少29条、相等0条、增加0条；其中末尾Edge严格成功1条、联合严格成功1条。

| 本轮区间 | Edge FP下降比例 | Face FP下降比例 | Face FN变化 | 联合成功变化 |
|---|---:|---:|---|---|
| 0→500 | 8.8241% | 8.8099% | 336→273 | 71→72 |
| 300→500 | 5.4837% | 13.9800% | 349→273 | 71→72 |
| 400→500 | 1.9117% | 3.9950% | 274→273 | 71→72 |

结果按成功UID和实际FP/FN判读。有限预算没有突破不能证明容量不可能；loss下降也不能替代结构清错或严格成功。此次无自动追加，旧74条模型仍完整保留，不能把两个模型的成功UID拼成一份模型结果。
continuous_joint_history.csv提供此前500次与本次500次的连续轨迹，前后起点不同，不把前一段当成本轮新的配对Control。旧74条模型来自另一条分阶段路径，不按本轮等预算对照排名。

材料索引：
- run/updates.jsonl：500条完整更新、三组LR、四模块实际位移、梯度与clip；edge_trace.jsonl：501×100条Edge计数、margin及Edge/Face训练pool loss。
- run/actual-new*.json与actual_evaluations.jsonl：6次、600条完整Edge与实际Face计数，pool内计数、候选覆盖与margin；final_real_network.json：末尾真实输入复核。
- actual_per_mesh.csv、per_mesh_progress.csv、initial_failed29_progress.csv、actual_group_trend.csv、late_changes.csv、retention.json：逐mesh与组别趋势、成功保留/丢失/新增和末段变化。
- restore_verification.json、cache_gradient_verification.json、benchmark.json、run/verification.json、independent_audit.json：状态恢复、真实/缓存梯度、冻结边界与84,669,234个末尾Edge logits的独立计数复算。
- runtime.py、joint_core.py、prior_core.py、head_core.py、face_core.py、evaluate.py、effective_code、runtime_dependencies：实际公式、评分与运行时替换；continuation_changes.diff记录相对父入口的续训改动。prior_core的历史训练入口本轮未调用。
- Review：日志、逐mesh结果、实际源码、初末尾末端/两个head权重及Adam/RNG；FullEvidence额外包含缓存输入、末尾hidden与评分embedding、全部Edge logits、中间checkpoint、固定Face pool。
- 巨大全模型未放ZIP；EXCLUDED_FILES.json列出服务器路径、已知SHA与排除范围。Face全候选逐项logits未单独保存，包含完整实际计数与评分表示；Face训练pool loss不等于全实际候选loss。
- 本轮始终是mu训练与验收，没有posterior噪声成功率结论。

父完整模型：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_joint_fixed100_20260919/run/model-joint500-tail1000-face1000-inference.pt`；SHA `5a36ab88ee147ef160945cfcec11d1d5cb98026e20c536ce602f6303f1448f7e`。
父Adam checkpoint：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_joint_fixed100_20260919/run/checkpoint-new0500-tail1000.pt`；SHA `4e328fcff09d85d001af29548cfd559bf39306054b8c05825249cc95e2ad3b68`。
新完整模型：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_joint_continue500_20260920/run/model-joint1000-tail1500-face1500-inference.pt`；SHA `d734627ad18b966f3110489240cbc12285f27c58cd0099260884d11cb9659cce`。
