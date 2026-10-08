# 最后两块Decoder联合训练：500次全100条更新对照

新诊断已完成有效轨迹500次更新并按预算停止。共同起点为Joint step500 / 原三组Adam step1000；复用已完成的单末块Control，未重跑或覆盖Control。
唯一训练范围变化：原末块15、最终LayerNorm、Edge/Face head之外，再开放原Decoder第14块。所有权重来自同一父模型；原三组Adam逐项恢复，新第14块单独空状态起步。末尾原三组Adam为1500、新第14块为500。预算不能与Control相加为同一模型训练量。
中途服务器实例重启，首轮日志记录179次更新，最近完整checkpoint在100步。恢复100步全部状态后重放101—179，日志中的前向指标、梯度范数、clip系数和实际位移范数与中断记录逐项完全相同；然后续到500。有效轨迹500步，另有79次恢复重放，实际累计执行579次optimizer step。未增加终点训练预算。中断现场保存在interrupted_attempt0，恢复入口为resume_run.py。
每次更新完整累积原100条，目标mean100(Edge fully-diff Soft4 + Face fully-diff Soft4)。两个block及最终LayerNorm LR=1e-5，两个head各1e-4；mu、KL=0、两轮固定Face pool、32维评分、threshold=0、math00、clip=1、wd=0均不变。
缓存重新导出于第14块输入之前，第14/15块均留在可微路径。起点100条真实前向、loss与所有开放参数梯度逐位复现缓存路径；测量耗时/显存后开始更新。末尾安装相同权重回真实完整网络复核100条，实际Face从各检查点预测Edge图重新枚举。

| 新增更新 | Control Edge/联合 | 两块 Edge/联合 | Control Edge FP/FN | 两块Edge FP/FN | Control实际Face FP/FN | 两块实际Face FP/FN |
|---:|---|---|---|---|---|---|
| 0 | 71/71 | 71/71 | 117054/1 | 117054/1 | 7537/336 | 7537/336 |
| 100 | 71/71 | 71/70 | 113762/1 | 116146/1 | 7225/312 | 8437/408 |
| 200 | 72/72 | 71/71 | 110226/1 | 113277/1 | 6851/296 | 7230/299 |
| 300 | 71/71 | 72/72 | 112917/1 | 110088/1 | 7990/349 | 6913/281 |
| 400 | 71/71 | 72/72 | 108805/1 | 106719/1 | 7159/274 | 6641/252 |
| 500 | 72/72 | 72/72 | 106725/1 | 102302/1 | 6873/273 | 5635/338 |

两块末尾联合72/100，原71条保留71、丢失0、新增1。后期300/400/500三个检查点共同成功72条。
新增UID：nexus_2k_000520。丢失UID：无。

| 预先固定组 | 条数 | Control末尾Edge FP/FN | 两块末尾Edge FP/FN | Control末尾Face FP/FN | 两块末尾Face FP/FN | Control/两块联合成功 |
|---|---:|---|---|---|---|---|
| all100 | 100 | 106725/1 | 102302/1 | 6873/273 | 5635/338 | 72/72 |
| initial_failed29 | 29 | 106725/1 | 102302/1 | 6873/273 | 5635/338 | 1/1 |
| N_gt1500 | 33 | 104399/1 | 100244/1 | 6761/271 | 5546/336 | 8/8 |
| initial_success71 | 71 | 0/0 | 0/0 | 0/0 | 0/0 | 71/71 |

按同预算实际FP/FN、成功UID保持及困难大mesh清错判读，不能只按loss。开放第14块同时引入其新Adam状态、改变总梯度与clip；这是训练范围策略对照，不是纯容量证明或梯度冲突证明。旧72/74条模型与共同父状态完整保留。

材料索引：

- run/updates.jsonl：500条完整更新，四组LR、五模块位移/梯度、全局clip；edge_trace.jsonl：501×100条Edge计数与两项训练loss。
- run/actual-new*.json、actual_evaluations.jsonl：600条实际Edge/Face计数、margin、候选覆盖与pool内诊断；final_real_network.json：末尾真实网络复核。
- paired_comparison.csv、paired_per_mesh.csv、paired_groups.csv：对齐Control的检查点、逐UID、预先固定组别；retention.json保存原71条的保留/丢失/新增。
- restore_verification.json、cache_gradient_verification.json、freeze_contract.json、benchmark.json、run/verification.json、independent_audit.json：缓存位置、梯度、成本、Adam状态、冻结边界、84,669,234条末尾Edge logit计数复算。
- run.py、prior_core.py、joint_core.py、head_core.py、face_core.py、runtime.py、evaluate.py、effective_code、runtime_dependencies：实际执行公式与数值路径。prior_core仅score/scoring_module/metrics/summary/norm被调用，历史训练入口未使用；run.py.diff/prior_core.py.diff明确本轮改变。
- Review含完整日志、评价、源码和初末尾开放模块checkpoint/Adam/RNG；FullEvidence另含第14块输入缓存、末尾hidden/embedding/全部Edge logits、固定Face pools和中间checkpoint。
- 巨大全模型不放ZIP，路径/SHA见EXCLUDED_FILES.json；Face全候选逐项logits没有另存，保留完整实际计数与评分表示。Face固定pool loss不能当作实际候选全集loss。

共同父完整模型：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_joint_fixed100_20260919/run/model-joint500-tail1000-face1000-inference.pt`，SHA `5a36ab88ee147ef160945cfcec11d1d5cb98026e20c536ce602f6303f1448f7e`。
共同父Adam：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_joint_fixed100_20260919/run/checkpoint-new0500-tail1000.pt`，SHA `4e328fcff09d85d001af29548cfd559bf39306054b8c05825249cc95e2ad3b68`。
新模型：`/guohaoran/nexus_fast_track/diagnostics/decoder_last2_joint_fixed100_20260920/run/model-last2-joint1000-tail1500-block14new500-inference.pt`，SHA `82cf33fbc0151d72332218c5e447f7bf84e8934947e4c70148205c6a308db884`。
