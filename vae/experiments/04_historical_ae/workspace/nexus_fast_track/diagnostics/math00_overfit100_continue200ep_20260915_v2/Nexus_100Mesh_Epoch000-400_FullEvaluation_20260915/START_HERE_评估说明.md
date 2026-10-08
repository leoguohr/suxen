# 固定100条：400 epochs完整评估包



两段预算均已完成。第一段5000更新，第二段完整恢复后新增5000更新；累计10000更新、400epochs，每条mesh直接参与400次。

本包基于保存日志和代码做只读汇总，没有新增optimizer update，也没有重新运行模型。



最终严格成功：5/100。成功UID：nexus_2k_001150, nexus_2k_000479, nexus_2k_000341, nexus_2k_001045, nexus_2k_000618。

末尾100条中Face枚举未完成：0条。



|阶段|Edge FP/FN|实际Face FP/FN|严格成功|

|---|---|---|---|

|epoch200|1620842/679|2989041/7413|2/100|

|epoch400|869893/118|752120/8042|5/100|



末尾漏面分解：缺边未入候选226，候选存在但判负7816。

末尾实际Face FP中训练pool之外的数量：738643。



## 先看这些文件

- full_training_evaluation.png：累计10000步的同口径完整评估趋势。

- evaluation_trend_epoch000-400.csv：11轮去重后的全量指标，Precision/Recall为micro汇总。

- per_mesh_final_epoch400.csv：最终100条逐mesh错误、候选覆盖、margin与严格成功。

- per_mesh_epoch200_vs400.csv：同UID前后对照；change=末尾减起点，负值表示计数减少。

- per_mesh_all_1100_evaluations.csv：全部去重后的逐mesh评估。原始epoch200加载复核仍保留在phase2中。

- per_mesh_training_40000_participations.csv：每次mesh直接参与训练的记录。

- phase1/：第一段完整评估包，原始5000步更新日志在evidence/run/updates.jsonl。

- phase2/：第二段原始5000步日志、5轮完整评估、恢复核验、完成记录、代码/配置/候选清单。

- SUMMARY.json、verification.json、SHA256SUMS.txt：汇总、独立一致性检查、包内哈希。



## 必须保留的解释边界

训练固定100条，μ路径，KL=0，logvar冻结；不等于已恢复sampling/KL的VAE阶段。有效batch4是4个完整mesh微批loss/4累积后统一更新，不是每条训练10000次。

完整验收在同checkpoint暂停更新后依次前向100条，不是一次packed100。Face从当次预测Edge图枚举。Face Soft4在固定训练pool上计算，不能当作实际候选全集loss。

原epoch0有未完整枚举，FP/F1保留null；统计表空格不是0。严格成功必须同checkpoint所有Edge/实际Face FP=FN=0；不同检查点成功身份不得合并计算。

区分已恢复的GT、仍然大量存在的误报与严格成功；loss下降和高Recall不能替代100/100验收。

包内不含大checkpoint、mesh/pool二进制或未记录的逐候选logit/embedding；它是结果审阅包，不是离线推理部署包。父文件路径/已有SHA见phase1/checkpoint_inventory.json和manifest；第二段验收checkpoint路径/SHA见phase2/run/eval-summary-*.json。

最终checkpoint：/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_continue200ep_20260915_v2/run/checkpoint-update10000.pt

SHA256：0804e7dabccd5dfd027215be4a2a8c98281aea8ffd148090913b658209371d14

源代码包括原始快照、实际运行时替换及恢复入口；args.precision历史字符串不能替代manifest.backend与实际math00实现。

当前预算已经停止。没有自动追加训练、调LR、改Face pool或新建其他实验。
