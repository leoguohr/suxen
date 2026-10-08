# 第二轮固定Face难负例：epoch800 → 900 配对对照

共同父状态：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_low_lr_continue100ep_20260916/run/checkpoint-update20000.pt`。
SHA256：`38087aeefbc12444cb51ba35dbe3715cc0f785c30cd6ed53e093a0ee7b929f80`。
Control复用已完成的`math00_overfit100_low_lr_epoch800_900_20260917`，不重跑。当前epoch900模型保留，不覆盖。两支预算不能相加。

## 唯一干预

仅在父checkpoint、μ路径上完整枚举预测Edge图的Face候选，挑选logit>0、GT负类、不在整个epoch800扩充pool（含第一批难负例）的候选。每条新增上限为GT Face数，按logit降序、规范化ID升序打破并列，全量枚举后取top-K。验证新增ID无重复且不与GT或旧pool重叠，原pool全部张量与第一批负例作为未变前缀保留。第二批固定后不刷新、不另加loss。

挖掘使用原round1的枚举/评分/排序核心，并逐mesh核验父checkpoint硬计数与共同旧pool loss；挖掘0次optimizer更新且模型参数哈希不变。训练随后重新从父checkpoint完整恢复四组Adam、LR、全部RNG与参与计数，挖掘不推进训练RNG。

新增100 epochs、2500次更新（20001—22500），逐epoch断言与Control的801—900遍历一致。E/μ=9e-7，D/heads=9e-6，不warmup、不改clip。固定100条、有效batch4（完整mesh逐条微批，loss各除4，累积后更新）、math00、fully-diff Edge/Face Soft4、μ、KL=0、logvar冻结、wd=0、global clip=1均不变。

## 评价口径

在epoch800、825、850、875、900完整实际验收，Face始终由该次预测Edge图枚举。每10 epochs及评价点保存完整checkpoint，到预算停止。

- 新分支`parts`是第二轮扩充pool训练目标，不能直接与Control训练loss排名。
- 新分支`old_pool_parts`及`old_face_training_pool`是共同epoch800 pool（含第一批负例）；对应Control的`parts`及`face_training_pool`，这是可比诊断。
- 为与历史实际验收逐项对应，`face.actual_fp_outside_training_pool`仍参考最初base pool，字段`actual_fp_outside_pool_reference`明确标记；不能将其误称为第二轮新pool之外FP。
- `first_round_negative_diagnostic`与`first_round_logits/`保留第一批固定负例的检查；`mined_negative_diagnostic`与`hardneg_logits/`对应第二批。
- 判据为实际Edge/Face TP/FP/FN、GT Face候选覆盖及同checkpoint严格成功UID。成功数始终以固定100条报告，不筛选子集冒充全部通过。

## 结果材料

`mining.jsonl`、`mining_complete.json`及`mined/*.npz`记录完整第二轮挖掘来源、数量、ID/logit及hash；`parent_mining_complete.json`和`first_round_mined/`保留第一轮证据。

`run/updates.jsonl`、恢复核验、基线比较、逐mesh完整验收、固定负例轨迹、完成核验及代码均保留。`control_comparison.json/csv/png`与`paired_per_mesh.csv`比较共同pool诊断loss和实际重建；复制Control原始验收到`control_evaluations/`。最终自动打包`evaluation_package.zip`并附SHA256SUMS。

大checkpoint、原始mesh及完整pool数组留服务器，路径和hash见manifest及mining记录。完整执行代码与运行时替换源码包含在评估包中。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_face_hardneg_round2_20260917`。
入口`bash run.sh`。不要重复启动。任何核验失败停止，不自动改参、重挖或延长预算。
