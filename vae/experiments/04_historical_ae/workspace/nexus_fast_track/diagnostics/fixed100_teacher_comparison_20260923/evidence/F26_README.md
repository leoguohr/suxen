# 固定100条：实际Face池外难负例补充对照

共同起点为B epoch500 / update12500，SHA256 `dc28a6e72a8e30a84d4536cbd09168d48635d66c5a924a3f52ef907c36fc0038`。
Control是已完成的`math00_overfit100_B_continue100ep_20260915`；本轮只新增难负例分支，不覆盖epoch600主线，也不重跑Control。

## 唯一训练干预

在固定父checkpoint的μ forward上，枚举预测Edge图中的完整Face候选。筛选logit>0、GT负类、且不在旧训练pool中的三元组。每条mesh按logit降序选择，至多其GT Face数量；同logit按规范化候选ID升序打破并列。与旧pool/GT去重。旧pool原内容与顺序保留，新负例追加到末尾，挖掘完成后固定，不动态刷新。

新负例并入原Face fully-diff Soft4，不新加难例loss、不改变公式、chunk设置或归约算法。Face监督集合与相应分母发生变化是本实验的干预。Edge监督不变。

四组Adam、父shuffle及其他RNG、参与计数完整恢复。E/μ LR=3e-6，D/heads LR=3e-5，四条完整mesh依次microbatch累计后按1/4平均、global clip=1、wd=0。μ，KL=0，logvar冻结；math00、数据、评分尺度与阈值均不变。逐epoch断言与已完成Control的100条遍历顺序完全一致。

新增epoch501—600 / update12501—15000，共2500次实际更新，预算结束停止。epoch500、525、550、575、600全量实际验收；沿用原30秒每mesh Face枚举预算，未完成明确标记，不能当作成功。训练前核验100条实际计数及**旧pool诊断loss**与共同父状态一致；新pool训练loss不要求等于旧pool。

## 评价口径

`parts`和`face_training_pool`指本分支新增后的训练pool；`old_pool_parts`和`old_face_training_pool`是同一旧pool的诊断。实际Face始终从当次预测Edge图枚举。`actual_fp_outside_training_pool`明确沿用**旧固定pool**作为统计参考，以便跨分支可比。最终比较实际Edge/Face FP/FN、严格成功UID与GT Face候选覆盖，不用不同训练pool的loss直接排名。

## 文件索引

- `mining_complete.json`、`mining.jsonl`：每条cap、新增数、全部合格池外FP数、来源checkpoint、数据哈希、固定起点复核。
- `mined/*.npz`：每条实际新增的候选ID、父checkpoint logit与负标签；会纳入评估包。
- `augmented_pools/*.npz`：服务器上的完整固定新pool，含旧前缀及新负例。
- `run/resume_verification.json`、`baseline_comparison.json`：模型、Adam、RNG与起点验收。
- `run/updates.jsonl`：完整2500次更新，逐mesh参与、四组梯度、实际更新及裁剪。
- `run/eval-epoch*.jsonl`：逐mesh完整实际验收、新pool与旧pool诊断。
- `completion_verification.json`：预算、父shuffle重放、参与计数、四组非零更新及checkpoint哈希。
- `control_comparison.json/csv/png`、`paired_per_mesh.csv`、`control_evaluations/`：与已有Control的同进度结构比较及原始验收记录。
- `runtime.py`、`mine.py`、`train_hardneg.py`、`source_archive/`、`review_runtime/`等：实际代码与运行时替换。

结束后自动生成`evaluation_package.zip`。大checkpoint、原始mesh及完整新旧pool数组保留服务器，不包含在评估包；新增负例ID/logit NPZ包含在包中。`job_status.json`记录挖掘、训练、核验、比较、打包阶段；发生错误即停止，不自动重试或延长。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_face_hardneg_20260915`。启动入口`bash run.sh`，启动后勿重复执行。
