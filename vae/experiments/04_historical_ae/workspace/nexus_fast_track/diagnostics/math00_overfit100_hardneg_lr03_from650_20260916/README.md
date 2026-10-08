# 固定100条：难负例分支epoch650的单档低LR对照

共同起点：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_hardneg_continue100ep_20260916/run/checkpoint-update16250.pt`。
SHA256：`7c4d4c7539eb4d4a0e483abd9457a44ed7a6700a7d3bff352c6e97fc8723f0e7`。

Control为已完成epoch651—700的原LR轨迹，不重跑。新分支完整恢复epoch650权重、四组Adam、所有RNG及参与计数，随后仅将E/μ LR设置为9e-7，Decoder/Edge/Face heads LR设置为9e-6。每epoch逐项断言与Control已有遍历一致。新增50 epochs、1250次更新，到epoch700/update17500停止；两支步数不能相加。

固定100条、现有扩充pool（187932个已挖掘负例）、batch4、μ、KL=0、logvar冻结、fully-diff Soft4、math00、clip=1、wd=0均不变。不重新挖掘、不重新warmup、不重置Adam，不覆盖原epoch700。

## 起点与验收

epoch650、675、700完整实际评价。起点须复现父状态的全部新/旧pool loss、实际计数和margin。实际Face仍从当前预测Edge图枚举，原30秒限制不变；未完成明确标记，不能算严格成功。

额外诊断：先对Control三个已保存checkpoint做只读前向，0次optimizer更新；核验loss和训练pool计数与原日志一致。新分支在相同检查点保存同一固定难负例的ID和完整训练pool评分调用中的logit。`negative_to_positive`统计两个相邻检查点之间从正确负类变成FP的候选，`positive_to_negative`统计反方向；这些是**检查点观测到的翻转**，不等于逐训练步的全部事件，也不等于实际Face候选图上的FP数。

## 材料索引

- `run/updates.jsonl`：完整1250次更新、UID、LR、梯度、clip和实际参数位移。
- `run/resume_verification.json`、`baseline_comparison.json`：父状态、仅LR差异、pool不变、起点完整复现。
- `run/eval-epoch*.jsonl`：完整实际重建、旧pool诊断loss与难负例汇总。
- `hardneg_logits/`、`control_hardneg/`：两支三个检查点同一负例的逐ID logit；包含在最终包。
- `control_forward_verification.json`：只读Control重放的权重不变、0更新、loss与pool计数核验。
- `completion_verification.json`：50轮shuffle、1250实际更新、参与计数、四组更新及最终checkpoint哈希。
- `comparison.json`、`matched_comparison.csv/png`、`per_mesh_comparison.csv`、`mined_sign_changes.csv`：同进度实际FP/FN、严格成功身份、负例反复情况。
- `control_evaluations/`：复用的原Control三个实际验收原始记录。

结束自动生成`evaluation_package.zip`，包括执行代码、日志、核验、逐候选难负例诊断；大checkpoint和完整mesh/pool数组保留服务器。状态见`job_status.json`，错误停止，不自动改参或续训。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_hardneg_lr03_from650_20260916`；入口`bash run.sh`。启动后勿重复运行。
