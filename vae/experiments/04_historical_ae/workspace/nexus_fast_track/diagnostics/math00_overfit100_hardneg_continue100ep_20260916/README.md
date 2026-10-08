# 固定100条：固定难负例pool续训epoch601—700

从已完成难负例分支epoch600/update15000续训，新增100 epochs/2500次实际更新，到epoch700/update17500停止，不自动延长。

起点：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_face_hardneg_20260915/run/checkpoint-update15000.pt`。
SHA256：`cf3f780ec9df15c60d49949891d8e5a6ff964940e63a837465e4744c9cd93082`。

保留完整四组Adam、所有RNG、每条参与计数。E/shared+μ LR=3e-6，Decoder/Edge/Face LR=3e-5，直接继承、不再乘0.3、不重新warmup。固定100条、每microbatch一条完整mesh，累计4条等权loss后clip=1及Adam更新。μ、KL=0、logvar冻结、wd=0、math00、fully-diff Soft4、评分、阈值与归约不变。

本轮不运行挖掘。`augmented_pools`、`mining_complete.json`、`mined`直接引用上一分支；逐文件SHA256必须匹配父checkpoint中记录的187932个新增负例所构成的固定pool。原Control与难负例epoch600均保留。

## 验收与日志

- epoch600起点先复现100条的全部新/旧pool loss、实际重建计数、margin和覆盖；不一致则停止，不做更新。
- 全量评价epoch600、625、650、675、700；每10 epochs及验收点保存权重、四组Adam和RNG。
- `run/updates.jsonl`记录全部2500次更新；`run/epoch-order-*.json`记录从父shuffle继续推进的100轮顺序。
- `run/eval-epoch*.jsonl`记录逐mesh FP/FN、候选覆盖与严格成功；Face从当前预测Edge图重新枚举。原30秒枚举预算不变，未完成明确标记，不能计为严格成功。
- `parts`是本次沿用的扩充pool loss；`old_pool_parts`仍保留原旧pool诊断。Face池外FP字段始终以旧pool为统计参考。
- `run/resume_verification.json`、`baseline_comparison.json`核验恢复；`completion_verification.json`核验2500更新、RNG遍历、参与计数、四组实际更新及最终checkpoint哈希。
- `per_mesh_evaluation.csv`、`evaluation_trend.csv/png`整理逐mesh及整体进展；严格成功UID可在原始逐meshJSONL中核对。

结束后自动生成`evaluation_package.zip`，含日志、代码、配置、核验、原挖掘记录与新增负例ID/logit；大checkpoint、原始mesh与完整pool数组保留服务器。此包不是独立推理模型包。

重点看Face FN是否恢复、严格成功集合是否扩大并保持，以及大mesh的Edge FP变化。训练完成不等于100条严格overfit，不能仅凭Face FP下降判断无代价改善。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_hardneg_continue100ep_20260916`。
入口：`bash run.sh`。进程启动后不要重复执行；`job_status.json`与`run/status.json`分别显示任务阶段和训练进度。
