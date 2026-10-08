# 固定100条 LowLR：epoch800 → 900

共同状态来源：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_low_lr_continue100ep_20260916/run/checkpoint-update20000.pt`。
SHA256：`38087aeefbc12444cb51ba35dbe3715cc0f785c30cd6ed53e093a0ee7b929f80`。

新增100 epochs、2500次实际更新（update20001—22500），到预算停止。完整继承权重、四组Adam及其LR、全部RNG、固定100条名单与参与计数。E/μ=9e-7，Decoder/Edge/Face heads=9e-6。不再次缩放，不warmup，不重跑对照。

现有扩充Face pool及187932个新增难负例保持不变，不挖掘或刷新。math00、确定性Graph、μ路径、KL=0、logvar冻结、fully-diff Soft4、wd=0和global clip=1保持原定义。有效batch4，每个microbatch为一条完整mesh，四条各自目标除4累计梯度后更新一次。

## 验收及日志

更新前先核验父checkpoint哈希、全部权重及Adam/RNG严格恢复、数据与pool哈希，再复现epoch800的100条完整验收；任何不一致均停止。起点应为47/100严格成功、Edge FP/FN=262832/0、实际Face FP/FN=54518/258。

全量验收在epoch800、825、850、875、900，固定同一个checkpoint，逐条完整评价100条。实际Face始终由当次预测Edge图枚举；未完成评价明确标记，不当作0错误。评价保留并恢复RNG。每10 epochs及验收点保存完整checkpoint。日志保留逐步loss、四组实际更新、梯度及clip；逐mesh验收、GT Face候选覆盖、margin、旧pool诊断loss及固定难负例ID/logit沿用原代码。

严格成功始终报告为固定100条中的成功数；不能将不同checkpoint的成功UID拼接，也不筛选成功子集冒充全部通过。训练pool FP与实际Face FP区分记录。末尾停止复盘，不自动追加。

## 材料位置

- `run/updates.jsonl`：全部2500次更新。
- `run/resume_verification.json`、`run/baseline_comparison.json`：完整恢复和起点验收核验。
- `run/eval-epoch*.jsonl`、summary：逐mesh实际结果、严格成功UID与汇总。
- `hardneg_logits/`：各检查点固定难负例ID/logit。
- `completion_verification.json`：预算、shuffle序列延续、每条新增100次参与及四组参数更新核验。
- `evaluation_trend.csv/png`、`per_mesh_evaluation.csv`：完整趋势与逐mesh表。
- `evaluation_package.zip`：日志、配置与provenance、有效执行代码、核验、图表和SHA256SUMS。

大checkpoint、原始mesh与完整pool数组留服务器；路径和哈希见manifest、provenance及验收summary。旧实验目录不覆盖。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_low_lr_epoch800_900_20260917`。
入口：`bash run.sh`。不得重复启动已有任务。
