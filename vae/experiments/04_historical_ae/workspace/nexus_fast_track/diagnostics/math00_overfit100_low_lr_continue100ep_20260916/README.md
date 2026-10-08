# 固定100条 LowLR：epoch700 → 800

起点：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_hardneg_lr03_from650_20260916/run/checkpoint-update17500.pt`。
SHA256：`20c75fc55fd0f432e327db25a96c8bc151008107e76b2ba9ac72e144edaa4e12`。

新增100 epochs、2500次实际更新，update17501—20000；完成即停止。恢复四组Adam全部状态与LR、所有RNG、100条名单及参与计数。E/μ=9e-7，Decoder/Edge/Face heads=9e-6；不再乘0.3，不重新warmup。

保持现有187932个新增难负例组成的固定扩充pool，不重新挖掘。math00、μ、KL=0、logvar冻结、fully-diff Soft4、wd=0、global clip=1均不变。有效batch4，每条完整mesh分别前后向并将目标除4，四次梯度累计后一次更新。

完整实际验收：epoch700、725、750、775、800，Face仍由当前预测Edge图枚举。更新前须复现父checkpoint的100条loss、实际计数、margin及难负例统计；否则停止。评价保存并恢复RNG，不推进训练随机序列。每10 epochs及所有验收点保存完整checkpoint。

## 评估材料

- `run/updates.jsonl`：2500次更新、逐mesh loss、LR、梯度、clip和四组实际更新。
- `run/resume_verification.json`：权重、完整Adam（包括LR）、RNG、数据及pool哈希一致。
- `run/baseline_comparison.json`：起点100条与父验收一致。
- `run/eval-epoch*.jsonl`及summary：实际Edge/Face计数、候选覆盖、旧pool诊断loss、难负例统计及严格成功UID。
- `hardneg_logits/`：各检查点固定难负例ID/logit。
- `completion_verification.json`：预算、100轮shuffle连续恢复、每条新增100次参与及四组参数实际更新核验。
- `evaluation_trend.csv/png`、`per_mesh_evaluation.csv`：完整趋势与逐mesh数据。
- `evaluation_package.zip`：上述材料、执行代码、有效运行时代码与SHA256SUMS。

大checkpoint、原始mesh及完整pool数组留在服务器；checkpoint路径和哈希见验收summary，数据与pool路径和哈希见manifest及provenance。旧Control和LowLR均不覆盖。

严格成功始终以固定100条中的实际成功数报告，不事后筛选成功子集冒充100条通过。训练pool的FP与实际Face FP是不同集合；不能用训练pool指标替代实际重建。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_low_lr_continue100ep_20260916`。
入口：`bash run.sh`。任务已启动时不要重复执行。
