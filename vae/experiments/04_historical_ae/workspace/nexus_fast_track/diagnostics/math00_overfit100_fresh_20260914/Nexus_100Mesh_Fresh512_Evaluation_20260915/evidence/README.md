# 固定100条mesh：fresh512 μ overfit

已生成并锁定100条清单、完成全部数据/候选校验及资源预检。正式入口已提交，初始全量验收与训练进度以服务器run/status.json为准。

## 清单

- 来源：人工去重后训练split，1060条；提前限定4～2600顶点，不按训练效果选样。
- 固定种子20260915；6个规模区间分别17、17、17、17、16、16条。
- 实际18～2547顶点，中位数953.5；共84669234个无向pair、204330个GT Face。
- 100条清单：overfit100_manifest.csv；完整原数据路径：data_manifest.csv；排除原因：excluded.csv。
- 与旧20条清单重叠：nexus_2k_001716。
- 清单SHA256：b90e14f5d318dc0d32ad806c631af7f66be28aa644ff3caf357d34ce3e208b93。

## pool与初始化

Face继续使用冻结positive+mixed监督集合。已有pool复用；新UID按历史规则准备全部GT false cycles、固定wedge/uniform，并使用固定旧模型挖出的负例替换最多一半非cycle位置。旧模型仅用于离线pool生成，不作为训练初始化，不参与正式训练；没有表示蒸馏。

离线挖掘曾因旧500万候选保护上限暂停；仅将资源保护上限提高至1亿，完整生成约1067万候选后继续。没有更换UID、截断候选或改变选择公式。prepare-attempt日志在服务器保留。

Face训练pool不等于当前预测Edge图候选。本轮暂不改变监督策略，实际Face评价记录pool之外的FP；未完成枚举不冒充成功。

## 训练协议

- 固定seed正常随机初始化512 latent、1024 decoder hidden、各32维评分embedding。
- 原XYZ/face centroid/incidence输入，math00 FP32 MATH和确定性Graph，原评分scale、fully-diff Soft4。
- μ路径、sampling/KL=0；logvar冻结、不进Adam；四组fresh Adam、wd=0、global clip=1。
- microbatch为1条完整mesh，累积4条各自loss/4后clip和step一次。
- 每epoch随机无重复遍历100条；25次更新；200epochs=5000更新=每条200次参与。
- LR在第1～100更新线性warmup：E/μ 1e-6→1e-5，D/heads 1e-5→1e-4，之后保持。
- 5000更新后停止，不自动延长。

## 资源预检

100条完整前向/反向遍历：43.105秒；峰值显存11.103GiB；A100 80GB。
独立微批梯度相加与一次对损失和反传核验误差：0.0。
预检optimizer更新0次，所有参数未变，Adam状态0。initial.pt是本实验自己的随机初始权重。
200epochs纯前向/反向估计2.39小时；不含Adam、参数位移统计、checkpoint和全量实际Face验收，不能当成总完成时间。

## 验收

epoch0/10/25/50/100/150/200：同一checkpoint、停止更新、依次完整评估100条。不是packed100。
每条Edge计算全部pair；Face从当次预测Edge图枚举。Face枚举采用32K块、每条30秒时间预算；超时的FP/F1写null，已评分FP仅记下界。GT Face FN可由GT logits与边图覆盖精确统计，未入候选仍计FN。未完整且不能确定全对的Face不算通过。
最终严格成功要求同一全量评估100条Edge+Face均FP=FN=0。首次100/100会重载同一checkpoint复核。

## 服务器目录与日志

`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914`

- train-console.log：进程输出；run/status.json：当前阶段/epoch/update。
- run/updates.jsonl：每次实际更新、4条UID及参与数、各loss、LR、梯度、clip、实际更新与表示尺度。
- run/epoch-order-*.json：每epoch完整顺序。
- run/eval-epoch*.jsonl：逐mesh完整验收及Face是否完成。
- run/eval-summary-epoch*.json：三个严格成功计数、四项错误合计/下界、未通过列表。
- run/checkpoint-update*.pt：模型、四组Adam、各RNG、参与计数；每10epochs及指定验收点保存。
- run/complete.json与runner_exit.json：完成及退出记录。尚未生成时不能报告已训练完成。

查看进度：

```bash
cat /guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914/run/status.json
tail -f /guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914/train-console.log
```

正式启动命令已执行一次：`bash /guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914/run.sh`。已有任务运行时无需重复执行。
