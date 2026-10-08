# 固定100条：B末尾连续训练至epoch600

从 B epoch500 / update12500 的完整checkpoint开始，新增100 epochs、2500次实际更新，到epoch600 / update15000停止。保留原B epoch475与epoch500文件，不覆盖旧实验。

起点：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_lr03_pair_20260915/B_lr03/run/checkpoint-update12500.pt`

SHA256：`dc28a6e72a8e30a84d4536cbd09168d48635d66c5a924a3f52ef907c36fc0038`

四组Adam（含历史矩与计数）、所有RNG及每mesh参与计数完整恢复。E/shared+μ LR=3e-6，Decoder/Edge/Face LR=3e-5，直接保留实际浮点LR，不再乘0.3、不重跑warmup。固定100条、每次一个完整mesh的microbatch，累计4条后统一clip=1及Adam更新，按mesh等权。μ路径，KL=0，logvar专属参数冻结；math00、fully-diff Soft4、数据、训练pool、评分、阈值、归约与wd保持不变。

训练前逐条复现父checkpoint的100条完整验收，失败即停止。全量验收epoch500、525、550、575、600；暂停更新，在同一checkpoint上逐条完整评估100条。Face从当次预测Edge图枚举；保留原枚举时间限制，未完成项明确标记，不能当作严格通过。每10 epochs及完整验收点保存完整checkpoint。

## 查看材料

- `run/status.json`：当前训练/验收状态；`job_status.json`：训练、核验、打包阶段。
- `run/updates.jsonl`：全部2500条更新记录，含四组实际位移、梯度与裁剪、loss、UID与参与次数。
- `run/resume_verification.json`、`run/baseline_comparison.json`：状态恢复与epoch500逐条一致性核验。
- `run/eval-epoch*.jsonl`、`eval-summary`：逐mesh实际Edge/Face错误、margin、候选覆盖。
- `per_mesh_evaluation.csv`、`evaluation_trend.csv/png`：完成后的对比表与曲线。
- `completion_verification.json`：预算、全100轮shuffle顺序、参与计数、非零更新与最终checkpoint哈希核验。
- `runtime.py`、`source_archive/`、`review_runtime/`、`C_graph_only/`及运行生成的源码：实际执行与运行时替换代码。
- `evaluation_package.zip`：结束后自动生成。包含日志、核验、数据/pool清单与代码；大checkpoint、原始mesh及pool数组保留在服务器，不包含在评估ZIP。

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_B_continue100ep_20260915`。

启动命令：`bash /guohaoran/nexus_fast_track/diagnostics/math00_overfit100_B_continue100ep_20260915/run.sh`。已启动后不要重复执行；入口有锁且拒绝覆盖更新日志。

训练完成与100条严格过拟合是不同结论。重点比较后期FP/FN、严格成功UID和实际候选覆盖，不能只凭训练pool loss下降判断成功。本预算结束不自动追加、不改配置。
