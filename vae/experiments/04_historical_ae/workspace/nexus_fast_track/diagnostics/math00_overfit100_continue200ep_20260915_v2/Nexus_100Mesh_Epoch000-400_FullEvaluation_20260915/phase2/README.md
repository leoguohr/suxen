# 固定100条 fresh512 第二段：epoch201–400

这是一段独立的新增预算：从第一段epoch200/update5000末尾完整恢复，再做5000次更新，到epoch400/update10000停止。保持同100条完整mesh，每条最终累计参与400次。

父checkpoint：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914/run/checkpoint-update05000.pt`

父SHA256：`6a652045b865b776ca9bfbddd68bf1fc454c1fe29ec5aa83dec6cc3629186457`

## 保持的设置

μ路径、KL=0、logvar冻结且不进入Adam。math00 FP32 MATH、确定性Graph、关闭TF32/autocast。fully-diff Edge+Face Soft4、原固定Face训练pool、原尺度/归约/threshold。每微批一条完整mesh，累积4条各自loss/4，global clip=1，四组Adam，wd=0。

恢复父权重、四组Adam所有矩状态/计数、shuffle/Python/NumPy/Torch/CUDA RNG、逐mesh参与计数。E/μ LR保持1e-5，D/Edge/Face heads保持1e-4；不重新warmup。runtime.setup仅构造同型网络和优化器容器，实际更新前由父checkpoint完整覆盖并核验，绝不使用构造时的随机参数/空Adam开始训练。

## 恢复与验收

- 更新前核对父SHA、全部模型张量、Adam全状态、RNG、UID顺序/组定义/参与计数，写入`run/resume_verification.json`。
- epoch200重新完整验收100条，与第一段末尾逐mesh比较loss、Edge/Face计数、候选覆盖及最小margin。必须复现原末尾结果；不一致则停止，不开始update5001。
- epoch250/300/350/400全量验收；暂停更新，同一checkpoint依次前向100条完整mesh。每次验收保存/恢复所有RNG。
- 实际Face由当次预测Edge图枚举，原30秒/mesh完整性策略不变。未完整记录null/下界，绝不算严格成功。
- 完整checkpoint仍每10epoch保存，包含参数、Adam、所有RNG和参与计数。
- 更新日志编号5001–10000；`additional_update`从1–5000。
- 达到同checkpoint100/100严格成功时重载复核，继续在当前预算内观察保持；不自动延长。

## 日志与交付

`run/updates.jsonl`、`run/eval-epoch*.jsonl`、`run/eval-summary-epoch*.json`、`run/epoch-order-*.json`是完整记录。`run/status.json`是当前状态；`run/complete.json`和`runner_exit.json`确认预算与退出。

数据与候选文件通过链接复用父实验的固定文件，不复制或重选UID。源文件仍按父manifest SHA核验。`runtime.py`与`evaluate.py`逐字复用父入口，改动仅在独立`train_continue.py`恢复和预算控制。

`run.sh`在训练退出后自动生成`evaluation_package.zip`。正常完成与失败分别如实记录；包内提供日志、验收表、配置、代码与文件SHA。大权重与候选数组仍保留服务器，以路径和已有SHA定位；评估包用于结果审阅，不冒充可离线推理的模型包。

运行入口：`bash /guohaoran/nexus_fast_track/diagnostics/math00_overfit100_continue200ep_20260915_v2/run.sh`。同目录已有日志时拒绝重复启动；不能把此入口当成中途任意checkpoint恢复工具。

部署说明：首次输出目录的文件操作出现存储阻塞，尚未启动训练，改用此独立_v2目录。不是新的模型实验或第二支训练。
