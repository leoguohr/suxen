# CAD50 Soft4 权重梯度配对实验

读取 `REPORT.md` 了解结果，`actual_trend.csv` 查看两支全部完整验收轨迹，`per_mesh.csv` 查看每个UID；`large66_274_per_mesh.csv` 单列16条较大CAD。训练逐步记录在各分支 `updates.jsonl`，逐步Edge指标来自更新前，不等于实际Face评价。

共同父状态：上一轮 `teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt`，SHA256为 `4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`。

- A：`--mode full`，Soft4权重及分母参与梯度。
- B：`--mode stopgrad`，仅权重概率 `sigmoid(logits)` detach；每次forward重算权重，BCE仍使用可微logits。
- 两支各100次全50条梯度累积更新，继承完整Adam和RNG，末尾各累计2600。独立两张A100、独立进程、同一代码，没有DDP。
- 四组LR为3e-6/3e-5/3e-5/3e-5，无warmup、调度、额外外层系数。Edge全部pair，原Face固定pool13946；μ、KL=0、logvar冻结；完整512 AE、math00。
- 完整验收新增0/25/50/75/100；Face从每次预测Edge图完整枚举三角形，logit>0，无输出修复。

`MINIMAL_LOSS_DIFF.patch` 是相对原有效loss/runtime的最小实现差异。`soft4_test.json` 保存唯一小测试；`release.json` 保存双GPU同起点及同full梯度核验。`independent_audit.json` 是训练完成后的CPU文件、状态和预测核验，不增加optimizer更新。

源数据与pool、完整评价JSON、预测数组、执行代码和实际依赖均在评估包。`FILE_MANIFEST.json` 给出包内每个文件的SHA256。模型大文件不重复打包，`MODEL_ARTIFACTS.json` 提供服务器绝对路径、大小和SHA256；每支的 `best_face_f1.json` / `best_strict.json` 引用保留的完整model/Adam/RNG checkpoint，最终另有完整推理文件。

服务器入口目录：
`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_soft4_stopgrad_pair_20260922`

本轮实际启动入口为 `/opt/conda/bin/python -u run_pair.py`。入口有防覆盖与空闲GPU检查；已执行的目录不能重复启动。`run_pair.py` 中绑定的是本轮已核验的两张GPU UUID，不应盲目在别的服务器复用。`audit_package.py` 仅做无GPU的交付核验。

所有依赖目录引用只用于复现代码定义。本轮没有加载原100条模型，也没有停止、修改或覆盖原100条任务。
