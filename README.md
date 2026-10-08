# NEXUS 实验与证据归档

**私密仓库 · 截至 2026-10-08 的现有结果快照**

本仓库汇总本会话的 NEXUS 八叉树 Vertex Diffusion：代码、历史实验、结构调研、已保存预测、训练日志与结论；同时保存老师原始交付包、候选复建代码及独立核验。独立 VAE/OwnAE-v2 已完成，本次未改动或重新训练。

- [完整下载包（Release）](https://github.com/leoguohr/nexus/releases/tag/evidence-2026-10-08)
- [各阶段结果与结论](docs/RESULTS_AND_CONCLUSIONS.md)
- [本次资产与完整性清单](docs/ASSET_INDEX.md)
- [Vertex 代码与研究文档](vertex/)
- [老师候选代码与核验文档](teacher/)

## 当前确认的结果

| 实验 | 已有结果 | 适用范围 |
|---|---|---|
| D2 / D15 两个固定对象 | 各 32/32 完整树正确 | 已训练对象过拟合 |
| D15 十对象 Phase2 | seen 16/40；unseen 0/8 | 固定点云/法向条件 |
| D9 A–G 筛查 | 各完整树 0/100 | 保留不同采样器、精度及执行后端边界 |
| E2 T，新增 2000 updates | GT-parent D9 单层纯噪声采样 39/100；pooled F1 0.990016 | 能力定位，不能记为完整树生成 |
| 结构 sweep S0 / S1 | 各新增 10000 updates；各完整树 7/100 | 50 个训练对象 × 2 个固定种子 |
| 结构 sweep S2 / S3 | 日志分别到 6098 / 5688；可恢复点 6000 / 5600 | 未完成、无终点生成成绩 |
| S4 / S5 / S6 | 尚未启动 | 已排计划，不计成绩 |

服务器最新实例未见本任务进程。保存的旧 `queue_status.json` 中 `running` 是上个实例的历史状态。本次仅采集与上传，没有重启训练。

## 材料对应关系

| 分类 | 可浏览内容 | 完整数组及原件 |
|---|---|---|
| 自己的 Vertex | `vertex/` | 21个 `Nexus_Vertex_Historical_*_part*.zip` 及服务器证据资产（见下载索引） |
| 老师候选复建与核验 | `teacher/` | `Nexus_Teacher_Reconstruction_20261008.zip` |
| 老师原始交付 | 原件身份见 `manifests/teacher_original_identity.json` | `Teacher_Original_nexus_overfit_data_results_no_code.zip` |

老师原始交付 ZIP 含数据、权重、条件缓存、配置、日志和预测，**不含原训练源码**。本仓库老师代码是经核验的候选复建。老师文本/坐标 prior 任务与自己的点云/八叉树任务分别报告。

## 读取顺序

1. 先读 `docs/RESULTS_AND_CONCLUSIONS.md`，区分已完成、未完成和诊断实验。
2. 用 `docs/ASSET_INDEX.md` 下载对应 Release ZIP。
3. 用 ZIP 内 `FILES_MANIFEST`、`ORIGIN_PATH_MAP` 与 SHA256 清单定位原始证据。

所有去重仅按字节 SHA256 完全一致进行；原路径映射保留。旧文档和状态快照保留原样，日期较新的顶层索引解释其时效。2026-10-01 的长期研究文档对 E/F/G 的“未实施”描述已过时。

## 排除与恢复边界

- 自己模型的约 28 GB 完整 checkpoint 未上传；保留服务器路径、SHA、model/Adam/RNG 身份及恢复边界。
- 老师原始 ZIP 保留原样，其中已有老师权重一并保留。
- 排除环境目录、轮子、缓存、Git 元数据和凭证文件；每包记录排除清单。
- 保存数组复算、GT-parent 诊断、表示往返测试和真实完整生成分别标注。
- 本次未重新运行模型；新增工作为证据采集、归档、完整性检查与上传。
