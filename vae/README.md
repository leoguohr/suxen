# OwnAE-v2 / VAE 代码、结果与证据归档

归档日期：2026-10-08。所属任务：拓扑 AE / VAE 与本任务的 CAD50 诊断。

这是项目的独立实现与实验记录。当前冻结模型为 **OwnAE-v2 VAE step36220**。其 μ 路径实际 Face micro-F1 为 **99.6306%**，联合严格成功 **26/100**；五组采样结果约 **99.6279%—99.6323%**。冻结是项目状态，不表示达到 0.997 或 100/100 严格重建。

- [当前结论与边界](CONCLUSIONS.md)
- [冻结主线原始报告](current/REPORT.md)
- [有效模型实现](current/code/native_models.py)
- [续训入口](current/code/train_continue.py)、[VAE 定义](current/code/vae_protocol.py)、[Hard4 与数据接口](current/code/data_objective.py)
- [完整配置](current/config.json)、[完成记录](current/run/complete.json)
- [分卷恢复与文件校验](RESTORE.md)
- [文件来源与校验清单](manifests/)
- [大文件 Release](https://github.com/leoguohr/nexus/releases/tag/vae-evidence-2026-10-08)

## 内容组织

`current/` 保留最终 VAE 的实际生效代码和配置；`experiments/` 保存历史 AE、OwnAE-v2、CAD50 与结构对照代码/报告；大体积逐步日志、预测数组、评价包、学习讲义、固定100条原始数据和最终完整 checkpoint 放在本任务专属 Release。

本分支只新增仓库的 `vae/` 目录。Vertex Diffusion、老师原材料、`evidence-2026-10-08` Release 及其标签和资产由另一个任务维护，本次没有改写。

## 复现入口注意事项

1. `native_models.Config` 的默认分支是 A。当前模型需要明确使用 `B_v2_teacher_blocks`，或读取完整 checkpoint 的 `model_config`，不可依赖默认构造。
2. `current/` 是历史有效源码的字节一致副本，保留原路径、GPU UUID、预算和运行目录记录。归档不代表这些资源现在可用；不要直接重跑历史启动脚本或覆盖原输出。
3. 从现有 VAE 恢复需完整 checkpoint 的 model、AdamW、RNG、数据游标和参数组映射。推理权重不能替代恢复状态。
4. 固定100条数据按原 FP32 数组、顶点编号和拓扑保存。数据入口核验 selection/manifest 的原始 SHA；历史 manifest 中的绝对路径需在新机器上正确映射。不要通过改数组来消除路径问题。
5. μ 重建、采样重建与 CAD50 的确定性 AE 对照分开报告；Face 的实际验收来自预测 Edge 图枚举的全部候选。

## 归档范围与未复制的大文件

最终 step36220 完整 model/AdamW/RNG checkpoint 已按 850 MiB 上限分卷。固定100条 mesh/topology 原数组与清单单独归档。已选实验的本地材料和服务器当前 VAE / 最新 CAD50 结果均带来源与 SHA256。

历史中间 checkpoint、重复 recovery checkpoint 和重复 `network-output.npz` hidden 缓存未全部复制。可用来源、文件大小及已有 checkpoint 哈希保留在 `manifests/` 与原实验元数据中；没有在本轮重新计算的哈希明确标记。实际 Face 预测分片、逐mesh评价与日志属于本次结果归档范围。

本次操作只读取原实验文件、制作归档和上传，没有训练、参数更新或新增 GPU 评价。历史文档中的授权和下一步建议仅作为当时记录保存。
