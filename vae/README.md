# 冻结 OwnAE-v2 VAE：step36220

按用户最终确定的范围，仅交付当前冻结版的有效代码、配置、固定100条数据、最后一段完整结果与最终可续训模型。旧AE、CAD50和其他历史实验不属于本次交付。

最终μ路径实际Face micro-F1为99.6306%，联合严格成功26/100；五组sampling的Face micro-F1约99.6279%—99.6323%。这是已冻结的工作版本，尚未达到0.997或100/100严格零错误。

- [当前有效代码](current/code/)、[配置](current/config.json)、[完成记录](current/run/complete.json)
- [原始报告](current/REPORT.md)、[结论与指标边界](CONCLUSIONS.md)
- [下载与模型恢复](RESTORE.md)、[来源和SHA256清单](manifests/)
- [本任务Release](https://github.com/leoguohr/nexus/releases/tag/vae-evidence-2026-10-08)

代码保持服务器实际版本；12个Python源码已逐字节核对，未新增GPU测试。显式使用B_v2_teacher_blocks，不能使用默认A结构。恢复训练必须加载完整model、AdamW、RNG和游标；推理权重不能替代续训状态。

代码中的绝对路径、GPU和预算是历史环境记录，不是自动启动授权。迁移时映射路径并保持原数据数组、顶点编号与拓扑不变。

本分支只修改vae/；另一个任务的Vertex Diffusion、老师材料和evidence-2026-10-08 Release保持独立。
