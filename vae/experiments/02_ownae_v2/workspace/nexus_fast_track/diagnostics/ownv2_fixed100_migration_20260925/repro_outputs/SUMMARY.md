# 本轮结果

**原100条OwnAE-v2的代码迁移和CPU验证已完成，真实V2 GPU训练尚未开始。**

- 固定100条真实数据核验通过：106325顶点、309194 GT边、204330 GT面、84669234无向pair，200个mesh/topology哈希匹配。
- 七类CPU检查通过，包括20个五mesh组、UID/负例游标恢复、真实AdamW小型CPU模型精确恢复、流式Face穷举一致、分片中断/孤立分片恢复、候选外GT FN、授权关闭拦截及新训练入口CPU替身执行。
- native_models.py保持交接SHA；评分、Hard4、负采样函数AST一致。
- 本AE有效正式更新0，GPU耗时0，暂无训练checkpoint、首步GPU性能或正式100条重建成绩。CPU替身的两次更新不能计入V2训练。

## 当前唯一GPU阻塞

用户已改为持续训练，因此不再以空预算字段阻塞。最初指定UUID在新服务器不存在；现有GPU0已经被其他任务占用，GPU1空闲但尚待明确选择。run_config.json仍为gpu_authorized=false，不能启动。已通过对话请求GPU1使用确认，不会停止或修改另一个任务。

独立服务器代码目录：/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_migration_20260925/
计划运行目录：同目录run/（未创建）。持续模式默认每1000次有效更新完整评价。
