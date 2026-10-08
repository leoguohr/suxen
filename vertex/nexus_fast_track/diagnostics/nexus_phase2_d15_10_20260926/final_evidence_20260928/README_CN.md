# 如何阅读本包

先看 FINAL_REPORT_CN.md，再看 per_mesh_summary.csv / per_sample_metrics.csv。

- server/post_training_final_012000/：48 条真实预测、720 份逐层数组、完整评估与条件测试。
- server/runtime/：实际运行代码、manifest、12 个对象的固定点云/法向与 D15 labels，含浮点 GT。
- server/run/、server/audit/：原始训练日志、配置、恢复事件、checkpoint 身份、进程输出和状态。
- server/evaluation_recovery_20260928/：缺失依赖修复、旧预测哈希、实际命令、CPU 独立复算及完整权重结构检查。
- server/speed_optimization_20260927/、server/migration/：SSD 迁移、执行提速、数值对照和恢复证据。
- FILES_SHA256.json：每个打包文件的大小与 SHA256；EXCLUDED_CHECKPOINTS.json：排除的大权重路径与身份。

大权重/Adam 张量未打包。teacher_candidate 的代码仅用于原评价函数，不含老师权重或新增老师重放成绩。训练完成、证据完整与模型通过是不同结论；本轮 seen 16/40、unseen 0/8，未整体通过。

CPU 复算命令（需安装 numpy/scipy；在包的 server/runtime/scripts 中）：
python verify_phase2_final.py --root ../..

模型生成命令与 GPU/RNG 协议见 server/evaluation_recovery_20260928/launch.json。恢复入口原用途是只补缺失 split；本包已经完整，不应将它重复运行后说成新的未见终验。
