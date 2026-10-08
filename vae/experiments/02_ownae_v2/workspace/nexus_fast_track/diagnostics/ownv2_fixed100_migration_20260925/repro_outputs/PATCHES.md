# 补丁

- Branch：repro/2026-09-25-fixed100
- 验证后的代码提交：4abd67a768cf6b4f2f72b42a178e4aa85b59696b
- 原始交接快照提交：2b9b15bd9e780aba2dedadc7d38923a3a4df3fed
- 范围：仅本独立目录；旧实验与diffusion源码未修改。
- native_models.py未改变；旧CAD入口与GPU预检移到reference/供阅读。
- data_objective.py：真实100条加载、逐mesh pair、20组调度与游标。
- train_fixed100.py / runtime_fixed100.py：独立fresh/resume/evaluate，完整AdamW/RNG恢复、设备授权与排他锁、账本、首步性能记录。
- evaluate_checkpoint.py / stream_faces.py：不可变checkpoint绑定、实际预测边图全量唯一候选、原子分片、续评与完整100条聚合。
- test_cpu.py：七类CPU检查，详见CPU_TESTS.json；真实GPU训练尚未验证。
- 最高未验证风险：真实大模型整步资源和稠密边图全量评价成本，不能用CPU合成测试替代。
