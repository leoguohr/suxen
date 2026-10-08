# 代码改动记录

独立分支：`repro/2026-09-22-cad-hard4`；训练实现commit：`e0c6a0644c2624872f71ab51b9911f3d9c110ce9`。

风险：loss切换属于改变科学定义的实验干预，已由本轮用户明确授权；不是暗中修复原模型。所有改动只在本轮独立目录。原工程README、旧Soft4/stopgrad、原100条与源checkpoint保持不动。

有效差异见EFFECTIVE_CODE_DIFF.patch：新增Hard4组统计及归约，接入Edge/Face objective；起点检查仅移除“不同loss应数值相同”的旧断言，保留全部预测/状态检查；以复用对照的核验替代双新进程屏障。预算、反传、clip、Adam及checkpoint/eval顺序不变。

验证：hard4_test.json、CONTROL_REUSE_AUDIT.json、H_paper_hard4/startup_gate.json与ACCEPTANCE.json。原README意图没有被冒称为论文结果复现；本README仅描述用户批准的有限实验。
