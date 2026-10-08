# 改动边界

独立目录和分支repro/2026-09-29-overfit50。基线d15_code与原脚本逐字复制，未修改基线或AE/V2。仅新增overfit50_prepare.py、overfit50_train.py、overfit50_test_ddp.py、overfit50_evaluate.py、overfit50_pipeline.py及记录。

最高实现风险为双卡梯度缩放与恢复/保存；CPU等效梯度和完整模型双卡最大样本预检通过，正式完整checkpoint的SHA回读与独立CPU结构检查已通过；恢复执行未另行中断测试。README新增的是用户授权独立实验，不声称原论文实现或最终通过。
