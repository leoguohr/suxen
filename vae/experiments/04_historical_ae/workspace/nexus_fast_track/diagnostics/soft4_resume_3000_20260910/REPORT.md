# Small-only Soft4：原配置续训到3000步

恢复step1200模型及Adam全部状态（step、exp_avg、exp_avg_sq、参数组），新增1800次更新。E LR=1e-5、D LR=1e-4、τ=1、detach membership、FP32组归约、ε=1e-8、μ mode、Face=KL=wd=0、clip=1，后端不变。

最终：TP=1133、FP=2、FN=19、TN=73151；F1=99.081767%；Soft4=0.52752823，GT-balanced BCE=0.021621039。

首次严格100%步数：2314；续训评估中严格100%次数：19；最长连续严格100%：3步。没有提前停止或调整LR。

|末尾窗口|F1 最小/中位/最大|FP=FN=0次数|
|---|---|---:|
|末200步|98.09819% / 99.69525% / 100.00000%|12/200|
|末500步|95.74661% / 99.60784% / 100.00000%|18/500|

首次出现100%只说明当前配置到达过完全重建，是否稳定需要结合连续保持长度和末尾窗口。完整100%区间保存在summary.json。

恢复时模型参数、Adam状态逐张量相等；step1200重新前向与旧评估的差异记录在soft4/resume_verification.json。沿用Flash/CUDA后端，不声称逐位数值确定性。

服务器每200步保存包含Adam状态的checkpoint；若出现首次100%，另保存该步checkpoint与embedding。原始step1200和旧日志保留。combined_trace.jsonl按0–3000对齐，step1200采用恢复后的前向记录。

![训练曲线](training_curve.png)
