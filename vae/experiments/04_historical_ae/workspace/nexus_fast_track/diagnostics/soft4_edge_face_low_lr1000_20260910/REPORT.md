# B支低LR续训1000步：两条mesh Edge/Face验收

从B支step11100模型和完整三组Adam状态恢复，续训1000步至step12100。E LR=1e-8，Decoder及Face head LR=1e-7，保持Edge Soft4+Face Soft4，Face权重1，两条mesh等权完整参与。

保持τ=1、ε=1e-8、detach membership、FP32组归约、μ模式、KL=wd=0、clip=1、原Flash后端及固定mixed面负例。判定阈值仍为0。每200步保存模型及Adam，每步记录Edge与固定训练候选Face指标，每50步完整枚举预测边图中的三角形并评分。

|阶段|Mesh|Edge F1|实际Face F1|Face TP / FP / FN|
|---|---|---:|---:|---|
|B原末步 / 11100|Small|100.000000%|100.000000%|768 / 0 / 0|
|B原末步 / 11100|Large|100.000000%|99.922209%|5138 / 0 / 8|
|起点重放 / 11100|Small|100.000000%|100.000000%|768 / 0 / 0|
|起点重放 / 11100|Large|100.000000%|99.951395%|5141 / 0 / 5|
|1000步后 / 12100|Small|100.000000%|100.000000%|768 / 0 / 0|
|1000步后 / 12100|Large|99.993523%|99.980564%|5144 / 0 / 2|

两条Edge同时100%：975/1000次更新。两条Edge+Face同时100%的更新后完整评估点：[]（共20次完整评估）。

|Mesh|Face Soft4 起点→结束|Face GT-balanced BCE 起点→结束|最后200步实际Face F1 min/median/max（4次完整评估）|
|---|---|---|---|
|Small|0.004434004 → 0.003642033|0.001244839 → 0.00100163|100.000000% / 100.000000% / 100.000000%|
|Large|0.1171239 → 0.1081358|0.01894469 → 0.01489615|99.912477% / 99.946532% / 99.980564%|

实际Face验收先由预测边图枚举三角形，再以Face logit>0筛选；因漏边未进入候选的GT面也计FN。固定训练候选Face F1不能代替实际面重建。完整评估仅每50步一次，成功次数不冒充每一步的计数。

记录原B末步、本次起点重放及最终checkpoint重放，保留原数值后端的前向差异。loss下降不能代替两条mesh同时FP=FN=0的验收。

最终checkpoint（服务器）：`/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_low_lr1000_20260910/continue/checkpoint-12100.pt`。E/D参数Adam step12100、Face head Adam step5600。旧checkpoint和实验代码未修改。

最终checkpoint仅forward重新核验，SHA256：`fb5dc4d642a545bb468393905bfd09726967c6d55e9a03e23629ec00d6322c06`。

- Small：Edge FP/FN=0/0，Face FP/FN=0/0，Face F1=100.000000%。
- Large：Edge FP/FN=0/0，Face FP/FN=0/3，Face F1=99.970843%。

![训练及重建曲线](edge_face_soft4_curve.png)
