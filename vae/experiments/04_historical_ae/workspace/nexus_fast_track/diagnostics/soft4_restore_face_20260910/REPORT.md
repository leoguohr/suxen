# 恢复Face loss：Edge Soft4 + 原Face四组损失

起点为C分支step6500，两条edge均已完全重建。继续deterministic μ、KL=wd=0、clip=1、E LR=1e-7、D LR=1e-6。两条mesh共同反传，L=mean_mesh(EdgeSoft4+Face_original)，Face权重1。

**本轮Soft4仅用于Edge；恢复的Face项仍是原hard TP/TN/FP/FN非空组均值，并未改成Face Soft4。** 使用旧edge+face诊断的固定mixed负例：small1152个、large7892个；所有GT faces均作为正例，不重新抽样。

旧参数恢复Adam动量与step6500；此前冻结的face head解除冻结，保留原权重、只新建该head的Adam状态，学习率同Decoder。先训练1000步，再完整恢复三组Adam状态、相同配置续训2000步；最终旧组Adam step9500、face head step3000。

|阶段 / mesh|Edge F1|实际Face F1|Face TP / FP / FN|边候选图漏掉的GT面|
|---|---:|---:|---|---:|
|起点 / Small|100.000000%|59.854015%|328 / 0 / 440|0|
|起点 / Large|100.000000%|76.139854%|3223 / 97 / 1923|0|
|Face训练1000步后 / Small|100.000000%|85.884101%|578 / 0 / 190|0|
|Face训练1000步后 / Large|100.000000%|98.193954%|5002 / 40 / 144|0|
|Face训练3000步后 / Small|100.000000%|100.000000%|768 / 0 / 0|0|
|Face训练3000步后 / Large|100.000000%|99.844419%|5134 / 4 / 12|0|

新增3000步中两条Edge同时100%的次数：2861。每50步检查完整edge+face恢复，成功的更新后评估点：[]。

|最终训练候选指标|Edge Soft4|Face原四组loss|Face GT-balanced BCE|候选Face F1|
|---|---:|---:|---:|---:|
|Small|0.0031878874|0.11775604|0.11775604|100.000000%|
|Large|0.01720842|0.42670262|0.13518047|99.274872%|

实际Face评估从当前预测边图枚举三角形候选，再按face logit>0筛选；GT面若因漏边未进入候选，也计入FN。训练候选上的Face F1单独报告，不能代替最终mesh的面重建。每次完整评估复用该步训练前向的embedding，未另采样latent。

3000步内的完整评估尚未通过联合验收，不等于证明模型没有面重建容量；原Face损失、固定负例覆盖、新启用face head的优化和共享参数变化仍需分别判断。

原代码与旧checkpoint不修改。每200步保存模型及Adam，完整日志和评估记录已取回本地。

最终checkpoint（服务器）：`/guohaoran/nexus_fast_track/diagnostics/soft4_restore_face_20260910/joint_resume/checkpoint-9500.pt`。已有E/D参数Adam step9500，face head Adam step3000。

最终两条Edge正确，small的Face正确，large仍有4个错面和12个漏面；最终所有GT面均进入边图候选，剩余错误来自Face筛选。训练过程中两条Edge同时100%的比例为2861/3000，不能表述为加入Face后边完全没有受到扰动。

![Edge与Face曲线](edge_face_curve.png)
