# Face改为Soft4：step9500续训1000步

完整恢复step9500模型和三组Adam状态，只将Face训练项由原hard four-group改为与Edge同一函数实现的Soft4。τ=1、ε=1e-8、membership detach、FP32组归约、固定除4。Face权重1，两条mesh等权参与。

E LR=1e-7，Decoder及Face head LR=1e-6；μ路径、KL=wd=0、clip=1、原Flash后端、原固定mixed面负例均保持。最终E/D Adam step10500，Face head Adam step4000。

|阶段|Mesh|Edge F1|实际Face F1|Face TP / FP / FN|
|---|---|---:|---:|---|
|9500原日志|Small|100.000000%|100.000000%|768 / 0 / 0|
|9500原日志|Large|100.000000%|99.844419%|5134 / 4 / 12|
|9500同权重重放|Small|100.000000%|100.000000%|768 / 0 / 0|
|9500同权重重放|Large|100.000000%|99.824937%|5132 / 4 / 14|
|10500结束|Small|100.000000%|100.000000%|768 / 0 / 0|
|10500结束|Large|100.000000%|99.824835%|5129 / 1 / 17|

|Mesh|Face GT-balanced BCE：起点 → 结束|Face Soft4：起点 → 结束|
|---|---:|---:|
|Small|0.1178091 → 0.001778671|0.2339022 → 0.006774301|
|Large|0.1353573 → 0.02713965|0.2707499 → 0.1468959|

两条Edge同时100%：827/1000次更新。每50步完整评估一次，两条Edge+Face同时100%的更新后评估点：[]。

实际Face指标先从预测边图枚举三角形，再由Face logit>0筛选；未进入候选的GT面也计入FN。固定训练候选上的F1不能代替此指标。完整Face并非每步枚举，因此只能报告已观察到的完整评估结果。

同权重起点保留旧日志与本次重放，Flash/CUDA数值差异并不等同于权重或优化器重置。Face Soft4下降说明其目标在改善，但不能单独证明实际面重建改善；旧hard-four与新Soft4数值不直接比较，采用共同的GT-balanced BCE及hard reconstruction作比较。

此次是有历史hard-Face训练的续训实验，不是Soft4从随机初始化的容量测试，也没有同步旧loss续训对照；不能据此归因所有残余错误。

checkpoint在服务器：`/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_20260910/joint_soft4/checkpoint-10500.pt`。逐步日志、公式梯度校验、完整面评估、配置与hash清单已保存本地。

本轮结果：small的实际面恢复在全部完整评估点均为100%；large换loss后先下降、随后恢复，最终1个FP、17个FN，尚未达到100%。相较本次起点重放4个FP、14个FN，总错误数仍为18；因此不能称为最终hard重建改善。最后全部GT面均进入候选，残余错误来自Face筛选而非漏边。

![训练与重建曲线](edge_face_soft4_curve.png)

最终checkpoint已重新加载、仅forward核验，两条mesh的Edge/Face TP、FP、FN均与训练末步一致；参数和文件hash保持不变。

最终checkpoint SHA256：`b12e8f031e2bf355c2ee5628937fc0beab9b2dc1b5b48552fea000b14c5ec0f2`。
