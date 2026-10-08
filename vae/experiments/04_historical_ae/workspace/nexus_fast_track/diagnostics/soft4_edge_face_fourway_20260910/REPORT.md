# B支续训：累计5次four-way perfect才提前停止

从step12100模型及完整Adam状态恢复，E LR=1e-8、Decoder/Face head LR=1e-7，Edge Soft4+Face Soft4；τ=1、ε=1e-8、membership detach、FP32归约，μ路径，KL=wd=0，clip=1及固定负例均不变。最大新增2000步至14100。

每50步完整评估同一forward：两条mesh的Edge与Face全部FP=FN=0才计一次。累计达到5次就保存并停止，不要求连续；起点重放不计数，非完整评估步也不计数。每200步保存，并额外保存每个成功完整评估点的模型、Adam和embedding。

实际新增2000步，结束于step14100；停止原因：2000-update budget completed。four-way perfect累计0/5，成功评估点：[]；更新后完整评估共40次。

|阶段|Mesh|Edge FP/FN|Edge F1|Face FP/FN|Face F1|
|---|---|---|---:|---|---:|
|前次末步 12100|Small|0/0|100.000000%|0/0|100.000000%|
|前次末步 12100|Large|1/0|99.993523%|0/2|99.980564%|
|起点重放 12100|Small|0/0|100.000000%|0/0|100.000000%|
|起点重放 12100|Large|1/0|99.993523%|0/4|99.961120%|
|本轮末步 14100|Small|0/0|100.000000%|0/0|100.000000%|
|本轮末步 14100|Large|0/0|100.000000%|0/2|99.980564%|

两条Edge同时100%：1914/2000次更新。它与four-way perfect分开统计，不能替代面重建验收。

|Mesh|Face Soft4起点→结束|Face GT-balanced BCE起点→结束|最后200步实际Face F1 min/median/max|
|---|---|---|---|
|Small|0.003799702 → 0.002440332|0.0009884923 → 0.0006570459|100.000000% / 100.000000% / 100.000000%（4次评估）|
|Large|0.1069843 → 0.07285673|0.01483954 → 0.009191307|99.970848% / 99.980564% / 99.990283%（4次评估）|

实际Face指标从当前预测边图枚举三角形，再按Face logit>0筛选；因漏边未进入候选的GT面也算FN。完整Face仅每50步评估，成功次数不表述为每一步成功次数。

最终checkpoint（服务器）：`/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_fourway_20260910/continue/checkpoint-14100.pt`。

原参数Adam step=14100，Face head Adam step=7600；旧checkpoint和实验代码未修改。

|Checkpoint重放（仅forward）|Mesh|Edge FP/FN|Face FP/FN|
|---|---|---|---|
|14100|Small|0/0|0/0|
|14100|Large|0/0|0/2|

重新forward的计数与保存时计数分别保留；原Flash/CUDA数值路径的敏感性没有在本轮通过换后端消除。若计数改变，不能表述为额外训练带来的变化。

本轮判断：Face Soft4与GT-balanced BCE继续下降，但在40次完整评估中，large始终有1～4个漏面，没有出现four-way perfect。最终两条Edge正确、small Face正确，large剩2个漏面；最终checkpoint重新forward仍为同样的FP/FN计数。累计5次的门槛未通过，因此按2000步预算上限结束。这个结论仅限本轮训练与评估，不能表述为已经完全过拟合，也不能据此证明模型容量不足。

![训练与重建曲线](fourway_curve.png)
