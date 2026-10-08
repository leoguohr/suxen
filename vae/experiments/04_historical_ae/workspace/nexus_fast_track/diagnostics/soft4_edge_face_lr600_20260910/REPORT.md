# step10500：Edge+Face Soft4，学习率A/B各600步

A：E=1e-7，Decoder与Face head=1e-6；B：E=1e-8，Decoder与Face head=1e-7。均独立从相同step10500 checkpoint及全部三组Adam状态恢复，先核对状态张量完全相同，再设置分支LR。两条mesh每步完整参与，Edge Soft4 + Face Soft4，每条mesh等权，Face权重1。

τ=1，ε=1e-8，membership detach，FP32组归约，μ路径，KL=wd=0，clip=1，同一后端、数据顺序和固定mixed面负例，threshold=0。GPU0依次跑A、B，各600步，不提前停止。每200步保存模型和Adam；每步记录Edge与固定训练候选Face，每50步完整评估实际面重建。

|分支 / 阶段|Mesh|Edge F1|实际Face F1|Face TP / FP / FN|
|---|---|---:|---:|---|
|A / 10500|Small|100.000000%|100.000000%|768 / 0 / 0|
|A / 10500|Large|100.000000%|99.844328%|5131 / 1 / 15|
|A / 11100|Small|100.000000%|100.000000%|768 / 0 / 0|
|A / 11100|Large|100.000000%|99.902743%|5136 / 0 / 10|
|B / 10500|Small|100.000000%|100.000000%|768 / 0 / 0|
|B / 10500|Large|100.000000%|99.815085%|5128 / 1 / 18|
|B / 11100|Small|100.000000%|100.000000%|768 / 0 / 0|
|B / 11100|Large|100.000000%|99.922209%|5138 / 0 / 8|

|分支|两条Edge同时100%的更新次数|两条Edge+Face同时100%的完整评估点|耗时|
|---|---:|---|---:|
|A|456/600|[]（共12次更新后评估）|3.68分钟|
|B|581/600|[]（共12次更新后评估）|3.64分钟|

|分支 / Mesh|Face Soft4 起点 → 终点|Face GT-balanced BCE 起点 → 终点|最后200步实际Face F1 min / median / max（4次完整评估）|
|---|---:|---:|---|
|A / Small|0.006653214 → 0.01349023|0.001784167 → 0.001026128|100.000000% / 100.000000% / 100.000000%|
|A / Large|0.1459891 → 0.1144582|0.02682608 → 0.01590944|99.902743% / 99.912476% / 99.951395%|
|B / Small|0.007234178 → 0.004422365|0.001827629 → 0.001226784|100.000000% / 100.000000% / 100.000000%|
|B / Large|0.1446941 → 0.1180612|0.02673322 → 0.01912024|99.854071% / 99.917343% / 99.922209%|

每次实际Face评估复用当步训练forward的embedding，从当前预测边图枚举三角形、按Face logit>0筛选；GT面未进入边图候选也计FN。训练候选Face F1与实际重建不同，完整Face成功次数不能表述为逐步成功次数。

相同起点指参数和Adam状态张量经逐项验证相同；既有Flash/CUDA数值路径不保证重新forward逐位一致。因此A/B起点重放分别保存，与原step10500日志分开。这里只改变LR，没有用阈值扫描最优值进行验收。

最终checkpoint（服务器）：

- `/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_lr600_20260910/A/checkpoint-11100.pt`
- `/guohaoran/nexus_fast_track/diagnostics/soft4_edge_face_lr600_20260910/B/checkpoint-11100.pt`

已有E/D参数Adam step11100，Face head Adam step4600。原始checkpoint和旧实验代码均未修改。启动前曾修正调度脚本与Python标准库queue的命名冲突，失败启动未生成训练trace、未做参数更新。

A最终checkpoint重新加载、仅forward复查（参数未更新），SHA256：`51cec71b269f7df8c6c12864678352b1adaa544668902d8e71914484231c6dfc`。

- Small：Edge FP/FN=0/0，Face FP/FN=0/0，Face F1=100.000000%。
- Large：Edge FP/FN=0/0，Face FP/FN=0/8，Face F1=99.922209%。

B最终checkpoint重新加载、仅forward复查（参数未更新），SHA256：`64bf0d75765353c3bc03859b8dfa1ff241fd99aec71afda4870d70dac302a2a9`。

- Small：Edge FP/FN=0/0，Face FP/FN=0/0，Face F1=100.000000%。
- Large：Edge FP/FN=0/0，Face FP/FN=0/7，Face F1=99.931940%。

本轮判断：B的两条Edge同时100%次数为581/600，优于A的456/600；但Face并未表现出同样明确的优势。B终点少2个漏面，A最后200步的4次完整评估Face F1中位数略高，且A最好的评估点为5个漏面，B为7个。两个分支均未出现完整边+面同时100%。

A的large Face Soft4和GT-balanced BCE终值更低，说明继续原LR仍在改善目标；B降低LR主要观察到边稳定性改善。仅凭此次600步实验，不能认定降低十倍LR已经解决Face残余错误，也不能由终点2个面的差距宣称B在Face上明确胜出。

最终checkpoint重新forward后large分别为A：0 FP/8 FN，B：0 FP/7 FN，与训练末步的10/8个FN有所差异。已有数值敏感性仍存在，报告同时保留末步与重放计数，不能把这次重新forward当作额外训练带来的进步。

![对照曲线](comparison.png)
