# Small-only：Original / Fixed4 / Soft4 / GT-balanced

Original直接引用上轮small-only；旧Fixed4和GT-balanced实际是双mesh，本次补齐small-only，未重跑旧实验。四组来自同一initial.pt，E LR=1e-5、D LR=1e-4、μ mode、Face=KL=wd=0、clip=1、fresh Adam，保留相同Flash后端。

Soft4：τ=1、ε=1e-8、sigmoid membership detach、四组分子分母均用FP32全顶点对求和，再分别归一化，固定除4。所有组每步用logit>0计算同一硬指标。

|训练|最终F1|最后200步F1 最小/中位/最大|最终GT-balanced BCE|训练objective|TP/FP/FN|
|---|---:|---|---:|---:|---|
|Original four-group|0.0000%|0.000% / 0.000% / 10.114%|0.4053928|0.4053911|0/0/1152|
|Fixed4|0.0000%|0.000% / 0.000% / 0.000%|0.4045195|0.2022589|0/0/1152|
|Soft4|99.1251%|93.036% / 97.378% / 99.345%|0.03191508|0.1813167|1133/1/19|
|GT-balanced|86.5660%|71.016% / 82.368% / 90.743%|0.02282153|0.02282153|1147/351/5|

训练objective定义不同，不能按其绝对数值直接排名；统一BCE和硬F1才是横向比较口径。

## TP出现又消失的事件

下表从每条硬loss轨迹的完整事件中选取birth objective上升最大的3次，完整事件与窗口在tp_events.json。Soft4列是相同步数的变化，不是相同参数下的反事实。

|训练|TP birth → return|birth objective变化|return objective变化|Soft4相同步数birth变化|
|---|---|---:|---:|---:|
|Original four-group|173 → 175|+0.1447124|-0.1407934|-0.0034446|
|Original four-group|974 → 976|+0.1439138|-0.1437789|-0.0003170|
|Original four-group|978 → 980|+0.0970987|-0.1536183|+0.0128017|
|Fixed4|112 → 115|+0.1725466|-0.3458836|+0.0000332|
|Fixed4|448 → 451|+0.1729053|-0.3499169|-0.0034999|
|Fixed4|840 → 850|+0.1727528|-0.1684974|-0.0007025|

Soft4自身的所有TP出现事件：

|birth|出现的TP数|前一步objective|本步objective|变化|再次TP=0|
|---:|---:|---:|---:|---:|---|
|132|45|0.48123360|0.46751666|-0.01371694|133|
|137|11|0.47722965|0.47731841|+0.00008875|之后未出现|

各组自身TP从0出现的次数：Original four-group=9, Fixed4=5, Soft4=2, GT-balanced=2。

Soft4没有硬预测分组开关，因此不存在由logit恰好跨0触发的离散分组重定义。但相邻训练步仍可能有较大变化：参数更新、相对很小的soft组质量以及stop-gradient重算权重都需区分。不能把时间序列的每个跳变都归于分组开关，也不能认为连续就必然稳定收敛。

stop-gradient意味着反传把本次membership视为常量；下一次forward仍会重算membership。其反传方向不是把membership也求导后的完整标量函数梯度。

## 结论边界

同一初始化和网络下，Original/Fixed4最终为空边图，Soft4最终达到99.13%硬F1，说明修改训练loss足以显著改变当前重建结果；不支持把失败简单归结为encoder或decoder容量不足。Soft4最后200步F1仍在93.04%–99.34%之间，没有一次严格达到100%，尚未完全overfit。

GT-balanced最终BCE更低（0.02282 vs Soft4的0.03192），却有351条错边，Soft4只有1条错边。这再次说明较低的平均BCE不自动等于更好的零阈值重建。

Soft4既取消硬组出现/消失，也改变了组内样本权重，因此本实验没有单独隔离“不连续性”的全部因果贡献。证据强烈指向loss分组/加权设计是当前训练的重要障碍，不能宣布已证明唯一根因。

每200步checkpoint和embedding保留在服务器本目录；本地保存完整逐步日志、对照图、事件窗口与核验记录。

![四组对照](comparison.png)

![TP事件窗口](tp_events.png)
