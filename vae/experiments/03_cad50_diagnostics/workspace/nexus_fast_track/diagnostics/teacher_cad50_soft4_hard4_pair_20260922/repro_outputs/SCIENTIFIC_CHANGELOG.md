# 科学定义改动

唯一训练干预是Edge和Face重建loss：fully-differentiable Soft4 → TP/TN/FP/FN硬四组BCE。硬组由logit.detach()>0与GT确定，BCE及各组分子保持可微。整条mesh跨chunk汇总N/C，各组N/max(C,1)，四组之和固定除4。空组0是用户明确指定的本轮约定。

没有改动数据、样本权重、Face pool、网络、评分scale/符号/阈值、中心化、Adam历史、LR、clip、更新频率、μ路径、sampling/KL、logvar冻结、math00/FP32 MATH与重计算上下文。H继承相同B2500完整状态；S复用已完成的合格100步对照。

本实验不是Soft4-stopgrad对照，不把Hard4的改变归因于单独删除Soft4第二项。100步不是充分收敛预算；继承旧Adam下的方向结果不代表随机初始化训练上限。
