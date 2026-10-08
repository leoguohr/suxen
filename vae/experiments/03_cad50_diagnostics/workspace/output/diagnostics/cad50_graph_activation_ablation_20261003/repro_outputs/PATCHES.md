# 独立实验补丁

分支：repro/2026-10-03-graph-activation
代码提交：8ba2ad2

在新建独立目录复制两份固定源码。data_objective.py与历史版本SHA一致。native_models.py增加三个显式操作开关，参数名称/形状不变；默认完整V2与历史版本输出/梯度逐位一致。新建独立训练、保存、完整枚举评价和冷加载入口；没有覆盖或修改历史实验。

架构开关属于用户明确授权的科学改动；不称为无语义变化的复现修复。CPU合成验证见CPU_TESTS.json，最小结构diff见native_models.diff。
