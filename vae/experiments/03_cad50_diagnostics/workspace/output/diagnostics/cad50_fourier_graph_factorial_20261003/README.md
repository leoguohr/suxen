# CAD50：Fourier与Graph/LN顺序交叉对照

范围暂按用户上文的两项问题：只改变坐标Fourier开关及Graph/LN顺序。
不把Decoder FFN、Graph/Encoder激活或neighbor bias一起归入结构变量。

| | LN在Graph前 | LN在Graph后 |
|---|---|---|
| 有Fourier | 复用Graph_LN_pre，2000更新 | 复用V2_control，2000更新 |
| 仅XYZ | 新增XYZ_LN_pre，2000更新 | 新增XYZ_LN_post，2000更新 |

原完整CAD50、seed0、共同随机初始化、fresh AdamW、Hard4、五mesh累积、LR/warmup、
负例规则、mu路径与math00均继承前一项实验。原100条与冻结VAE不参与更新。
每条CAD直接参与200次。不是每条参与2000次，也不将分支预算相加。

## 输入配对

两类输入保留相同39维投影权重、bias和所有其他初值。
有Fourier输入为原39维；仅XYZ输入为XYZ与36个零拼接，等价于只用投影的前三列。
因此避免Linear(3,512)重新构造造成fan-in初始化尺度及后续RNG变化。
仅XYZ分支的36列不贡献前向、loss梯度为零；AdamW的weight decay仍按原配方处理。
名义参数数相同，实际参与输入计算的列数不同，须明确报告。
此实验测量去除Fourier特征的整体效果，包含输入表示与尺度变化，不独立隔离频率与尺度。

## 复用门槛

必须核对原initial.pt SHA、环境、数据哈希，并在当前GPU复现两种有Fourier结构的
step0全部50条Edge与实际Face结果。输入路径不同属于迁移问题，不能静默替换数据。
若门槛未通过，先报告；不能直接拿不同环境的结果作严格配对。
step0一致仍不能保证不同物理GPU上数千步轨迹逐位一致。

## 执行与判断

先运行test_factorial_cpu.py，再运行verify_controls.py；主训练从相同initial.pt独立开始。
训练入口仅允许两个新增分支；完整评价0/500/1000/1500/2000，并保存model/AdamW/RNG。
最终比较实际Edge/Face FP/FN、micro-F1、严格成功UID和66—274顶点组。

令Q为同一评价指标，编码收益分别为Q(有Fourier,后置)-Q(仅XYZ,后置)及
Q(有Fourier,前置)-Q(仅XYZ,前置)；顺序收益也须在两种编码下分别比较。
两组差值不一致说明存在交互，不能强行指定唯一根因。

这是CAD50、一个种子、有限预算的局部因果实验。结果可筛选主要因素，不能直接宣称
原固定100条历史瓶颈已获得唯一根因；必要时再在原100条上确认。

本文件写入时训练尚未执行；实际状态以CPU_TESTS.json、CONTROL_GATE.json和runs各支日志为准。
