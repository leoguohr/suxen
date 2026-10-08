# 可比性与结论边界

共同父为 CAD B2500 完整 model/Adam/RNG，SHA256 为 `4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`。本轮两台服务器均重新读取并哈希核对。原 Soft4 对照已有100步合格结果，复用它，本轮不重复训练。

`CONTROL_REUSE_AUDIT.json` 重新核验旧 Control 的起点模型、Adam 参数名映射与状态、RNG、数据/pool、有效代码、100条更新日志、五次完整评价及250个预测数组。它是本次事后核验，不能冒称旧实验当时新增了日志。

H 启动时，新增模块后的全50初始预测与父点数组逐字节一致；真实同GPU未扩展网络与扩展网络的旧参数裁剪前梯度一致。UID20 的 loss=0.84434574842453、gradient norm=6.520879316263456、梯度SHA256=`d445214cae052baa2d8979b29314451971110649dd876f6409f2d12e32f47955`，也与旧Control启动记录一致。`release.json` 记录核验后放行100步。

唯一结构干预是最后attention残差后、原decoder_output_norm前新增pre-LN FFN残差，1024→4096→GELU→1024，共8,395,776参数。最后Linear权重和bias为0，实现初始恒等。新增参数采用独立fresh Adam组，LR3e-5；旧参数完整继承Adam。这是新增模块无法具有历史moment的明确实验约定。

两支仍训练完整旧Encoder/μ/16层Decoder和原共享head；同一全50累积更新、Soft4、数据/pool、阈值、数值路径、学习率与global clip。新FFN梯度也参加global clip=1，因此旧参数第一次裁剪后位移不保证与Control相同；这属于干预真实影响，不应通过排除新梯度人为消除。

比较100步后的实际Face F1、错误数、困难16条、严格成功集合和轨迹，不按训练loss排名。新增FFN同时增加非线性路径、参数和优化自由度，单个对照不能把收益唯一归因于“缺FFN”，也不能证明全部16层都应加FFN。100步只检验从此父点短期续训方向，不是充分收敛上限。

本轮CAD50与原固定100条不同。CAD50达到0.997才达到本阶段Face F1门槛；严格50/50另报。本轮任何改善不能写成原固定100条已通关。原100和其他旧任务没有训练更新。

主README中的“CPU小测试不调用optimizer”表述不准确：合成toy模型做过2次Adam更新，用于梯度与恢复测试；真实CAD父模型在放行前始终0更新。精确记录见 `H_terminal_ffn/terminal_ffn_test.json` 和 `startup_gate.json`。此处保留启动README原件并说明更正。
