# 可比较性

相同初始parameter/buffer张量、fresh AdamW、同seed/epoch/UID调度和负例、同参数量246575680、同预算2000更新。每条CAD直接参与200次。全部Decoder FFN与Fourier保留。

四支只有预先指定操作不同。各检查点使用同一个共享模型，真实前向，完整预测Edge三角枚举，threshold>0，不混合checkpoint。单种子结果不能宣称全局唯一根因。
