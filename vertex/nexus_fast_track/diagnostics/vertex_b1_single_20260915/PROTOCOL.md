# B1_single_mesh_fixed_t：独立R0/R1

用户2026-09-15最新方案替代旧A100先行门槛。本次只回答同一个8顶点物体能否在不同噪声下稳定恢复，不运行A100、不恢复A权重、不恢复随机时间或完整八叉树生成。

- 训练清单、循环实际消费和验收UID全部仅nexus_2k_000105。入口、循环、评估均断言；每步记录UID，报告汇总实际已消费UID及更新数。完整模型2,332,430,344参数，depth9、GT parents、固定实际8192×6点云、0/1占据标签、t=.5，无增强。
- R0/R1各fresh model与fresh AdamW，seed20260914，不加载检查点。R0原初始化；R1只改depth std=.02、time MLP两层std=.02/bias0、cross-attention输出weight/bias0，其余初始化一致。
- LR1e-5，WD0，warmup100，clip1，BF16 autocast+FP32参数和优化器。每update8份新独立噪声，各loss/8反向后更新一次。R0/R1训练noise序列相同。
- 每组500update、每50评估；仅最近3次验证严格改善且100update内mean MSE降低至少10%，延长一次到1000。预算未过就保存结果并结束该组；不自行改LR/loss/阈值/结构。

## 同一份evaluation中的三种证据

1. training_cache_probe：第1次update第0个microbatch实际用到的noise、xt、target_velocity、GT parents、target occupancy保存为training_cache.npz。metadata保存张量SHA，与train.jsonl第一份noise SHA可核对。之后直接读取缓存xt/velocity来评估，不依赖重建seed。第0步尚未训练，故该项null，不能宣称已见输入。
2. probes：同一UID16份固定验证噪声（8000000..8000015），每例MSE、TP/FP/FN、顶点数、精确坐标和实际全空基线，另存预测数组。缓存输入不是这些验证输入。
3. 若候选16例均MSE<=.01且坐标全对，且4份FP32扰动响应signed gain在[-2.2,-1.8]、relative error<=.1，再仅一次使用64份独立holdout（9000000..9000063），逐例同样MSE/坐标/FP32响应门槛。使用后无论结果如何都结束该组，不反复在holdout调参。

报告内嵌实际训练配置与scope、初始化R0/R1、缓存来源及指纹。responses_fp32包含输入/首尾及全部block RMS、每token top1/top8能量占比。train.jsonl逐步记录实际LR、输入投影/最终输出层裁剪前后梯度范数，以及优化器真正更新后的参数变化范数。零初始参数的relative_update_norm使用极小分母，只是诊断数值；应优先解读绝对update_norm和parameter_norm_before。

空预测器v=-2xt的实际MSE与4p并列，同时显示它预测0个点，不算形状恢复。cached输入能否拟合、验证噪声能否恢复、方向响应是否正确分别解释，不能从单个平均MSE宣布根因。B1通过也不代表全时间或完整生成通过。

## 文件与运行

冻结代码来自上一轮服务器快照，只替换B1入口和新增诊断helper及相应测试，不混入本地其他Topology修改。部署64个源码/数据文件SHA全部核验。用户不要下载权重，报告/JSONL/NPZ按监督周期下载。

实际服务器目录/tmp/vertex_b1_single_20260915；上一轮共享盘写入阻塞，不宣称权重持久化。实例被替换仍可能丢失权重，因此每次监督同步小型证据到本地本目录。不要与旧A100证据混用。

## 新服务器CPU运行环境检查

默认环境两次各33/34通过，同一个旧overfit CPU测试子进程signal11；本轮B1测试两次都通过。限制OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1后全34通过。因此正式启动使用这3个进程级环境变量，训练脚本仍显式torch.set_num_threads(8)，不修改模型或训练协议。该现象提示线程环境相关性，未确认底层根因。保留所有测试日志。
