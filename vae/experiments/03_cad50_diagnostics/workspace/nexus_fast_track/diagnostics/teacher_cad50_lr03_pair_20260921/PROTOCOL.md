# CAD50 step2000：原LR vs 0.3倍LR

这是独立CAD50实验，原固定100条不在本任务范围内。不得停止其他训练、改其他实验代码/权重或占用已经有计算进程的GPU。

共同父checkpoint为teacher_cad50_fresh512_20260921/run/checkpoint-step2000.pt，SHA256为ae7c2835fe7ad9da72edd413f66373b415721b2cfac72df8c6b3fc141b987358。两支分别恢复完整model、四组Adam、RNG及父参与计数；只在恢复后修改目标LR，不清空历史。

| 分支 | encoder_mu | decoder | edge_head | face_head | 新增预算 |
|---|---:|---:|---:|---:|---:|
| A_control_lr1 | 1e-5 | 1e-4 | 1e-4 | 1e-4 | 500次全50条 |
| B_lr03 | 3e-6 | 3e-5 | 3e-5 | 3e-5 | 500次全50条 |

每支末尾各自累计2500步。无warmup、LR调度、目标权重变化或动态Face建池。B不继承A末尾。microbatch=1，每条loss/50，50条固定顺序完整累计后clip=1、Adam一次；每步断言LR未被覆写，Adam step和参与计数正确。

数据、pool通过只读使用父目录文件接入。50个UID、2872顶点、8364边、5576面、235741个Edge pair、13946个Face候选。已有NPZ、manifest及全部依赖生效源码SHA均核对，不重新建池。

完整重建网络可训练，logvar冻结；model.eval()配合autograd保留上一轮真实行为，明确z=mu。math00、FP32 MATH、确定性Graph、原评分/scale/中心化及fully-diff Soft4均复用未改的runtime、loader、evaluate和评分文件。历史construction_args中的precision文本不决定当前已安装的math00实际路径。

每次更新记录的是更新前同一参数状态下50条loss与Edge计数；位移、LR和Adam计数记录对应更新。实际Face只在新增0、50、100、150、200、250、300、350、400、450、500完整验收，不冒充逐步Face结果。

共同起点和A/B末尾导出00、02、03、13、20、24、34、39，共24份只读中间表示。没有蒸馏、表示MSE、隐藏状态训练缓存或输出修复。诊断hook在所有8条核对输出、完整参数梯度和RNG不变。

GPU限定为上一轮CAD使用的GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351。执行gpu_guard.py后才能创建CUDA上下文；检查到该GPU有任何计算进程时直接拒绝启动，不停止、不挤占已有任务。A、B顺序运行，其他GPU不访问。守护检查只核对启动时资源，不能替代集群调度器的独占预留。

无更新预检完成后运行run_pair.sh。它先A，再重新检查GPU、从共同父状态启动B，末尾audit_pair.py做只读核验及打包。发生异常保存现场、停止，不自动重置或新增预算。GPU占用或缺依赖导致未启动时如实报告。

稳定性比较：同500条更新前记录的总loss、裁剪前梯度范数Max/P95/P99；保留总loss最高10步及其UID分解/实际位移。另将超过此前20步中位数2倍的记录列为描述性尖峰，前20步不作该判定。对12/28记录逐项loss，02/24/34记录Edge逐步变化。该统计不参与训练决策。

报告先比较末尾严格覆盖、父30条保持、66—274点组和四项FP/FN，再报告最佳检查点及完整轨迹。曲线更平滑本身不代表解决；单次有限配置失败不代表容量不可能。

## 文件索引

- precheck.json、common/：0次更新的父状态/完整评价和hook核验。
- A_control_lr1/、B_lr03/：各自配置、恢复核验、500步日志、11个完整checkpoint和实际评价。
- partition.json：固定父成功30/失败20，仅用于分析。
- predictions-newXXXX/：实际预测Edge、预测Edge图全部Face候选及logit。
- representations-newXXXX/：指定样本张量、编号、原坐标及形状/位置说明。
- REPORT.md、actual_trend.csv、per_mesh.csv、step_per_uid.csv、watched_uids.csv、stability.json：完成后生成。
- MODEL_ARTIFACTS.json：大模型、optimizer/RNG checkpoint路径、大小及SHA。
- TeacherCAD50_LR1_vs_LR03_500_Review.zip：评估代码、数据/pool、配置、全部日志和报告。
- TeacherCAD50_LR1_vs_LR03_500_Predictions_Representations.zip：两支1100份实际预测和24份指定表示。

以上报告/压缩包为完成后产物，文件实际存在并核验通过前不得宣称已完成。
