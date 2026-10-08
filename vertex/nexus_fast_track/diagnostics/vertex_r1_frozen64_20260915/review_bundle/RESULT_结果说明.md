# 冻结R1第1000步：64独立噪声几何/回归全部通过

- geometry_regression_passed: true。64/64完整整数坐标集合正确；平均MSE0.0015148376，最差MSE0.0066605806，每例都<=.01。
- local_response_diagnostic_passed: false。9000006、9000027、9000060三例超原诊断门槛，最大响应误差0.1958486。
- strict_passed_on_this_followup: false。未修改原阈值，也未改写旧严格B1的passed=false。

本次固定原R1 step1000模型，未更新参数。checkpoint SHA a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279，已备份到服务器持久目录并回读校验；权重不在包中。

本地复算64份几何数组、68份FP32原始扰动数组，指标一致；原8000000..8000003四个探针复现一致。evaluation/arrays保存x、delta_x、velocity_before/after等，可独立复算。verification.json记录复算结果，local_tests.txt仅代表小模型CPU入口测试。

这64种子9000000..9000063此前未使用，本次之后已消费，不再作为未见终验集。当前验证只有同一8顶点物体、depth9、GT parents、t=.5，不能外推随机时间、纯噪声ODE、全树或多物体。

按用户新协议，几何/回归通过允许推进B2，并保留局部响应诊断标记。B2已另开实验，从同一模型/Adam/RNG继续随机t，预算1000新增更新，未重新训练A或扩大mesh。本包仅包含冻结64评估，B2结果另存。

主要文件：evaluation/report.json为完整报告；evaluation/arrays为132份几何/响应NPZ；evaluate_vertex_b1_frozen.py为只读入口；code与data为原冻结源码和实际数据；original_strict_B1_result.json保留旧结果。
