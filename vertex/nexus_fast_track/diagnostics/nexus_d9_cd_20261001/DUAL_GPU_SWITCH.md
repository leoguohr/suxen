# D 双卡续训记录

用户要求两张卡共同运行 D。只改变执行并行方式，保留 D 的8个事件/更新、UID-major调度、velocity MSE、Adam lr=1e-4、coupled weight_decay=0.01、clip=1，以及2000更新总预算。

- 安全停止点：D update1550，累计step23550；完整checkpoint SHA256 `0c9a5d642919721893195d85eb87ec2b812be15493711b3052dd2c710d081036`。
- 原断点已在服务器 `experiment/training/audit/dual_gpu_boundary_checkpoint/checkpoint-023550.pt` 建立硬链接，避免被分支轮换清理。
- 新入口 `train_d_dualgpu.py` SHA256 `965476b19b167cbac1b1fc1a7dcf3f95db63d6aa13c6aff76a310eadec2ab86c`。
- 物理GPU1为primary，物理GPU0为secondary。两卡分别重建固定点云的FPS/Fourier缓存，学习到的VecSet特征仍每次前向计算。
- 相邻micro成对并发，主卡梯度按0→7累加；所有8份完成后统一裁剪、一次Adam、同步副卡参数。仅主卡保存权重与Adam；保存另含secondary RNG。
- 从原断点重新执行1551，测速更新均丢弃，正式剩余450更新。C已完成的评估复用；D训练和最终评估完成后再统一打包ABCD。

## 测速与验证边界

初版4+4统一归约的10步测速为2.1914→1.4307秒/更新，但未采用。当前逐对归约版的10步测速（前2步暖身）为2.0321→1.6505秒/更新，速度比1.2312；计时包含梯度传输、参数同步和Adam，不包含保存checkpoint、固定探针或终评。

两轮10更新轨迹比较均未通过最初的数值门槛，这些报告的 `numerically_equivalent=false` 保持原样。随后发现两次原单卡基线相互比较也超过相同norm门槛（最大相对差约3.58e-4，大于1e-4）；因此该多步轨迹检查不能单独区分并行实现问题与基线本身的数值非确定性。未将原因唯一归因于BF16。

基于上述对照，事后修正验证方法为“同一完整状态的一次更新”，而非放宽原参数/norm阈值。新增的Adam moment相对L2阈值1e-5在该检查运行前固定。独立 `verify_dual_one_update.py` 实际GPU检查通过：

| 检查 | 实测 |
| --- | --- |
| 8个事件、UID、depth、noise hash、time | 相同 |
| 参数组、907份Adam状态字段、更新后step | 相同；均23551 |
| 参数最大绝对差 | 3.5169e-7（阈值1e-5） |
| 参数全模型relative L2 | 1.6637e-8（阈值1e-6） |
| clip前梯度norm相对差 | 1.0251e-7（阈值1e-4） |
| Adam一阶/二阶moment relative L2 | 6.9120e-7 / 8.1103e-10（阈值1e-5） |
| 参数/moments finite、副卡参数同步 | 通过 |

逐张量误差、事件和参数组保存于 `audit/dual_gpu_one_update_verify.json`，SHA256 `2c928d5fc1917c719baf3cb2285a81c481574d58fcd684351c80087e6f47fabf`。原始测速、单卡重复对照及v1代码均保留。

该验证确认一次更新的一致性达到既定容差，不声称后续训练轨迹逐位一致，也不代表完整树生成成功。双卡正式训练与后续评估的当前状态见 `experiment/training/audit/dual_gpu_dispatch_started.json`、`phase.json` 和 `D/train.jsonl`；本记录不宣布2000步或最终评估已完成。
