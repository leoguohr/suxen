# 三卡加速：只迁移调度，保留实验语义

用户新增授权两卡服务器32483。实测主机e0e9g8mah9rqb-0，两张A100-SXM4-80GB均无GPU进程、显存0。原单卡31548主机9q2418hi3rc9q-0继续用于本实验。

| 角色 | 设备 | 入口 |
|---|---|---|
| A原版训练 | 双卡GPU0，GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351 | recovery/parallel.py --role A |
| B V2训练 | 双卡GPU1，GPU-e4d3702b-901b-8690-efa2-4f2a7a81f277 | recovery/parallel.py --role B |
| 全50完整评价与最终打包 | 单卡GPU0，GPU-0a371e11-9041-9b22-d2d5-8469622c4681 | recovery/parallel.py --role eval |

交接时A正在单卡执行10001—11000，B已完整保存到10000。只暂停本实验的旧CPU调度器PID401，训练子进程继续。B立即由新独立GPU从10000完整状态续跑；A的新控制器等待旧段完成。评价控制器要求A11000完整保存、RigorPilot run-train成功退出、单卡GPU释放后，才结束旧CPU调度器并发布交接记录。此时A从11000接入新卡，旧单卡开始评价。没有杀掉训练子进程，没有从陈旧checkpoint回滚，没有增加交接专用optimizer更新。

每个分支只在自己的完整评价完成后继续下一段，另一分支可独立训练。完整评价仍是同一原生Encoder→μ→Decoder，预测Edge完整枚举Face候选，logit>0。达标后自动冷重放。训练器、网络、loss、采样器、原数值路径及Adam/RNG恢复入口没有修改；新增的parallel.py仅负责进程调度。

仍遵守RECOVERY_PLAN.json的共同有效上限19356、每分支物理更新不超过20000，以及包括历史消耗的24累计GPU小时。更多设备用于减少等待时间，没有增加实验预算。完整结果自动汇总到repro_outputs/，分包到delivery/，大权重留服务器并提供身份清单。

以repro_outputs/PARALLEL_HANDOFF.json的state=parallel_released确认A实际迁移边界；PARALLEL_A/B/eval_LAUNCH.json记录各主机PID和脚本哈希，PARALLEL_*_STATUS.json记录阶段，各段RESTORE_AUDIT.json记录完整恢复检查。旧RECOVERY_LAUNCH.json是历史调度器记录，不能用它判断新三卡任务的存活状态。
