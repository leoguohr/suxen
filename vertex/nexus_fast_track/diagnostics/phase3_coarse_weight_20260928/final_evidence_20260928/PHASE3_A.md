# Phase3-A

独立实验，从Phase2累计step12000完整恢复。仅修改层级loss权重：1–5层×2，6–15层×1，统一除以4/3，继续8次梯度累积。训练2000更新，最终step14000；GPU1。

不在训练中评估，保存每micro原始/加权loss、UID/depth/t/noise哈希。每200更新保存完整checkpoint，并额外保存首个更新作为恢复检查；Phase3保留最近两份，Phase2源权重不动。

最终沿用seed94026000–94026003，10seen+2unseen共48条D15完整树，20步Euler/层、阈值0.5、BF16，保持原顺序与每层独立RNG。保存全部逐层预测、完整树/occupancy/首错和两split对比；不因loss下降宣布通过。

完整流程由一次性phase3_pipeline.py顺序执行：训练成功退出→最终权重哈希门禁→48树评估及原条件切换→CPU指标复算→轻量ZIP。失败则记录错误并停止，不自动重试或追加预算。没有创建持续监督任务。

训练入口与评估入口差异见train_git_diff.patch和eval_git_diff.patch。原目录不是Git仓库，因此使用git diff --no-index，不伪造commit。来源路径/哈希与具体协议见EXPERIMENT_CARD.json。

注意：只有加权续训这一个实验，不能把相对Phase2的提升完全归因为加权；其中也包含新增2000更新的影响。
