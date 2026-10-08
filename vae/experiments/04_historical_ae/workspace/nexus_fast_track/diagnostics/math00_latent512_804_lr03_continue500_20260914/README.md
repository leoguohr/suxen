# 804 latent512：B配置累计4500至5000

父checkpoint：`../math00_latent512_804_lr03_continue1000_20260914/checkpoint-update1000.pt`。
父SHA256：`d620530cb3654d08f98204f538e5b9bd754e260ce20a6f79a43776ac0cb3ad1b`。

仅新增500次更新，不修改LR、四组Adam、μ路径、KL=0、logvar冻结、math00、fully-diff Soft4、clip=1及其他设置。
检查并保存：新增0、100、200、300、400、500。

## 运行状态

已完成500步有效续训、累计到5000并停止。权重/Adam/LR/RNG、冻结logvar与四组实际更新核验均通过。
第一次进程在新增71步后随连接中断结束，当时只保存到step0；原输出完整保留在服务器同名目录加`_interrupted_update0071`后缀。恢复后从同一4500父状态重放，前71步除walltime外全部日志精确一致。
实际计算包含丢失的71步，共571次；有效最终轨迹为4500→5000，没有把丢失计算加到训练进度。见recovery_verification.json；interrupted_attempt/updates.jsonl为中断的71步记录。

## 新增错误记录

- errors-updateNNNN.json：当次所有Edge/实际Face FP/FN，局部0基顶点ID、label、logit、signed margin、训练pool归属、实际候选归属。
- candidate-snapshot-updateNNNN.npz：全部无向Edge pair logits、实际Face候选logits、全部GT Face logits及覆盖标记。GT但未进入候选仍计FN，额外评分只用于诊断。
- Face训练pool定义为当前有效目标使用的positive与mixed并集；不使用历史其他pool替代。
- analyze.py在训练结束、取回日志后离线核验错误记录并生成逐ID轨迹，不运行模型、不执行更新。
- 某个非GT Face不再是实际候选时，该点未计算的logit记为空，不能冒充logit降到负数。
- 六个检查点都错误只代表检查点上的持续错误，不声称每个中间训练步都错。

analyze.py已执行，报告与错误轨迹保存在本目录。
