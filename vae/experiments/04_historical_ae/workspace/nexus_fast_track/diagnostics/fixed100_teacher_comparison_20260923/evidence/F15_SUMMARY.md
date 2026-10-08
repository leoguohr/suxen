# A500最后两块/三块可训练性对照结果

两支各新增500次全100条累积更新，预算已关闭。CPU逐文件/状态/预测计数验收通过；没有后继训练。

| 分支 | 最终联合严格成功 | Edge FP/FN | 实际Face FP/FN | 丢失A500成功UID数 |
|---|---:|---:|---:|---:|
| Control_last2 | 73/100 | 80606/0 | 5277/155 | 0 |
| Treatment_last3 | 73/100 | 78865/0 | 5163/149 | 0 |

预注册Treatment首要判据：False。分支进展（>72）：True；刷新历史纪录（>74）：False；主目标（同一checkpoint100/100）：False。

本实验只检验额外开放已有block13的效用；无论结果如何，都不能单独证明整个架构容量上限或冻结是唯一根因。

## 全程Edge证据与Face采样边界

- Control_last2：完整覆盖更新后状态0..500；最低Edge成功72/100；曾丢失A500 Edge成功UID：[]；发生步骤：[]。
- Treatment_last3：完整覆盖更新后状态0..500；最低Edge成功72/100；曾丢失A500 Edge成功UID：[]；发生步骤：[]。

完整Face仅在0/100/200/300/400/500验收；没有声称检查点之间Face全程保持。每步参数实际位移/clip/逐UID Edge见step_records.jsonl、updates.jsonl与edge_every_step.csv。

## 执行边界

新网络层数始终16；Control开放14/15，Treatment多开放13；旧Adam组/四类RNG继承，新增组单独空state；原数据/pool/目标/后端/评分不变。0步对两支全部100条验证完整/缓存前向及梯度逐位一致；父A500预测数组也逐项一致。

预检曾因新缓存导出包装中的no_grad与现有math00日志上下文不兼容失败；当时更新0，失败日志保留。仅使新包装恢复原grad模式后重试，未改模型/后端/损失。

所有结果以独立checkpoint为单位，不拼UID、不延长预算；源权重和数据只读。精确来源、argv、环境、GPU身份与运行日志保留在repro_outputs和源码manifest中。

## 已审计的容器中断与恢复

原容器在Control完成204次更新后消失；原runtime的进程已不存在，状态按skill恢复规则标为interrupted，原状态和日志已另存不可变快照。GPU UUID、源码与Python/torch/CUDA/cuDNN/依赖版本保持一致。

只读CPU检查确认latest.pt为204更新，Adam旧组2204/block14组1204；跨容器第200点全部100条预测数组逐项一致，第204点全100条完整/缓存前向及梯度逐位一致。两项恢复校验optimizer更新0。

显式从204接续Control剩余296次，Treatment仍独立从共同A500执行500次。旧日志字节前缀保持不变，最终每支0..500状态/1..500更新连续；没有重跑204次或叠加预算。恢复命令和retry_of谱系见recovery_204与_runtime/train。
