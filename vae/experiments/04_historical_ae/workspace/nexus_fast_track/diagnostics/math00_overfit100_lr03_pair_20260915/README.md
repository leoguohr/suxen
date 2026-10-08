# 固定100条，epoch400工作点的LR对照

两支独立从`math00_overfit100_continue200ep_20260915_v2/run/checkpoint-update10000.pt`开始。父SHA256：`0804e7dabccd5dfd027215be4a2a8c98281aea8ffd148090913b658209371d14`。

| 分支 | E/μ LR | Decoder/Edge/Face heads LR | 新预算 |
|---|---|---|---|
| A_hold | 1e-5 | 1e-4 | 100epochs / 2500updates |
| B_lr03 | 3e-6 | 3e-5 | 100epochs / 2500updates |

每支均为epoch401–500、update10001–12500，每条mesh新增参与100次、累计500次。两个分支的2500步不能相加成一个模型的训练量。A保留父LR的实际浮点值，B在完整加载Adam后乘0.3；不清空任何矩状态，不重新warmup。

## 保持与配对

父权重、四组Adam、Python/NumPy/Torch/CUDA及shuffle RNG完整恢复。100条UID与固定池不变。每条完整mesh作为一个微批，四条loss/4累积后clip=1并统一更新。μ路径、KL=0、logvar冻结、不进入Adam；fully-diff Edge+Face Soft4、math00、评分/阈值、归约、候选池、wd=0不变。

`paired_epoch_orders.json`从父shuffle状态预先生成未来100个epoch顺序；它不消耗训练进程RNG。A、B每个epoch仍从各自恢复的shuffle流生成顺序，并与这份共同计划逐项断言一致。每次验收保存/恢复所有RNG。

起点先验证模型及Adam张量与父状态一致，再应用唯一有意的LR改动；额外核验除了LR之外优化器全状态不变。epoch400全量重建必须复现父末尾100条的loss、实际Edge/Face计数、候选覆盖和最小margin，否则不更新。

## 新服务器执行安排

当前只有一张A100 80GB。驱动先做B的零更新完整起点预检，再顺序训练A、B。B不会继承A的训练结果；正式开始时重新读取同一父checkpoint，并再次复核起点。

前一阶段已完成，无需重新训练。新目录独立保存两支状态。驱动或分支锁防止重复启动，已有更新日志时拒绝覆盖。出现执行错误则记录并停止队列；不会改LR或自行补跑新预算。

## 验收与记录

epoch400、425、450、475、500：暂停更新，同一checkpoint逐条完整评价100条。实际Face始终从当前预测Edge图枚举，未入候选的GT Face计FN。原30秒/mesh枚举完整性规则保留，未完整FP/F1为null，不能算严格成功。

逐mesh记录Edge/Face TP/FP/FN、严格成功UID、最小margin、GT Face候选覆盖、训练池之外实际Face FP。训练日志保留loss、各组梯度/clip、实际参数位移及表示尺度。完整checkpoint每10epoch及规定验收点保存。

主判断比较后期多个检查点的实际错误、严格成功身份和漏面分解，而非单看训练pool loss、曲线平滑或clip频率。两支完成后自动生成`pair_evaluation_package.zip`，包含原始日志、代码、配对顺序、配置/恢复核验、逐mesh对照和曲线。大checkpoint及数据数组留在服务器，以路径/已有SHA定位。

运行入口：`bash /guohaoran/nexus_fast_track/diagnostics/math00_overfit100_lr03_pair_20260915/run.sh`。
状态：根目录`pair_status.json`，分支`A_hold/run/status.json`、`B_lr03/run/status.json`。
