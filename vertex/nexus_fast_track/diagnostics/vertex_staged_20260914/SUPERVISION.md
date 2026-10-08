> 已被用户新计划替代。当前监督入口：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_a100_b1_20260914/SUPERVISION.md 。不要再启动旧诊断。

# 当前实验监督状态

用户最新要求：先交付当前A/B复核包（已生成并校验），后续仍按A–D门槛推进；不要模型权重。失败阶段先排查，不跳关。不要凭现象宣称学习率是根因。

- 本地目录：`/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_staged_20260914`
- 源码目录：`/Users/luthier/Documents/sophomore/nexus_fast_track/mini_nexus`
- SSH：`root@172.16.78.10:31548`，ControlPath `/tmp/nexus-vertex-31548-sep14.sock`。不要保存/打印密码。
- 当前主机：`7kmmnksisffhn-0`；A100 80GB。
- 正式运行目录：`/tmp/vertex_staged_20260914`。共享目录 `/guohaoran` 的 mkdir 出现阻塞，本次正式实验在服务器本地盘运行。
- 首轮正式训练已退出，原 PID `2154`，`/proc/2154/stat` 的 start_ticks `314451776`。先核验身份再操作。
- 启动记录：远端 `launch.json`，训练状态 `run/status.json`，`console.log`，`run/A` 和 `run/B`，最终 `run/result.json`。
- 环境：`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /guohaoran/envs/nexus-algo/bin/python`。解决此新容器旧 ONNX/protobuf 导入冲突，不改模型或依赖版本。
- 完整模型 2,332,430,344 参数，FP32 参数/优化器，BF16 autocast，checkpoint，AdamW weight_decay=0，LR仍1e-4，最初100步warmup，clip=1。固定8192x6实际点云，无增强。
- 首次流水线预算 A1000/B2000/C3000/D3000，每25步评估；任何阶段不通过立即停止后续阶段。模型从头初始化，A之后连续优化同一模型。
- A 已在150/175/200步连续通过，200步MSE=1.2576074368553236e-5，占据和全部8个整数坐标完全正确。
- B已完成2000步且未通过，总更新2200。最终20个holdout只有5个精确，均值MSE=0.8954436868；4个纯噪声采样得到26/20/17/23点，坐标集合均错。C/D没有启动。run/result.json已下载并校验完整A200/B2000日志；B确实使用2000种时间和2000份噪声。
- A权重已留在服务器 `checkpoints/stage_A_step200_model.pt`，SHA256 `1c39eee356fe352809abf6dabc81d585d35310128fc958a8ee7a36c7f31d74dc`。用户明确不要下载权重，已取消大文件下载。最终状态 `run/checkpoint-last.pt` 包含optimizer，保留服务器即可。
- 如果B到预算仍失败：保存完整证据、保持C/D未运行，继续在B内做有针对性的诊断。不要修改运行中的冻结代码。任何诊断条件更简化的实验都不能冒充B通过。不要盲目扩数据或改学习率。
- 已完成：21个GT oracle案例（20真实+1非对称），2268次单层检查，252次完整八叉树恢复，所有坐标完全一致；oracle非模型成绩。
- 20个实际样本已下载，280个数据/源文件SHA验证；点云/法向采样可bitwise重现；实际父子标签与独立XYZ编号实现一致；原保存GT与现数据精确一致。量化需沿用预处理float64，不能用FP32重新量化替换冻结GT。
- 本地和服务器相关25个测试均通过。源码快照在本地/远端 `code/`；本地 `code_sha256.json`，源数据 `data_source_sha256.json`。
- 最终包要包含 code、data、preprocessing_source、run日志/评估/NPZ（排除所有.pt和.tmp）、oracle、alignment、plot/结果说明、previous_overfit20、SHA256清单，不含传输中间tar包。用户要求几个阶段结果齐后再一起压缩。

## 2026-09-14 22:06 后的诊断进展（当前活动）

SSH复用连接曾断开，已用用户本轮更新的认证成功重连，同一主机。没有停止仍在训练的B；B是按门槛预算自然退出。

只读脚本 `diagnose_stage_b.py` 已执行并下载结果 `stage_B_forward_diagnostics.json`：
- A权重重新前向，原固定输入BF16 MSE 1.2576e-5，FP32 8.3418e-6；同t=.5换两份噪声MSE约1.80/2.06。这不否定A验收，A只测固定回归。
- B权重FP32/BF16误差接近；新噪声t=.5 MSE约0.763/1.184。仅改推理精度未解决问题，不能据此排除BF16训练历史影响。

当前活动为一对独立诊断训练，**不能算正式B通过**：
- PID `2949`，start_ticks `314699317`，主机不变。
- 远端脚本 `/tmp/vertex_staged_20260914/diagnose_noise_time_control.py`，本地同名脚本已保存。
- 启动身份 `/tmp/vertex_staged_20260914/noise_time_controls_launch.json`；日志 `noise_time_controls.log`；状态 `noise_time_controls/status.json`。
- 按 fixed_time / random_time 顺序各500步。两组均从同一A200权重开始，均重建相同AdamW（LR1e-4, WD0, clip1），同种子9142601、相同噪声序列；唯一组间变化是t固定.5或随机。两组都会消耗随机t的RNG，结束后核验噪声hash和drawn_time序列完全一致。
- 注意两组都重建了优化器，不能把random_time组说成对原B优化器轨迹的精确重放。
- 每50步4种噪声×t=.1/.5/.9评估，观察t=.5上能否对新噪声拟合。简化对照的成功不能解锁C/D。
- 不要重复启动任何训练；等该对照完成，下载小型日志/评估，排除所有.pt，不提前压缩交付，再根据组间证据决定如何排查正式B。

## 当前包已按用户最新要求生成

`last_delivered_package.json`记录本次当前阶段快照。A200步和B2000步日志完整；C/D未启动；配对诊断抓取于fixed_time第225步，尚未完成。这次只要求打包，没有要求停止正在进行的诊断；继续监督但不要重复发送同一包。
