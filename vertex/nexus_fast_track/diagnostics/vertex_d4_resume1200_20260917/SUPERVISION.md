# 已切换 FPS 索引复用版本
当前入口：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d4_fps_speedup_20260917/SUPERVISION.md。旧 PID270 已按计划停止，新 PID1009 正在恢复；不要重新启动旧入口。

# 正在执行已授权的 FPS 加速切换
2026-09-17：当前主任务已向原 PID270 发送 SIGTERM，等待它保存完整检查点并完成恢复用评估。不要重复启动原训练。候选代码与小模型CPU/BF16等价性证据在 diagnostics/vertex_d4_fps_speedup_20260917；切换完成后本文件会指向新监督入口。

# 当前 D4：新实例从 update 1200 严格恢复
当前主机8mddvkd6v007p-0，root@172.16.78.10:31548，ControlPath=/tmp/nexus-d4-31548.sock，PID270/start_ticks336368244。入口/tmp/vertex_d4_resume1200_20260917/code/scripts/train_vertex_d4.py，SHA9e90d753238a8ad56219594b37c19ae38c44e9bfea7beb28f0c036d142ac5dc8。先检查主机、进程、代码身份；旧实例状态不是新实例的运行状态。
本地diagnostics/vertex_d4_resume1200_20260917；当前run=/tmp/vertex_d4_resume1200_20260917/run，持久=/guohaoran/tmp/vertex_d4_resume1200_20260917；console=/guohaoran/tmp/vertex_d4_resume1200_console_20260917.log，launch=/guohaoran/tmp/vertex_d4_resume1200_launch_20260917.json。禁止下载权重、tmp、partial。
恢复源/guohaoran/tmp/vertex_d4_20260916/checkpoint-last.pt，NVME副本/tmp/vertex_d4_step1200.pt，两次SHA已核对7318fd6f5bb6ed52d8a67ff1e1e4357497857650e4aef972bd8b0f0963874757，27990388058字节。D4 update1200、累计8600，保留原总预算7200，剩余6000。旧训练日志到1241，1201..1241重放，不能混合重复计数。
启动后先核对resume_verification.json/recovery_verification.json：模型、Adam2721张量、RNG完整恢复，216份开发数组与原1200一致；训练前41更新实际UID/depth/t/noise哈希必须匹配旧输入。失败保存证据排错，不绕过检查。确认训练行中每步8次VecSet前向及非零VecSet/DiT参数更新。有效最终日志为旧1..1200加新1201..7200；旧1201..1241另存标注重放。完整原始日志均保留。
原1200开发：全树8/16（A4/4,B4/4,C0/4,D0/4），共同父格24/24；不是D4通过。当前恢复初始评估只验证一致性。
四对象A000105/8、B000195/52原样；C001045/54、D001885/176，闭合、无非流形/退化，点云GT对齐已检。4个目标两两不同。6对象对独立求最早分叉，此批实测都depth2/8parents，但实现测试还含depth1/9，不能推断写死。原父格比例/seq长度不强加到新对象。
训练D4新增7200步，累计7400+update（最终14600）。lr1e-5 WD0 clip1 accum8 BF16，联合VecSet/DiT，固定条件，每micro独立t/noise；k=((update-1)*8+micro)%36，mesh=k//9，depth=k%9+1。每物体每层1600 micro。不得自行调整预算/LR/模型/阈值。非有限/明确实现错误先保存证据并停止排错，不自动更换超参。
开发29000000..29000003，每400更新4×4=16树；final30000000..30000015仅训练7200后冻权重一次，16×4=64树，full_match_matrix每组必须4×4单位矩阵。共同父格每种子6对，开发24、终验96对，分别报告每对depth、2×2匹配。连续误差/F1/阈值余量均诊断，不新增硬门槛。20Euler/.5，无GTparents注入/强制非空/topk；保留逐层数组及最早失配。容量4096中止计失败。
每次同步JSON/JSONL/NPZ与console到本地；开发有实质改善、异常、完成才通知，其余安静；用户主动问则报当前已确认状态。时间预估基于实测，不以loss短期波动改方案。若再换实例，先看持久checkpoint/ledger，从D4对应update恢复剩余预算，代码支持--resume-update/--resume-evaluation/--resume-training-log；旧未保存的重放记录保留。不得把开始过的终验种子称未见。
完成后核对training_complete、checkpoint_verified update7200/cumulative14600、final_evaluation_complete及result。离线复算64树与96对象对，核对自生成父格、实际同噪声、4×4矩阵、坐标、count/FP/FN/MSE/F1/阈值余量，合并有效日志；打包完整评估证据/运行代码/四份真实条件GT/协议来源/测试/代码哈希，不含权重。暂停本监督（工具若报不存在，不能声称暂停成功）。不自动D10/20、不同时重新采样点云；后续由用户决定。
