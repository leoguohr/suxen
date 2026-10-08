# 已换实例：当前监督入口
/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d4_resume3600_gpu0_20260917/SUPERVISION.md
旧端口31548与旧PID1009不再代表活跃任务，先按新入口核验。

# 当前 D4：FPS 索引复用版本
服务器 root@172.16.78.10:31548，hostname 8mddvkd6v007p-0，ControlPath=/tmp/nexus-d4-31548.sock。PID1009/start_ticks337325829。启动身份以 /guohaoran/tmp/vertex_d4_fps_speedup_launch_20260917.json 和本地 launch.json 为准，恢复尚在切换中时不要重复启动。
代码 /tmp/vertex_d4_fps_speedup_20260917/code/scripts/train_vertex_d4.py，入口 SHA256 7db542a26e81f6603baffcc9c59c791f08092f972624043e0973c47d7a0c775d。只缓存固定输入的 FPS 索引，可学习条件特征不缓存。原 /tmp/vertex_d4_resume1200_20260917/data 仍是当前 manifest 引用的数据，不能删除。
输出 /tmp/vertex_d4_fps_speedup_20260917/run，持久 /guohaoran/tmp/vertex_d4_fps_speedup_20260917，console /guohaoran/tmp/vertex_d4_fps_speedup_console_20260917.log；本地 diagnostics/vertex_d4_fps_speedup_20260917/run。禁止下载权重/tmp/partial。
恢复源 /tmp/vertex_d4_resume1200_20260917/run/checkpoint-last.pt，对应持久 /guohaoran/tmp/vertex_d4_resume1200_20260917/checkpoint-last.pt。D4 update2933、累计10333，SHA95a537dc72a4dd2b97606f09fd518d6689dd3d0f0c0a30cb88a4ed5532d9cc65，剩余4267更新。必须完整恢复Adam和RNG；recovery_verification需216份2933开发数组一致，resume_verification需Adam2721张量相等及RNG保留。旧过程已优雅停止，没有丢弃或重放训练更新。新过程从2934开始。
最终日志拼接：diagnostics/vertex_d4_20260916 1..1200 + vertex_d4_resume1200_20260917 1201..2933 + 本次2934..7200。旧原实例1201..1241另存重放记录。额外2933评估只作代码切换数值对照，不增加任何验收门槛。
基准单步4.304414秒。FPS独立约0.081724秒/次，但并行训练会影响分项测量；不能把0.654秒直接当成实测端到端省时。恢复后用完整9更新周期、排除初次热身报告均值，并注明两次训练输入不同、不是严格性能AB。
四对象A000105/8、B000195/52原样；C001045/54、D001885/176，闭合、无非流形/退化，点云GT对齐已检。4个目标两两不同。6对象对独立求最早分叉，此批实测都depth2/8parents，但实现测试还含depth1/9，不能推断写死。原父格比例/seq长度不强加到新对象。
训练D4新增7200步，累计7400+update（最终14600）。lr1e-5 WD0 clip1 accum8 BF16，联合VecSet/DiT，固定条件，每micro独立t/noise；k=((update-1)*8+micro)%36，mesh=k//9，depth=k%9+1。每物体每层1600 micro。不得自行调整预算/LR/模型/阈值。非有限/明确实现错误先保存证据并停止排错，不自动更换超参。
开发29000000..29000003，每400更新4×4=16树；final30000000..30000015仅训练7200后冻权重一次，16×4=64树，full_match_matrix每组必须4×4单位矩阵。共同父格每种子6对，开发24、终验96对，分别报告每对depth、2×2匹配。连续误差/F1/阈值余量均诊断，不新增硬门槛。20Euler/.5，无GTparents注入/强制非空/topk；保留逐层数组及最早失配。容量4096中止计失败。
每次同步JSON/JSONL/NPZ与console到本地；开发有实质改善、异常、完成才通知，其余安静；用户主动问则报当前已确认状态。时间预估基于实测，不以loss短期波动改方案。若再换实例，先看持久checkpoint/ledger，从D4对应update恢复剩余预算，代码支持--resume-update/--resume-evaluation/--resume-training-log；旧未保存的重放记录保留。不得把开始过的终验种子称未见。
完成后核对training_complete、checkpoint_verified update7200/cumulative14600、final_evaluation_complete及result。离线复算64树与96对象对，核对自生成父格、实际同噪声、4×4矩阵、坐标、count/FP/FN/MSE/F1/阈值余量，合并有效日志；打包完整评估证据/运行代码/四份真实条件GT/协议来源/测试/代码哈希，不含权重。暂停本监督（工具若报不存在，不能声称暂停成功）。不自动D10/20、不同时重新采样点云；后续由用户决定。

切换已完成：resume_verification/recovery_verification通过；2941..2958平均3.652144秒/step，基准4.304414秒。下一次正常开发评估3200；2933是已见恢复对照。
