# 当前D4：端口31248，仅物理GPU0
服务器root@172.16.78.10:31248，hostname6u3rbffr0rqe3-0，ControlPath=/tmp/nexus-d4-31248.sock。PID751/start_ticks337987377，启动记录/guohaoran/tmp/vertex_d4_resume3600_gpu0_launch_20260917.json。
用户明确要求只用GPU0。CUDA_VISIBLE_DEVICES绑定物理GPU0的UUID GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d，PyTorch应只看到1张卡；GPU1 UUID GPU-3534263c-6584-9f34-9273-0ef6a7852fbe，禁止在GPU1启动本任务任何训练或测试。
代码/tmp/vertex_d4_resume3600_gpu0_20260917/code，入口scripts/train_vertex_d4.py SHA7db542a26e81f6603baffcc9c59c791f08092f972624043e0973c47d7a0c775d，维持FPS索引复用；70个代码文件与16份输入数据已核对SHA。当前manifest和selection在/tmp/vertex_d4_resume3600_gpu0_20260917，实际数据在该目录data，已重建并更新绝对路径。
输出/tmp/vertex_d4_resume3600_gpu0_20260917/run；持久/guohaoran/tmp/vertex_d4_resume3600_gpu0_20260917；console /guohaoran/tmp/vertex_d4_resume3600_gpu0_console_20260917.log。本地diagnostics/vertex_d4_resume3600_gpu0_20260917/run。不要下载权重/tmp/partial。
恢复源/guohaoran/tmp/vertex_d4_fps_speedup_20260917/checkpoint-last.pt，NVME副本/tmp/vertex_d4_step3600_gpu0.pt，27990403994字节，SHA36f18ca0829d3e23e650e97184a0837f5a94710e1120fcfcd6f5ff73300b99c9，复制和读回SHA均通过。来源D4update3600/累计11000；原总预算7200，剩3600更新。
原实例日志到3867，267步未保存。恢复检查必须先比对216份3600开发数组，再逐步重放3601..3867的UID/depth/time/noise哈希。resume_verification需Adam2721张量一致和CPU/CUDA RNG保留；每步VecSet前向8次且组件更新非零。恢复复核不是新终验，不绕过数值一致性检查。
有效最终日志：原D4 1..1200 + resume1200 1201..2933 + FPS优化版2934..3600 + 本实例3601..7200。原实例1201..1241及FPS版3601..3867都保留为重放前记录，不重复计数。最新已完成3600开发是11/16（A4/4 B4/4 C3/4 D0/4），共同父格24/24；仍未通过D4。
四对象A000105/8、B000195/52原样；C001045/54、D001885/176，闭合、无非流形/退化，点云GT对齐已检。4个目标两两不同。6对象对独立求最早分叉，此批实测都depth2/8parents，但实现测试还含depth1/9，不能推断写死。原父格比例/seq长度不强加到新对象。
训练D4新增7200步，累计7400+update（最终14600）。lr1e-5 WD0 clip1 accum8 BF16，联合VecSet/DiT，固定条件，每micro独立t/noise；k=((update-1)*8+micro)%36，mesh=k//9，depth=k%9+1。每物体每层1600 micro。不得自行调整预算/LR/模型/阈值。非有限/明确实现错误先保存证据并停止排错，不自动更换超参。
开发29000000..29000003，每400更新4×4=16树；final30000000..30000015仅训练7200后冻权重一次，16×4=64树，full_match_matrix每组必须4×4单位矩阵。共同父格每种子6对，开发24、终验96对，分别报告每对depth、2×2匹配。连续误差/F1/阈值余量均诊断，不新增硬门槛。20Euler/.5，无GTparents注入/强制非空/topk；保留逐层数组及最早失配。容量4096中止计失败。
每次同步JSON/JSONL/NPZ与console到本地；开发有实质改善、异常、完成才通知，其余安静；用户主动问则报当前已确认状态。时间预估基于实测，不以loss短期波动改方案。若再换实例，先看持久checkpoint/ledger，从D4对应update恢复剩余预算，代码支持--resume-update/--resume-evaluation/--resume-training-log；旧未保存的重放记录保留。不得把开始过的终验种子称未见。
完成后核对training_complete、checkpoint_verified update7200/cumulative14600、final_evaluation_complete及result。离线复算64树与96对象对，核对自生成父格、实际同噪声、4×4矩阵、坐标、count/FP/FN/MSE/F1/阈值余量，合并有效日志；打包完整评估证据/运行代码/四份真实条件GT/协议来源/测试/代码哈希，不含权重。暂停本监督（工具若报不存在，不能声称暂停成功）。不自动D10/20、不同时重新采样点云；后续由用户决定。

恢复已通过：216份数组一致，Adam2721张量一致、RNG保留。当前已核对重放3601..3628输入一致；尚需继续核对至3867，不要把部分核对说成267步全部完成。GPU0实测占用，GPU1当前0MiB。
