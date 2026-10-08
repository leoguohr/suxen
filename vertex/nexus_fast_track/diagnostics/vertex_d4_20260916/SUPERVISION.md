# 已迁移：当前监督入口
请先读 /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d4_resume1200_20260917/SUPERVISION.md。以下仅为旧实例历史。

# D4监督：已授权启动，不扩D10
当前主机43ag0soirt181-0，root@172.16.78.10:36910，ControlPath=/tmp/nexus-d4.sock，PID3197/start_ticks332524129；入口/tmp/vertex_d4_20260916/code/scripts/train_vertex_d4.py，SHA9e90d753238a8ad56219594b37c19ae38c44e9bfea7beb28f0c036d142ac5dc8。先核验进程和代码身份；连接失效先按任务最近凭据重连，不因一次ControlPath断开便认为服务器换了。
本地diagnostics/vertex_d4_20260916，当前run=/tmp/vertex_d4_20260916/run，持久=/guohaoran/tmp/vertex_d4_20260916；console=/guohaoran/tmp/vertex_d4_console_20260916.log，launch=/guohaoran/tmp/vertex_d4_launch_20260916.json。小文件日志每步fsync，checkpoint每400更新，先共享持久读回SHA后开发评估。不要下载*.pt/*.tmp/*.partial。
起点D2 step7400/d2_update3600，SHA2384b430fa52d5012cd793fc2781c79791814f6a4b2e8ac120a711fb91112e1b，源/guohaoran/tmp/vertex_d2_resume3000_20260916/checkpoint-last.pt，NVME副本/tmp/vertex_d2_final7400_for_d4.pt。两次文件SHA一致。本地及服务器D4测试各3项通过，详test_results.json。模型/Adam/RNG必须完整恢复；resume_verification.json需source_step7400、Adam2721张量一致、trainable VecSet/DiT、初始评估不消耗RNG，然后查看训练日志非零参数更新及每更新VecSet前向8次。启动初始评估4种子、16全树和24对象对，是D4基线，不是追加D2门槛。
四对象A000105/8、B000195/52原样；C001045/54、D001885/176，闭合、无非流形/退化，点云GT对齐已检。4个目标两两不同。6对象对独立求最早分叉，此批实测都depth2/8parents，但实现测试还含depth1/9，不能推断写死。原父格比例/seq长度不强加到新对象。
训练D4新增7200步，累计7400+update（最终14600）。lr1e-5 WD0 clip1 accum8 BF16，联合VecSet/DiT，固定条件，每micro独立t/noise；k=((update-1)*8+micro)%36，mesh=k//9，depth=k%9+1。每物体每层1600 micro。不得自行调整预算/LR/模型/阈值。非有限/明确实现错误先保存证据并停止排错，不自动更换超参。
开发29000000..29000003，每400更新4×4=16树；final30000000..30000015仅训练7200后冻权重一次，16×4=64树，full_match_matrix每组必须4×4单位矩阵。共同父格每种子6对，开发24、终验96对，分别报告每对depth、2×2匹配。连续误差/F1/阈值余量均诊断，不新增硬门槛。20Euler/.5，无GTparents注入/强制非空/topk；保留逐层数组及最早失配。容量4096中止计失败。
每次同步JSON/JSONL/NPZ与console到本地；开发有实质改善、异常、完成才通知，其余安静；用户主动问则报当前已确认状态。时间预估基于实测，不以loss短期波动改方案。若再换实例，先看持久checkpoint/ledger，从D4对应update恢复剩余预算，代码支持--resume-update/--resume-evaluation/--resume-training-log；旧未保存的重放记录保留。不得把开始过的终验种子称未见。
完成后核对training_complete、checkpoint_verified update7200/cumulative14600、final_evaluation_complete及result。离线复算64树与96对象对，核对自生成父格、实际同噪声、4×4矩阵、坐标、count/FP/FN/MSE/F1/阈值余量，合并有效日志；打包完整评估证据/运行代码/四份真实条件GT/协议来源/测试/代码哈希，不含权重。暂停本监督（工具若报不存在，不能声称暂停成功）。不自动D10/20、不同时重新采样点云；后续由用户决定。
