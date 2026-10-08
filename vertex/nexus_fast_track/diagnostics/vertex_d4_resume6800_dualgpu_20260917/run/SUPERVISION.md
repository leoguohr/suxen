# D4 当前实例：31248，两张卡参与；2026-09-17
服务器 root@172.16.78.10:31248，hostname d2ff77umjvrgu-0，ControlPath /tmp/nexus-d4-31248-new.sock。PID786，start_ticks341141994。启动记录 /guohaoran/tmp/vertex_d4_resume6800_dualgpu_launch_20260917.json。不要使用任何旧实例PID或GPU UUID重启。
用户已允许两卡加速，覆盖此前GPU0-only限制。正式训练仍在GPU0 UUID GPU-0a371e11-9041-9b22-d2d5-8469622c4681；GPU1 UUID GPU-d0e44065-83ff-2218-e56d-37794fd052b9运行独立评估进程。两个评估进程平分原种子：开发每卡2个、终验每卡8个；每种子4树+6共同父格对象对，合并后原指标/预算不变。不是DDP训练。
代码 /tmp/vertex_d4_resume6800_dualgpu_20260917/code；入口SHA71dc0415098d5b163e526f92bfa6d9ebc224e3e64ceccf8b5c32fd7da0e2fbf2。90份代码/数据文件部署SHA通过，CPU串行/双进程评估及恢复测试4项通过。新增worker scripts/evaluate_vertex_d4_worker.py。manifest/selection/provenance位于同一运行根目录。
输出 /tmp/vertex_d4_resume6800_dualgpu_20260917/run，持久 /guohaoran/tmp/vertex_d4_resume6800_dualgpu_20260917，console /guohaoran/tmp/vertex_d4_resume6800_dualgpu_console_20260917.log。本地同名diagnostics目录。只同步小文件，排除pt/tmp/partial；worker持久目录有数组冗余备份，最终计数以合并顶层报告array_sha256为准。不要把worker重复数组计入样本数。
恢复源 /guohaoran/tmp/vertex_d4_resume3600_gpu0_20260917/checkpoint-last.pt；NVME /tmp/vertex_d4_step6800.pt，27990411994字节，SHAc0bec9a4319473f0bfc756418ad08bf744f73de605c38c527bc7978dc7729fd4；复制读回均通过。来源D4update6800/累计14200；总7200，剩400。旧日志到6935，所以6801..6935共135步是重放，输入UID/depth/time/noiseSHA必须完全相同。
保持已验证FPS索引复用，VecSet仍每micro前向。唯一训练计算改动是关闭VecSet/DiT激活重计算，--no-activation-checkpointing。不改变模型、lr/WD/accum8/clip/采样协议。完整模型BF16对照8个loss逐项相同；原配置重复运行梯度relativeL2 0.001799；关闭重计算vs原为0.001817，同量级但不逐位相同。不能宣称梯度/后续轨迹逐位等价。最终纠正Adam step临时共享问题后的计时：每项1热身+9测量，原均值3.67896秒，关闭重计算2.08780秒；峰值allocated39.466→42.033GiB。短测，不承诺长跑相同比率。记录在 diagnostics/vertex_d4_capacity_20260917/timing_result.json；完整权重未下载。
启动后先验证216份旧6800开发数组（新增双卡评估），Adam2721张量、RNG恢复，之后才训练。若恢复断言失败立即排错，不绕过验证。最新旧开发6800为14/16，A/B/C各4/4，D2/4，共同24/24；6400曾15/16，不应只报告历史最好值。
有效最终日志：原D4 1..1200 + resume1200 1201..2933 + FPS版2934..3600 + resume3600版3601..6800 + 本次6801..7200。全部旧未保存重放日志保留，不重复计数。

四对象A000105/8、B000195/52原样；C001045/54、D001885/176，闭合、无非流形/退化，点云GT对齐已检。4个目标两两不同。6对象对独立求最早分叉，此批实测都depth2/8parents，但实现测试还含depth1/9，不能推断写死。原父格比例/seq长度不强加到新对象。
训练D4新增7200步，累计7400+update（最终14600）。lr1e-5 WD0 clip1 accum8 BF16，联合VecSet/DiT，固定条件，每micro独立t/noise；k=((update-1)*8+micro)%36，mesh=k//9，depth=k%9+1。每物体每层1600 micro。不得自行调整预算/LR/模型/阈值。非有限/明确实现错误先保存证据并停止排错，不自动更换超参。
开发29000000..29000003，每400更新4×4=16树；final30000000..30000015仅训练7200后冻权重一次，16×4=64树，full_match_matrix每组必须4×4单位矩阵。共同父格每种子6对，开发24、终验96对，分别报告每对depth、2×2匹配。连续误差/F1/阈值余量均诊断，不新增硬门槛。20Euler/.5，无GTparents注入/强制非空/topk；保留逐层数组及最早失配。容量4096中止计失败。
每次同步JSON/JSONL/NPZ与console到本地；开发有实质改善、异常、完成才通知，其余安静；用户主动问则报当前已确认状态。时间预估基于实测，不以loss短期波动改方案。若再换实例，先看持久checkpoint/ledger，从D4对应update恢复剩余预算，代码支持--resume-update/--resume-evaluation/--resume-training-log；旧未保存的重放记录保留。不得把开始过的终验种子称未见。
完成后核对training_complete、checkpoint_verified update7200/cumulative14600、final_evaluation_complete及result。离线复算64树与96对象对，核对自生成父格、实际同噪声、4×4矩阵、坐标、count/FP/FN/MSE/F1/阈值余量，合并有效日志；打包完整评估证据/运行代码/四份真实条件GT/协议来源/测试/代码哈希，不含权重。暂停本监督（工具若报不存在，不能声称暂停成功）。不自动D10/20、不同时重新采样点云；后续由用户决定。


启动后实测：216份6800恢复数组全部通过；Adam2721张量一致且RNG保留；初始完整树14/16、共同24/24不变。正式训练已到6802，前两步2.134/2.597秒。135步重放尚在进行，不能提前声称全部通过。parallel进度文件只记录主进程本地2个开发seed，完整计数以evaluation-006800.json等顶层合并报告为准（4组）。
