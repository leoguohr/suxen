# 2026-09-15 新端口36910 / B2 restart1（本段优先）

用户提供新服务器要求继续。旧实例1fl3imip3trjj-0已不可达，新主机5utktasmq5anv-0没有旧/tmp，所以旧B2最后观察191步不可续接。持久R1 step1000检查点已校验同SHA；本次从同模型/Adam/RNG重跑原1000次B2，不能称为从191继续，也不从随机初始化开始。旧run/证据保留。

当前PID1113、start_ticks320424510、SSH root@172.16.78.10:36910、ControlPath /tmp/nexus-vertex-36910.sock。新入口SHA7fcb55970b9a90ff6b70ffaf1b2a241bc1eb9fdf82b37e0c9867e2a6a26d2d24。实际远端/tmp/vertex_b2_20260915。当前launch为本地restart1/launch.json，所有新日志/评估下载本地restart1/run，禁止覆盖旧run。训练原始source SHA仍a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279，当前临时读取副本/tmp/vertex_r1_step1000.pt。

唯一代码变化：每100步checkpoint保存后同步做持久备份及回读SHA；数学/数据/时间噪声/Adam/LR/预算/门槛不变。本地及新服务器40测试通过。持久目标/guohaoran/tmp/vertex_b2_restart1_20260915/checkpoint-last.pt，metadata backup_verified.json；第100步第一次备份完成前只有R1持久权重，不能提前称B2已备份。state=backing_up是正常操作，等待完成再继续更新。备份期间不另起复制或训练进程。若IO长期阻塞先诊断，不静默跳过校验。

先核验resume_verification以及新旧前几步times/noise SHA一致。每次监督同步证据，保留last_remote和last_downloaded区别。原R1备份不覆盖，已消费的64评估不重做。训练结束后程序自带最后持久备份，无需重复大文件复制；验证最终backup metadata步数及SHA后打包小结果，暂停监督。

历史协议细节如下（本段新身份/路径/备份规则优先）：

# 当前授权任务：B2随机时间，1000次新增更新

先读本目录PROTOCOL.md及最新用户消息。用户最新方案授权冻结R1 step1000补做64独立噪声；几何/回归64/64通过后才继续B2。该前提已实际通过，报告在frozen64_report.json。旧严格B1和新增followup严格结果仍为false（局部响应3/64超标），不得改写为全部通过，也不要重跑这些64种子作为未见终验。

## 运行身份

主机5utktasmq5anv-0，SSH root@172.16.78.10:36910，ControlPath /tmp/nexus-vertex-36910.sock。当前PID1113，start_ticks320424510。冻结入口SHA 7fcb55970b9a90ff6b70ffaf1b2a241bc1eb9fdf82b37e0c9867e2a6a26d2d24。先核验host/proc start_ticks/hash再操作。完整启动命令见launch.json。目录/tmp/vertex_b2_20260915，本地对应本目录。

来源R1 step1000 SHA a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279，持久备份/guohaoran/tmp/vertex_r1_preserved_20260915/checkpoint-step1000.pt，已回读验证。B2从同SHA原/tmp副本加载完整模型、Adam及CPU/CUDA RNG；检查resume_verification.json，不能仅凭配置声称恢复成功。启动初期读28GB checkpoint+验证optimizer，status.json尚未生成可以是正常准备；检查PID与console。

## 监督规则

唯一对象000105、depth9 GT parents、固定实际点云/法向，完整模型。每update8个独立random t~U[0,1)和高斯噪声；LR1e-5、WD0、不新增warmup、不改结构/阈值/loss。预算固定1000新增更新，每100开发评估，完成后一次8份fresh单层采样。不得自加预算、重复启动、换小模型/裁剪父节点，也不要自动C/D。

开发验证每个t=0,.1,.3,.5,.7,.9,.95分别16probes；看by_time各组，不能用跨时间平均。20步Euler GTparents单层采样8种子是验收，40步只是对照。最终8份fresh种子13000000..13000007仅一次，非用于调参。responses_fp32参考-1/(1-t)，不一律-2，也不是本轮几何硬门槛。result.passed要求开发几何+8份最终20步采样都过，complete仅意味着结束。

每次监督同步小型run JSON/JSONL/NPZ、console、resume验证到本地本目录；数组目录按更新隔离，未完成评估不要称为完整结果。不要下载或打包模型权重。遇NaN/OOM/进程退出异常先保存证据并诊断，不自改参数。保持正常进度安静，仅首次有意义的评估/门槛通过失败/异常/连接问题/用户需决策时通知。用户询问时直接报告当前新增update和最新完整评估。

B2新checkpoint仍在/tmp，不能说已持久化；结束后可备份到现有/guohaoran/tmp/vertex_r1_preserved_20260915目录的新文件名并回读SHA，保留原R1文件。共享盘曾解包阻塞但本轮大文件流式备份验证成功；若备份阻塞不要干扰训练。完成预算后整理完整小型复核包，暂停监督，不再训练。
