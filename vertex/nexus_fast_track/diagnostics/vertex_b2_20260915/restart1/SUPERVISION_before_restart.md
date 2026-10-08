# 当前授权任务：B2随机时间，1000次新增更新

先读本目录PROTOCOL.md及最新用户消息。用户最新方案授权冻结R1 step1000补做64独立噪声；几何/回归64/64通过后才继续B2。该前提已实际通过，报告在frozen64_report.json。旧严格B1和新增followup严格结果仍为false（局部响应3/64超标），不得改写为全部通过，也不要重跑这些64种子作为未见终验。

## 运行身份

主机1fl3imip3trjj-0，SSH root@172.16.78.10:31548，ControlPath /tmp/nexus-vertex-frozen64.sock。当前PID5799，start_ticks320293069。冻结入口SHA e1c21c0b60c7cd0d87241e00e6c8fb0a9c04fca8c45dd43d03b2d6098036f7d8。先核验host/proc start_ticks/hash再操作。完整启动命令见launch.json。目录/tmp/vertex_b2_20260915，本地对应本目录。

来源R1 step1000 SHA a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279，持久备份/guohaoran/tmp/vertex_r1_preserved_20260915/checkpoint-step1000.pt，已回读验证。B2从同SHA原/tmp副本加载完整模型、Adam及CPU/CUDA RNG；检查resume_verification.json，不能仅凭配置声称恢复成功。启动初期读28GB checkpoint+验证optimizer，status.json尚未生成可以是正常准备；检查PID与console。

## 监督规则

唯一对象000105、depth9 GT parents、固定实际点云/法向，完整模型。每update8个独立random t~U[0,1)和高斯噪声；LR1e-5、WD0、不新增warmup、不改结构/阈值/loss。预算固定1000新增更新，每100开发评估，完成后一次8份fresh单层采样。不得自加预算、重复启动、换小模型/裁剪父节点，也不要自动C/D。

开发验证每个t=0,.1,.3,.5,.7,.9,.95分别16probes；看by_time各组，不能用跨时间平均。20步Euler GTparents单层采样8种子是验收，40步只是对照。最终8份fresh种子13000000..13000007仅一次，非用于调参。responses_fp32参考-1/(1-t)，不一律-2，也不是本轮几何硬门槛。result.passed要求开发几何+8份最终20步采样都过，complete仅意味着结束。

每次监督同步小型run JSON/JSONL/NPZ、console、resume验证到本地本目录；数组目录按更新隔离，未完成评估不要称为完整结果。不要下载或打包模型权重。遇NaN/OOM/进程退出异常先保存证据并诊断，不自改参数。保持正常进度安静，仅首次有意义的评估/门槛通过失败/异常/连接问题/用户需决策时通知。用户询问时直接报告当前新增update和最新完整评估。

B2新checkpoint仍在/tmp，不能说已持久化；结束后可备份到现有/guohaoran/tmp/vertex_r1_preserved_20260915目录的新文件名并回读SHA，保留原R1文件。共享盘曾解包阻塞但本轮大文件流式备份验证成功；若备份阻塞不要干扰训练。完成预算后整理完整小型复核包，暂停监督，不再训练。
