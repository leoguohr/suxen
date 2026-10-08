# 七组 Vertex 实验：2026-10-07 换实例恢复

## 恢复依据

- 新实例 `6nmeepeo9c39q-0`；初次检查无训练进程，两张 A100 80GB 空闲。旧 queue/status 为已终止实例的记录。
- S0 日志最后为 update7727；S1 为 update7614。两支最近完整恢复点均为 update7600 / 累计31600；本轮重新读取整文件计算 SHA256，均匹配。
- S0 checkpoint SHA256：`324c40e4f39b7ca3bdf43cab32732239aa2198692051cd788ead495256d7dcca`。
- S1 checkpoint SHA256：`cb561c290455f9358594e20989bcadd9604eaaf83fa0e3950dfc3346924c0119`。
- 原恢复点：`/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261006/runs/S{0,1}/checkpoint-031600.pt`。
- 本轮输出：`/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs`。
- 持久挂载点 `/guohaoran`，文件系统 `storage`（GPFS）。16 MiB 写入/fsync、430 GiB 实际分配/fsync 检查通过；探针已释放，不构成未来空间预留。
- PyTorch `2.3.0a0+6ddf5cf85e.nv24.04`、CUDA 12.4，与原训练一致。

## 续训边界

- S0/S1 恢复模型、全部 Adam、RNG、事件游标60800及各对象/层级参与次数，从7601继续至原终点10000，各剩余2400次更新。
- S0 多出的127条、S1多出的14条未提交日志存档，权威日志不重复计数；7601使用原事件、实际噪声和time，核对原记录。
- S2–S6保持从原 A24000 独立开始，各10000更新；模型、数据、loss、优化器配方、global8、microgroup2及采样协议均不改变。
- 新增独立恢复脚本；保留此前 `recovery` 元数据，追加 `recovery7600`，核验并归档前次恢复代码。
- GPU0：`GPU-49342101-4d41-f5bd-7cf1-3de6c20fbe33`；GPU1：`GPU-7da9c7e3-27c0-2089-0464-a2b98018b994`。
- 取得原始队列、前次恢复队列、本轮队列锁，防止重复启动。
- 恢复首个更新7601完整保存并读回，随后每200保存、每组保留最近两份；训练中不做生成评估。
- 七组依既定队列训练、终点评估、汇总打包；不创建持续监督。

## 核验状态

- 完整文件哈希、环境与空间预检通过，证据见 `audit/preflight_identity.json`。
- Astra 静态复审通过；本地队列测试5项通过。
- 服务器CPU测试10项全部通过，耗时2.338秒，GPU不可见；含完整恢复后下一步与不中断模型/Adam/RNG/事件的一致性检查。
- 后台队列PID605，GPU0/S0进程805，GPU1/S1进程808；与SSH终端分离。启动命令、代码哈希见 `audit/launch.json`。
- 两支真实恢复验证通过：全部模型/Adam张量与源checkpoint相等，RNG、事件游标、曝光计数、固定条件缓存一致。S0共有907个Adam状态为31600；S1原907个状态为31600，新增2个状态为7600。
- 两支7601实际更新已完成，UID、depth、实际noise SHA和time与此前未提交的7601记录相同。
- 两支7601完整checkpoint已在新持久目录保存、SHA256计算及全部模型/Adam/RNG/config/游标/曝光状态读回通过；随后均已继续超过7601。真实启动快照见 `audit/startup_verified.json`。
- S0累计31601 checkpoint SHA256：`d0bd3cb9a1640850e35f72110d2b8418c57dd9b3be031894e00dd3ee590ce5c0`。
- S1累计31601 checkpoint SHA256：`3aac47bd511818c632fd60f1a205be626147c4065f7d1f2939076a4ced88faff`。
- 启动期间共享盘小文件/目录操作有明显延迟；未改训练配方或保存协议。以上是启动核验，不代表最终训练或生成验收完成。
- S2–S6仍在自动队列；不进行持续监督。原代码、A24000、固定数据和前两轮恢复来源继续保留。
