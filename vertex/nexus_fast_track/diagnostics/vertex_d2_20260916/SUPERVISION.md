D2已完成3600步及16对终验，32/32全树正确。勿重启训练或重复终验。结果目录：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/Vertex_D2_final3600_20260916。以下为历史监督说明。

# 2026-09-16 新实例续训（本段优先）

旧实例在记录1206后消失，完整checkpoint1200已恢复启动。当前状态、身份和规则请以 ../vertex_d2_resume1200_20260916/SUPERVISION.md 为准；不要使用下方旧PID或从C重跑。有效预算仍3600，续训范围1201..3600。

# D2监督：固定3600更新，不自动扩到4/10/20

先读PROTOCOL.md和最新用户消息。当前主机1ni1p7s53t4df-0，SSH root@172.16.78.10:36910，ControlPath /tmp/nexus-d2.sock；凭据仅使用本任务用户提供的信息，不写入文件或输出。
当前PID1838/start_ticks328553149。入口 /tmp/vertex_d2_20260916/code/scripts/train_vertex_d2.py，SHA ad65d16250cb742510718923057322251704f2825952a583c992b47adc5aae58。操作前核验hostname/proc start_ticks/entrySHA。启动命令在本目录launch.json及持久/guohaoran/tmp/vertex_d2_launch_20260916.json。控制台直接写持久/guohaoran/tmp/vertex_d2_console_20260916.log。

run=/tmp/vertex_d2_20260916/run；持久evidence+checkpoint根目录 /guohaoran/tmp/vertex_d2_20260916。provenance.tar.gz可恢复完整代码/数据/协议。每条训练日志由训练本身append+fsync持久化；每个评估seed_started/seed_completed写evaluation_ledger.jsonl，完成后NPZ原子复制并回读SHA。每200先权重持久备份/校验再评估，故checkpoint_verified可能领先开发报告，不混淆。

来源C step3800/c_update1800，SHA f66c3b8e28440b39df1dd557a0631efa6228c27cc89fa0e04f171f757509eff4；保留C文件，不重新初始化、不重置Adam/RNG、不新增warmup。A000105=8顶点、B000195=52顶点；数据/condition/各层GT已审计，最早depth2分叉，双方共同8parents。

训练固定3600新增update，每200评估。LR1e-5 WD0 clip1 accum8 BF16，VecSet+DiT联合训练。微批k=((update-1)*8+micro)%18，mesh=k//9，depth=k%9+1。每9更新每个物体每层4micro，最终每对1600micro。逐条核对train.jsonl真实UID/depth与随机time/noiseSHA；确认vecset_forward_calls=8，VecSet梯度和参数更新不是全空/冻结。若异常先保存并诊断，不擅自换结构/loss/学习率/时间分布/阈值。

开发4对base seeds27000000..27000003；终验16对base seeds28000000..28000015，3600训练和权重备份完成后冻结执行一次，共32全树。每个全树从共同根节点出发，仅使用模型generated parents，20Euler/.5，无修补。输出同时对两套GT比较，正确矩阵[[true,false],[false,true]]。共同parents depth2切换用一次生成的实际noise（base+500000）clone给两条件；这项独立诊断，误差和余量不加硬门槛。不能把输出不同当正确，也不能把GT父格诊断当全树成绩。

完成状态分开看：training_complete.json；checkpoint_verified.json必须d2_update3600/cumulative7400；final_evaluation_started.json；final_evaluation_complete.json及result.json。看过或ledger已started的seed不得再称未见。若新实例替换，先读取持久记录和完整checkpoint身份；当前CLI只接受C起点，禁止用它偷偷重跑D2。需要续接时另作保留D2模型/Adam/RNG及剩余预算的恢复实现，不直接重复3600。旧/tmp消失时不能据此判所有训练丢失。

监督每次同步小型JSON/JSONL/NPZ、持久console与launch到本地当前目录；禁止下载*.pt/*.tmp/*.partial。正常不变状态保持安静；仅首次有意义评估、实际异常、完成或需用户处理时通知。用户主动询问直接报当前步数、最新完整开发矩阵和共同parents切换，不报猜测。完成后核验持久文件hash清单、离线复核数组并打包所有现存日志/数据/代码（不含权重），更新结果文档，暂停监督。自动任务不存在时不能宣称暂停成功；实验自身的持久日志保存不依赖自动任务。
