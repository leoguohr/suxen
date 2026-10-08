# D2 当前续训监督：3000 → 原预算3600
当前主机43ag0soirt181-0，root@172.16.78.10:36910，ControlPath=/tmp/nexus-d2-latest.sock；PID272，start_ticks331741245。入口/tmp/vertex_d2_resume3000_20260916/code/scripts/train_vertex_d2.py，SHA6e18d5b1ca3fbe4d14d2adb7547962d92a6779daac97ef68ff3da96e904b9266。首先核验身份。
当前run=/tmp/vertex_d2_resume3000_20260916/run；持久=/guohaoran/tmp/vertex_d2_resume3000_20260916；console=/guohaoran/tmp/vertex_d2_resume3000_console_20260916.log；launch同前缀_launch_20260916.json。本地diagnostics/vertex_d2_resume3000_20260916。
旧实例日志到3030，完整检查点3000/累计6800，来源/guohaoran/tmp/vertex_d2_resume1200_20260916/checkpoint-last.pt，SHA3bd18daa5ec377eb469f2512f835696f8b37968517e289a82859eda369668980，27990376090字节，已复制/tmp/vertex_d2_step3000.pt并读回核验。只执行3001..3600共600更新；3001..3030是重放，原记录保留。禁止重启整轮或改变预算超参。
入口代码未改，沿用测试通过的完整模型/Adam/RNG恢复。初始3000开发评估应和旧84份NPZ匹配，然后recovery_verification.json与resume_verification.json需成功。1200恢复链后这次source_step应为6800，resumed_d2_update3000，Adam2721张量一致。训练3001..3030需matches_original_replayed_input=true，逐micro UID/depth/time/noiseSHA均比较。恢复一致性不通过不可绕过断言。恢复评估为已见种子，不是新的终验。
最新开发3000：A4/4、B4/4、全树8/8、共同父格4/4。但2200/2600也是4/4，2400/2800回到3/4，不能声称稳定最终通过。最终16对28000000..28000015尚未开始，只在3600权重冻结后运行。
原实验A000105/B000195、全9层、联合训练VecSet/DiT，lr1e-5 WD0 clip1 accum8 BF16，20Euler/0.5，无GT父格注入、强制非空、topk，不追加误差门槛。详见PROTOCOL.md。
每次同步小型日志JSON/JSONL/NPZ及console，排除*.pt/*.tmp/*.partial。安静处理无变化进展，重大进步/异常/完成通知。用户主动询问直接报实测。日志每步、checkpoint每200先持久写回SHA，再评估。
最终有效日志合并：原vertex_d2_20260916取1..1200，resume1200取1201..3000，当前resume3000取3001..3600。旧1201..1206与旧3001..3030分别保留标为重放，不重复计入训练量。3600/累计7400检查点+16对终验完成后，复算所有数组并打包日志、配置、来源、代码、数据、趋势、不含权重，暂停监督。不得自动扩D4/10/20。

原始实验协议（其旧路径仅供历史来源）：
# D2从1200恢复，保留原总预算3600

旧实例停止在已记录1206，完整模型/Adam/RNG检查点1200（累计5000）。只从1200恢复，范围1201..3600，共2400次更新；其中1201..1206是丢失内存状态后的重放，不是新追加预算。旧持久目录全部保留。
来源SHA e6328c17e507df41f116c69d706d46796fb523e17fd9653de8c58c93631a4ad6。
改动仅增加D2检查点恢复入口和接续核验；数据/UID/模型/loss/Adam/LR/时间噪声/层级调度/原预留种子/阈值/3600总预算不变。
先比较原1200开发报告84份NPZ：parents/noise/目标/预测整数集合完全相同，连续estimate允许rtol1e-5/atol1e-6浮点误差。这个检查用于恢复一致性，不要求原本3/4的开发结果变成4/4。随后训练1201..1206逐micro比较原UID/depth/time/noiseSHA，保持原日志单独可审计。
新run=/tmp/vertex_d2_resume1200_20260916/run，持久=/guohaoran/tmp/vertex_d2_resume1200_20260916；原checkpoint保持在旧目录。训练与评估日志由程序本身持久保存。终验仍仅3600之后原16对28000000..28000015，未提前消费。不得从C重跑整个D2。

原方案：
# D2执行协议（启动前固定）

对象A=nexus_2k_000105，8顶点；B=nexus_2k_000195，52顶点90面。B是开口mesh（20边界边），无非流形边、无退化三角形，Stage3碰撞/删面率0。选B而非另一8顶点薄片，是为了同时验证不同形状和适度不同序列长度。点云至量化mesh最大距离A0.002037、B0.002339，均小于sqrt(3)/512；两份点云/法向固定，均通过loader与八叉树校验。
两者第1层相同8格，第2层最早不同：A8子格、B16子格，共同父格为同一8格；selection.json含原始整数集合hash、实际condition hash、各层父格数量。

起点C累计step3800/c_update1800，SHA f66c3b8e28440b39df1dd557a0631efa6228c27cc89fa0e04f171f757509eff4。保留C，不新增C验收门槛。严格加载完整模型、Adam和CPU/CUDA RNG；VecSet和DiT train/requires_grad True，禁止训练使用detach/cached VecSet context。
新增3600次更新，每200开发评估，LR1e-5、WD0、clip1、accum8、BF16，无warmup。k=((update-1)*8+micro)%18，mesh=k//9，depth=k%9+1。每9次update，每个物体每层4micro；3600次结束每对1600micro。每micro独立time~U[0,1)、新噪声；condition/level必须来自同一single-mesh batch。

开发base seeds27000000..27000003，16对终验base seeds28000000..28000015预留到3600权重保存后冻结执行一次，不早停选优，不用于调参。完整生成使用20Euler/.5阈值；每层实际noise seed=base+depth*1000，两对象根层parents与实际noise完全相同。后续父格来自各自模型输出，节点数不同允许噪声张量长度不同。无GTparents替换、top-k、非空修复；沿用C容量4096parents中止记录失败，禁止截断后称通过。

每个生成结果同时对比A/B目标，保存2x2精确坐标匹配矩阵。完整条件切换通过要求每对矩阵[[true,false],[false,true]]，终验共32条全树。各层保存parents/noise/连续值/输出cells/目标集合及最早失配。
额外共同父格切换：固定depth2同8parents，实际noise只生成一次（seed=base+500000）并clone给两个context，只切换VecSet条件。保存双方原始数组和各自对两套层级GT的矩阵。这项单独报告；连续MSE、阈值余量都只是诊断，不增加事后硬门槛。

证据持久化：/guohaoran/tmp/vertex_d2_20260916是主备份目录；每update train.jsonl两端append+flush+fsync，状态/配置原子复制并回读SHA。初始provenance.tar.gz含实际代码、测试、两份数据、manifest与选择/对齐报告。每200先完整模型+Adam+RNGcheckpoint保存并回读SHA，再开发评估；每个评估seed开始先持久ledger，完成后复制该对全部NPZ并验证SHA、记录seed_completed。训练完成、checkpoint_verified、final_evaluation_started、final_evaluation_complete分开记录。若seed_started后未completed，恢复时不称该seed未见。

训练仅当前固定预算；异常先保存证据，不自动改学习率/结构/时间分布/阈值，也不自动扩到4/10/20。GPU smoke测试不计生成成绩。完整模型的恢复核验和首批实际日志必须单独检查。
