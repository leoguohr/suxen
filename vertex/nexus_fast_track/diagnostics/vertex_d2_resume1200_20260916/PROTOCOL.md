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
