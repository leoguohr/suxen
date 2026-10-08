# D2 已启动

A=nexus_2k_000105（8顶点），B=nexus_2k_000195（52顶点）。两者在depth2最早不同，具有完全相同的8个父格；选样与点云空间对齐检查见selection.json、point_alignment_audit.json。
从C累计step3800模型/完整Adam/CPU及CUDA RNG继续。完整2.332B模型，VecSet与DiT均可训练；恢复核验2721个Adam状态张量相等。固定3600更新，每200开发评估；原优化与采样设置保持不变，详细协议在PROTOCOL.md。

训练前4对开发种子：无论A还是B条件，完整生成均匹配A，矩阵[[true,false],[true,false]]；共同parents切换也全部为该矩阵。因此初始完整条件切换0/4、共同parents诊断0/4；该结论指离散坐标输出，不能说连续输出或VecSet特征完全不受条件影响。

已核验前34个实际update：每micro的UID/层级调度正确；前3个完整9-update区间，每个对象每层4份micro；VecSet每update前向8次、34次都有非零参数更新。此为训练接线检查，不是D2生成验收通过。
NVME和持久train.jsonl前34条逐条相同，持久初始84份NPZ hash均实际回读核验。本地有日志和完整初始评估。D2首次checkpoint要等第200步；尚不能称已保存D2模型检查点，原C检查点仍保留。

后台监督vertex-d2已创建。训练进程自身每步fsync日志并保存评估ledger/数组，证据持久化不依赖后台任务。D2结束前不使用预留16对终验seed，不自动扩到4/10/20。
