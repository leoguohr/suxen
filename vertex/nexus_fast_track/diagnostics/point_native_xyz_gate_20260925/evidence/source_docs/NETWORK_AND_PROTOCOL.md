# 网络结构与运行约定

## 输入与输出

- 点云条件 `[B,8192,6]`：同一归一化空间的XYZ和法向；本轮每个物体的实际输入固定，关闭增强。
- VecSet：1024个FPS query，宽2048，16 heads，1 cross + 7 self，输出 `[B,1024,2048]` 条件。
- 深度d的GT父格编码 `[B,N,3]`，子格占据 `[B,N,8]`；depth=1..9，共用同一个网络。
- Vertex DiT：36层、宽1536、12 heads；输入占据投影+父格绝对坐标Fourier投影+深度嵌入，self attention 使用3D RoPE；逐块时间调制，cross attention读取VecSet条件，输出连续速度，无sigmoid。
- 合计2,332,430,344参数（VecSet402,987,008，DiT1,929,443,336）。这些具体层数/宽度/初始化组合是本地实现选择，不是作者源码证明。

## R0 与 R1：必须读取真正的构造入口

`code/scripts/train_vertex_a100_b1.py::make_model` 在基础 VertexStageSystem 后做初始化覆盖。
R1把depth embedding和时间MLP权重设为std=0.02的正态初始化、时间MLP偏置清零；每层cross attention的output投影权重/偏置清零。基础AdaLN和末端速度投影亦有零初始化。
因此只读 `vertex.py` 或直接构造默认 VertexStageSystem，不能自动代表成功实验的R1从头初始化。恢复已保存权重时必须strict load，并核对对应config。
零输出初始化会让第一步反传的部分上游梯度为零，这是初始化行为；不能只凭第0步VecSet无梯度断定训练断链。后续联合更新由专项测试和生产日志验证。

## Flow 与采样

`x_t=(1-t)*epsilon+t*Y`，监督速度 `Y-epsilon`，t从0的噪声端积分到1。
解析测试速度 `(Y-x_t)/(1-t)` 只用于GT oracle，不能用于模型成绩。
实际验收每层20步Euler，阈值0.5。子格bit顺序 `4*x+2*y+z`。深度9整数坐标范围0..511，叶子中心解码 `-1+(cell+0.5)*2/512`。
这是现有Euler实现；不能说已经实现论文提到的DPM-Solver。
完整生成从根格开始，下一层parents完全来自当前预测；不替换GT parents、不强制非空、不按目标数量top-k修补。代码允许容量中止，需计为失败，不能当成功过滤。

## 代码导航

| 文件 | 作用 |
|---|---|
| `mini_nexus/vertex.py` | VecSet、DiT、RoPE、QK norm、FPS |
| `mini_nexus/training.py` | VertexStageSystem联合前向、loss；文件另有拓扑依赖，勿随意删 |
| `mini_nexus/flow.py` | 线性路径与监督速度 |
| `mini_nexus/octree.py` | 量化、子格展开与坐标解码 |
| `mini_nexus/data_2k.py` | 固定manifest数据加载与batch |
| `mini_nexus/vertex_evaluation.py` | 层级采样与完整生成评估 |
| `scripts/check_vertex_sampler.py` | GT解析oracle、轴bit/解码检查 |
| `scripts/train_vertex_a100_b1.py` | R0/R1构造、A/B1及噪声响应诊断 |
| `scripts/train_vertex_b2.py` | 随机时间单层训练 |
| `scripts/train_vertex_c.py` | 单物体九层训练与sample_tree |
| `scripts/train_vertex_d2.py` | 双对象条件切换、共同父格测试 |
| `scripts/train_vertex_d4.py` | 四对象续训，包含恢复核对、FPS索引缓存及双GPU评估 |
| `scripts/evaluate_vertex_d4_worker.py` | 第二张GPU独立执行另一半评估种子 |

## 成功阶段训练设置

R1；VecSet与DiT联合训练；lr=1e-5、weight_decay=0、clip=1、8次梯度累积、BF16；除A固定回归外每micro重新采样t/noise。只缓存固定点云的FPS索引，训练时不得缓存detach后的VecSet特征。
D2轮转2×9个组合，D4轮转4×9个组合；每micro一个对象一个深度，不把不同物体attention混合。
D4总新增7200updates，起点D2累计7400；最终应累计14600。最新本地恢复源是D4 6800，总预算没有增加。

最后部署关闭激活重计算，用更多显存减少重算；GPU0仍单卡训练，GPU1只拆分评估，**不是DDP训练**。完整模型短基准3.68→2.09秒/步，峰值allocated39.47→42.03GiB；正式早期26步记录均值约2.25秒。BF16梯度不逐位相同，原设置重复运行relativeL2约0.001799，关闭重计算对照约0.001817。详见capacity证据，不外推到任意样本规模。
