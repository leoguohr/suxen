# PHASE3_COARSE_ANALYSIS

结论：优先处理现有10对象的粗层分叉稳定性，有证据支持把用户提出的粗层2:1加权作为下一项单变量实验；尚未证明原权重不足是根因。现有证据不支持立即加深条件网络、扩数据或调整模型容量。本轮没有训练，也没有实现loss加权。

## 1. 实际做了什么

- CPU读取Phase2真实生成结果，核验48条树、720份逐层数组的原始哈希，重新统计所有15层。
- 在用户指定的物理GPU1上加载Phase2最终checkpoint，严格核对完整SHA256和冻结代码哈希。仅执行冻结前向：180个生成初始状态，观察3个DiT block；另做540次GT父格回归前向。检查期间eval、requires_grad=False、无optimizer、无backward、无checkpoint写入；原runtime代码与checkpoint大小/mtime均未变化。
- 观察hook开启/关闭的首个前向输出逐元素相同。GPU诊断用时81.59秒，峰值allocated约8.95GiB，已结束；GPU0和VAE/AE V2未使用或修改。
- 下载并在本地复核本轮导出的540条回归标量、attention分布归一性、条件token RMS及文件哈希。没有下载模型权重。

权重：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_phase2_d15_10_20260926/run/checkpoint-012000.pt`，Phase2 update6000 / cumulative step12000。
SHA256：`44ca719f853f064e60024fd09a9c9d021a9455013af09577145ae139841b5501`。

## 2. 逐层占据准确率与首错

所有数据来自原4个种子94026000–94026003，没有换种子或重新生成验收树。seen=10对象×4，unseen=2对象×4。

occupancy accuracy的统计域是“预测父格与GT父格并集的8个子格”，不是包含海量空格的整个2^(3d)网格。GT缺失父格下的目标子格记FN，没有静默排除。主表按占据位数汇总（micro）；逐物体/种子等权macro值和仅预测父格域的值另存在CSV。

- accuracy=(TP+TN)/(TP+TN+FP+FN)。
- F1=2TP/(2TP+FP+FN)。
- 整层正确：该层完整整数格子集合与GT相等，不能用点数相同代替。
- 首错：此前所有层均正确、本层第一次失配。后续错误不会重复记入首错。

| depth | seen占据准确率 | seen F1 | seen整层正确 | seen首错条数 | unseen占据准确率 | unseen F1 | unseen整层正确 | unseen首错条数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100.00% | 100.00% | 40/40 | 0 | 100.00% | 100.00% | 8/8 | 0 |
| 2 | 99.65% | 99.35% | 36/40 | 4 | 82.42% | 80.85% | 0/8 | 8 |
| 3 | 99.46% | 98.98% | 32/40 | 4 | 65.45% | 28.73% | 0/8 | 0 |
| 4 | 99.24% | 97.86% | 26/40 | 6 | 84.64% | 10.63% | 0/8 | 0 |
| 5 | 98.80% | 96.00% | 21/40 | 5 | 85.83% | 3.65% | 0/8 | 0 |
| 6 | 98.51% | 94.18% | 20/40 | 1 | 87.20% | 2.05% | 0/8 | 0 |
| 7 | 98.02% | 92.09% | 19/40 | 1 | 87.48% | 0.96% | 0/8 | 0 |
| 8 | 97.40% | 89.14% | 18/40 | 1 | 87.29% | 0.00% | 0/8 | 0 |
| 9 | 96.96% | 86.54% | 18/40 | 0 | 87.51% | 0.00% | 0/8 | 0 |
| 10 | 96.46% | 83.93% | 17/40 | 1 | 87.50% | 0.00% | 0/8 | 0 |
| 11 | 95.97% | 81.28% | 17/40 | 0 | 87.53% | 0.00% | 0/8 | 0 |
| 12 | 95.64% | 79.19% | 17/40 | 0 | 87.50% | 0.00% | 0/8 | 0 |
| 13 | 95.01% | 75.95% | 17/40 | 0 | 87.49% | 0.00% | 0/8 | 0 |
| 14 | 94.75% | 73.84% | 16/40 | 1 | 87.47% | 0.00% | 0/8 | 0 |
| 15 | 94.49% | 72.10% | 16/40 | 0 | 87.53% | 0.00% | 0/8 | 0 |

训练对象24条失败中19条（79.2%）首次错在depth2–5，另5条分别始于6、7、8、10、14。depth2只有6个FP和3个FN，却已让4条树失败；因此99.65%的位准确率仍不够。depth4、5在此前仍正确的树中分别有6/32、5/26首次失败。

depth15的928个FN中，868个（93.5%）位于已经缺失的父格下，不能把它们全部解释为depth15自身的去噪错误。CSV另列incoming_parents_exact和正确父格子集，防止混淆上游错误传播。该子集有幸存者偏差，并非给所有对象GT父格的新采样。

未见对象8条全部在depth2首次错；depth8以后F1为0，accuracy却约87.5%，体现空格比例导致的表面高准确率。不能按位准确率宣布泛化成功。

![逐层统计](depth_diagnostics.png)

## 3. 当前loss与新冻结回归诊断

实际入口`phase2_train_ssd_fast.py:293–310`按k%150循环10×15组合，每个microbatch只取一个物体的一层。8次累积执行一次optimizer更新。`VertexStageSystem.forward`在`mini_nexus/training.py:78–94`构造线性flow目标，并按有效父格×8归一MSE。当前没有depth权重，也没有对全部15层同时求和。每个物体×层级实际消费320份microbatch，未因父格数量多而获得更高loss权重。

原始6342行日志包括中断重算；最终有效1–6000更新均完整。日志micro记录UID/depth/t/noise哈希，不记录每份micro的loss，故无法从已有曲线反推出历史逐层loss或逐层梯度。

为补充当前状态，本轮固定最终权重，给每个对象每层GT parents，在一个新保存的实际噪声上分别测试t=0.1/0.5/0.9。12×15=180份实际噪声，共540次前向。预测仍是v，目标仍是y−epsilon；没有训练、没有改采样器。下表按对象、层级、时间等权平均，不是完整时间积分或噪声总体期望。

| split | depth1–5 velocity MSE | depth2–5 velocity MSE | depth6–15 velocity MSE |
|---|---:|---:|---:|
| seen | 0.038278 | 0.034508 | 0.021858 |
| unseen | 0.474661 | 0.584523 | 0.276125 |

seen粗层1–5约为细层的1.75倍；在t=.1/.5/.9，粗层分别为.02005/.02295/.07184，细层为.01202/.01536/.03820。占据位上的velocity误差也较高（粗层.05880，细层.04063）。这是支持尝试粗层加权的有限证据，不证明梯度弱或原权重错误。粗细层占据比例不同，depth1全占据，MSE难度不能完全等同；depth1虽然MSE .05336，但本次所有生成均正确。

单步端点估计x_t+(1−t)v仅作为回归诊断，不能替代20步Euler完整生成成绩。原始输入/目标/输出在`frozen_probe/frozen_regression_arrays.npz`。

## 4. 条件token与cross-attention

真实前向观察范围：原生成seed94026000，每个UID/depth的实际父格和保存噪声，t=0，DiT第1/18/36个block（JSON为0/17/35）。共180个输入、540条block记录。Q/K来自原forward的QK norm输出；另行用FP32计算softmax权重统计，未替换原BF16 SDPA。因此这些权重是观测Q/K的高精度诊断，并非逐bit重现融合attention内核。

同一对象的VecSet条件tokens跨depth复用，故token norm本来就不随depth变化。12对象token RMS范围0.99051–0.99257；VecSet有末端LayerNorm，norm接近1不能证明条件信息强或弱。

以下是seen对象等权平均，每格为“depth1–5 / depth6–15”：

| DiT block（从1计） | 归一化attention熵 | 平均最大token权重 | cross输出RMS/分支加入前残差RMS |
|---|---:|---:|---:|
| 1 | 0.9606 / 0.9388 | 0.28% / 0.33% | 1.41% / 1.87% |
| 18 | 0.9401 / 0.9318 | 0.26% / 0.27% | 1.03% / 1.21% |
| 36 | 0.9195 / 0.9197 | 0.30% / 0.29% | 0.12% / 0.12% |

归一化熵=−sum(a log a)/log(1024)，均匀分布时为1；平均最大权重的均匀基线为1/1024≈0.0977%。这些attention较分散，初块粗层的相对cross更新较小，但还不能归因为conditioning失效：条件token已通过自注意力混合了全局信息，cross分支也跨36个block累积作用。

原有共同父格、相同噪声、只换条件的45对测试中21对两侧都正确；44对在depth2、其中20对通过，另1对在depth3通过。输出变化RMS范围0.213–0.832，说明条件确实影响预测，但影响还不够稳定。不能把“有响应”写成“条件学习完全通过”。未见对象000002有两个完整输出恰等于seen000195，是已有的错误目标匹配。

本轮attention仅观察一个种子的t=0，不覆盖整条ODE；深层实际父格可能已偏离GT。另提供仅正确父格子集的统计，但样本构成不同，不能当成严格配对因果比较。原45对条件检查仅保存JSON，未补造原占据数组或宣称重跑。

## 5. 失败与点数、结构、对称性、输入质量

| UID | split | GT去重N | 完整树正确 | 点数正确 | GT顶点到条件最近点的最大距离 |
|---|---|---:|---:|---:|---:|
| nexus_2k_000105 | seen | 8 | 4/4 | 4/4 | 0.02106 |
| nexus_2k_000195 | seen | 52 | 3/4 | 3/4 | 0.04437 |
| nexus_2k_001045 | seen | 54 | 0/4 | 0/4 | 0.03830 |
| nexus_2k_001885 | seen | 176 | 0/4 | 0/4 | 0.03577 |
| nexus_2k_000014 | seen | 12 | 4/4 | 4/4 | 0.03227 |
| nexus_2k_000022 | seen | 40 | 1/4 | 2/4 | 0.03171 |
| nexus_2k_000064 | seen | 132 | 0/4 | 0/4 | 0.01920 |
| nexus_2k_000084 | seen | 128 | 0/4 | 0/4 | 0.04856 |
| nexus_2k_000119 | seen | 154 | 0/4 | 0/4 | 0.01165 |
| nexus_2k_000368 | seen | 16 | 4/4 | 4/4 | 0.04572 |
| nexus_2k_000002 | unseen | 27 | 0/4 | 0/4 | 0.04040 |
| nexus_2k_000009 | unseen | 168 | 0/4 | 0/4 | 0.05605 |

1. 点数/复杂度：8、12、16顶点的三个seen对象12/12全对；40、52顶点合计4/8；54及以上五个seen对象0/20。失败与复杂度有明显共现，但52点3/4与54点0/4说明点数不是唯一解释，而且精确集合判据本身随顶点/分支增加更难。旧两对象还继承了更充分的预训练，不能据这10例认定模型容量不够。
2. 粗结构：19/24失败从2–5层开始，支持粗分叉稳定性是优先问题。`000064`、`000119`是细长且顶点密集的对象，也失败，不能笼统说只有外形体积大才失败。逐对象GT与条件图如下。
3. 对称性：只检验固定坐标轴、围绕GT包围盒中心反射后的最近点RMS，未做旋转搜索或预测对齐。对称的000105全通过；至少两个轴近对称的001885全失败；非对称001045也全失败。现有小样本不支持“对称物体导致失败”。这只是顶点集合的轴反射诊断，不是完整mesh对称分类。
4. 输入：每个对象都有8192个不同且有限的点，法向范数0.99999994–1，条件/label哈希一致；12对象均无D15量化碰撞。全部10个训练对象的GT顶点都在某个输入点0.05距离以内，最大值0.04856（包围盒最长边2）。`000119`最大仅0.01165仍失败，没有明显的整片缺失证据。

0.05只是描述统计，不是新增数据合格门槛。这不能证明表面采样足以保留每个细节、法向朝向全部正确或细分顶点可由表面唯一识别。当前包里没有manifest所指mesh_normalized.npz，无法独立计算全部三角面覆盖率、面积或采样法向与原面的一致性；sampled_face只提供已有采样face ID。不能因此编造“输入肯定充足”或“重新采样已经验证”。

![对象形状与条件](object_geometry.png)

## 6. 只推荐一个下一步，尚未执行

保留6000-update权重和现有10训练/2验证划分，下一项可检验用户提出的**depth1–5相对depth6–15为2:1的velocity loss权重**。不加条件模块、坐标prior或offset，不换loss目标、DiT、D15、Euler或阈值，不扩数据。depth1已经正确，提案目的是提高粗层整体优先级，不能称为修复depth1失败。

为了把实验尽量限制为相对层级权重，建议用全周期常数归一：

```python
# 提案，未修改训练入口、未执行
w = 2.0 if depth <= 5 else 1.0
loss_for_backward = loss * (w / (4.0 / 3.0)) / 8
```

平均权重由(5×2+10×1)/15=4/3归一到1，实际粗层1.5、细层0.75，保持2:1比例；不要每次按当前8个micro的权重和归一，那会随顺序改变相对权重。此处只固定整体权重尺度，并不保证梯度范数不变。

验收继续看完整树、首错层、TP/FP/FN和训练对象/未见对象分开统计，不能仅以MSE变小通过。应预先记录新增更新预算和有效UID/depth次数；如果只做这一条加权续训，变化同时包含“多训练了一段”的影响，不能没有同预算对照就证明提升由加权导致。本轮没有实施提案，也没有启动Phase3训练。

## 7. 文件与复现命令

- `results/per_depth_summary.csv`：全部15层micro/macro准确率、F1、首错、正确父格子集。
- `results/per_sample_depth.csv`：720行逐样本逐层计数和指标。
- `results/object_diagnostics.csv`：坐标点数/覆盖/对称诊断。
- `results/frozen_regression_groups.csv`、`frozen_regression_per_depth.csv`：冻结GT父格回归，含t分组与零速度/全空基线。
- `results/attention_groups.csv`、`condition_token_norms.csv`：粗细层attention、条件norm。
- `frozen_probe/`：真实冻结前向数组、attention每head平均token权重/每query熵、执行身份和哈希。没有完整每query的1024权重矩阵，不宣称能独立重算所有attention标量。
- `phase2_evidence/server/runtime/`：引用的冻结实际代码和输入；代码证据为`d15_code/mini_nexus/training.py:78–94`、`vertex.py:241–258,272–284`、`scripts/phase2_train_ssd_fast.py:291–315`。

本机已执行CPU命令（Python路径缩写，需numpy/matplotlib）：
```sh
python analyze_saved.py --source ../nexus_phase2_d15_10_20260926/final_evidence_20260928/server --output results
python summarize_probe.py
python write_report.py
```

解压交付包后CPU数组复算改为：
```sh
python analyze_saved.py --source phase2_evidence/server --output results
python summarize_probe.py
```

本轮实际GPU命令如下，仅用于记录，不会自动重跑：
```sh
CUDA_VISIBLE_DEVICES=GPU-1e134e5f-66eb-70f8-299a-53abdeda77a3 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /guohaoran/envs/nexus-algo/bin/python -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/phase3_coarse_analysis_20260928/frozen_condition_probe.py --root /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_phase2_d15_10_20260926 --output /ssdwork/guohaoran/nexus_fast_track/diagnostics/phase3_coarse_analysis_20260928/frozen_probe
```

CPU数组复算、GPU冻结前向、历史日志和静态代码判断在本文分开描述。以上均为NEXUS2K上的点云/法向条件任务，不是CAD50或老师文本模型成绩。原Phase2终验种子已公开使用，后续只能称固定回归种子，不能再称未见终验种子。
