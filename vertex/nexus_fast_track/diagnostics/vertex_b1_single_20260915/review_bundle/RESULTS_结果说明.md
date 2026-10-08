# 最终结果：R1明显改善，严格B1验收仍未通过

两组训练已结束，未被信号中断。R0跑完500更新，未满足延长条件；R1满足预定改善条件后延长至1000更新。没有再增加预算。模型权重不在此包，按用户要求仅保留服务器。

| 最终指标 | R0，500更新 | R1，1000更新 |
|---|---:|---:|
| 验证平均velocity MSE | 1.12475316 | 0.00190335 |
| 验证最差velocity MSE | 1.42115808 | 0.00679483 |
| 精确整数坐标集合 | 0/16 | 16/16 |
| 实际训练缓存MSE | 0.99390203 | 0.00122413 |
| 缓存坐标集合精确 | 否 | 是 |
| FP32响应误差全部<=0.1 | 否 | 否，3/4达标 |
| 严格passed | false | false |
| 64份独立holdout | 未触发 | 未触发 |

R1最终4个signed gain为[-1.9808003,-1.9993193,-1.9261805,-1.9956918]，均在[-2.2,-1.8]。对应响应误差为[0.0361623,0.0418965,0.1214020,0.0379879]，第三例seed8000002仍超0.1。因此不是MSE或顶点坐标阻止最终验收，而是一个响应误差门槛。不能将校准验证16/16称作64个未参与调参的新噪声通过。

R1在400步首次16/16坐标精确，在850步首次所有验证MSE<=.01。最终结果是1000步，未挑选中途最优步骤替代最终结果。R0/R1前500步的实际噪声SHA序列逐项相同。

## 实验究竟训练了什么

两组都只训练和验收nexus_2k_000105（8顶点），完整2,332,430,344参数模型；固定实际点云与法向、depth9、GT parents和0/1占据，t=.5。每次更新8份独立新噪声。两组均从头初始化和新建AdamW，无A权重。LR1e-5、WD0、warmup100、clip1，BF16 autocast/FP32参数。实际循环UID和输入噪声哈希均有逐步记录。

R0原始初始化。R1仅组合修改depth embedding std=.02、time MLP两层std=.02/bias0、cross-attention输出weight/bias0，其他模型结构和loss不变。该对照支持这组初始化在当前协议中显著改善噪声学习；不能分离归因到某一个初始化组件，也不能外推多物体、随机时间或完整生成。

最终seed8000000的FP32特征RMS：R0首/尾block约1.4783/17.4001，R1约0.1856/0.2159。这是记录到的相关变化，不是唯一根因证明；逐token能量和其他层见responses_fp32.layers。

training_cache来自第1更新第0个microbatch真正使用的输入，后续读取保存的xt和target_velocity评估。该输入只参与过一次训练，并非反复优化同一缓存问题；因此它是已见输入诊断，不等同阶段A的固定回归overfit。初始step0还未训练，cache_probe=null正常。

## 文件导航与离线复算

- run/B1_R0、run/B1_R1：config、train.jsonl、全部evaluation、缓存张量及哈希、各step预测NPZ、result与延长决策。
- 每份evaluation内含training_config、scope实际消费UID、缓存probe、16个验证probe及FP32响应。TP/FP/FN位于occupancy，顶点数位于cells。
- run/status.json与run/result.json：进程结束状态与两组门槛结果。complete不是passed。
- data/：实际点云、量化GT、octree、topology；manifest只有该UID。manifest里的绝对路径是当时运行路径，迁移运行时需重写路径。
- code/：运行时冻结源码及相关测试。sha256.json对照部署的64个源码/数据文件。
- console.log、launch.json：运行输出与主机/命令/环境身份，无密码。
- evidence_sha256.json、MANIFEST_SHA256.json：原始证据与整包哈希。
- verify_results.py、verification.json：只需Python和numpy，解压后运行 python verify_results.py。它复算512个保存probe、全部缓存probe、计数和坐标集合，核对UID/噪声SHA/报告门槛。它不加载模型，也无法从未保存的原始扰动张量重算FP32 signed gain；该部分只核验已记录数值与门槛。

全空基线实际MSE约0.5，预测0顶点，不能算恢复形状。每个probe均保存该对照。

## 环境与边界

本地34测试通过。服务器默认CPU线程环境下两次各33/34通过，旧overfit CPU子进程signal11；限制OMP/OPENBLAS/MKL线程后34通过，正式运行使用同样环境，底层原因未确认。测试日志全部保留。实际两组训练完成且loss有限。

未执行64独立噪声验收、B2随机时间、单层纯噪声ODE采样、完整多层八叉树生成或多物体条件验证。此次包只对应2026-09-15独立单mesh B1，不混入旧A100数据。
