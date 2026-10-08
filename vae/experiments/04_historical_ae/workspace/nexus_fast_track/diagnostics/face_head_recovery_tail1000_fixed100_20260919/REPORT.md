# 固定Tail1000：100条共享Face head恢复

已完成500次新增Face-head更新并停止。起点为本轮末端Edge累计1000步的完整模型，保留当时Face head初始化；Encoder、μ、logvar、全部Decoder、最终LayerNorm和Edge head完全冻结。只有同一份32×1024线性Face head及bias训练，共32800参数。
沿用上轮Face恢复协议：fresh Adam，LR=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0，global clip=1；每次累积全部100条的完整固定pool，外层等权平均；fully-diff Soft4内部除4不变，无额外系数、MSE、Edge loss、sampling或KL。
源checkpoint：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_edge_continue500_20260919/run/model-tail1000-face500-inference.pt`；SHA256：`342fffbb7c2aaea4082dbbbde577757bb985909b4c613c56c4c7709a51c1fa63`。
从当前源模型重新导出安装后的hidden，FP32共435,507,200字节。原两轮扩充pool共750,215训练候选，当前固定预测Edge图实际产生504,475个Face候选。候选图不是GT图；1个缺边导致的GT候选缺失始终计FN。

| 新增更新 | Edge严格成功 | Face严格成功 | 联合严格成功 | 实际Face FP | 实际Face FN | Face Soft4均值 | 起点55条保留/丢失/新增 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 75 | 55 | 55 | 98872 | 744 | 1.106857627 | 55/0/0 |
| 50 | 75 | 63 | 63 | 69565 | 2186 | 0.283809103 | 54/1/9 |
| 100 | 75 | 68 | 68 | 62456 | 2864 | 0.195996117 | 55/0/13 |
| 200 | 75 | 62 | 62 | 54793 | 4160 | 0.160771006 | 52/3/10 |
| 300 | 75 | 72 | 72 | 64102 | 2866 | 0.150010049 | 55/0/17 |
| 400 | 75 | 73 | 73 | 55995 | 3937 | 0.136395719 | 55/0/18 |
| 500 | 75 | 74 | 74 | 55260 | 4043 | 0.132400575 | 55/0/19 |

末尾同一模型联合成功74/100。起点55条保留55、丢失0、新增19。与较早的71条联合成功基线相比，保留71、缺失0、新增3。
前轮丢失的16条中恢复16条，前轮新增的4条Edge成功中有3条转化为联合成功。step300、400、500三个检查点的共同联合成功集合为72条；没有把未验收步骤声称为全部保持。

全部7次检查，Edge保持75/100、FP130832、FN1。固定当前边图下联合成功最多75条；Face head本轮结果不能解决其他25条的Edge错误，也不代表100条全部通过。

## 四条新Edge成功样本

| UID | 顶点 | 起点Face FP/FN | 末尾Face FP/FN | 末尾联合成功 |
|---|---:|---|---|---|
| nexus_2k_000520 | 1734 | 53/3 | 0/0 | True |
| nexus_2k_001317 | 1536 | 29/29 | 8/11 | False |
| nexus_2k_001764 | 1416 | 4/18 | 0/0 | True |
| nexus_2k_001825 | 1277 | 0/57 | 0/0 | True |

## 核验与材料范围

当前源模型的全部100条真实forward与前轮保存的hidden、Edge/Face embedding逐位相同；缓存训练与真实网络的Face表示、完整pool loss和Face-head梯度逐位一致。起点两次完整100条backward也逐位一致。
末尾重载Face checkpoint，再从真实mesh输入完整推理100条，核对hidden、Edge表示、Face表示和实际重建复现缓存结果。所有冻结权重、buffer、metadata，训练pool、cached hidden、预测Edge图及实际Face候选均保持不变。
独立读取落盘的源/派生完整模型，只有Face head两个参数允许变化；初始Adam确为空、末尾Adam为500步。保存的逐候选Face logits与标签已独立重算全部100条计数，GT缺失候选仍按FN处理。
Face loss基于固定训练pool，不代表全部实际候选上的loss。较低loss不替代实际Edge/Face验收。该实验没有自动延长，也没有覆盖旧71条基线或本轮Edge候选模型。

## 文件索引

- run/updates.jsonl：完整500条更新及每步100条Face loss、梯度、clip、实际位移。
- run/evaluation-step*.json、evaluations.jsonl：7×100条实际Edge/Face、训练pool、候选覆盖与margin；trend.csv和per_mesh_evaluations.csv方便比较。
- retention.json追踪本轮起点55条；historical_joint71_retention.json追踪旧71条、曾丢失16条与新增Edge4条；initial_edge_perfect_face_failed20.csv列出起点20条待恢复样本。
- run/checkpoint-step*.pt：每个检查点的同一Face head及本轮新Adam；run/final_real_network.json为末尾真实网络复核。
- restore/freeze信息见freeze_contract.json、cache/manifest.json、run/verification.json、checkpoint_file_verification.json；saved_logits_recount.json为独立计数复核。
- runtime.py、face_core.py逐字复用上轮；protocol_adaptation.diff仅展示来源核验与结果路径适配。effective_face_scoring.py.txt及runtime_dependencies保留实际运行定义。
- Review包含完整日志、代码、Face checkpoint/Adam、末尾Face表示与logits。FullEvidence另含全部新hidden、Edge表示、原mesh、固定训练/实际候选ID、label和预测Edge ID。
- 原完整网络与派生完整网络不入ZIP，路径/SHA在EXCLUDED_FILES.json；Review省略的缓存也逐文件列明位置与SHA。

末尾完整推理模型：`/guohaoran/nexus_fast_track/diagnostics/face_head_recovery_tail1000_fixed100_20260919/run/model-tail1000-face1000-inference.pt`。SHA256：`0816b3f7358f7a39fc1bd1fbe70ea338a726a0c89fe9ecafc777bbd9b3868170`。
