# 100-mesh fresh512：完整评估包

本包取自已完成实验，不新增训练或模型前向。优先阅读本文件、training_evaluation.png、evaluation_trend.csv 和 per_mesh_final.csv。

## 完成状态与结果

- 5000个实际optimizer更新，200epochs；每条完整mesh参与200次；退出码0，按预算停止。
- 末尾严格成功：2/100，UID：nexus_2k_000341, nexus_2k_001045。
- 末尾 Edge FP/FN：1620842/679；实际 Face FP/FN：2989041/7413。
- 最终100条Face枚举全部完成。epoch0部分枚举超时，FP/F1为null；图中不将其当作0。
- 后半程错误总体减少，但仍有大量多余边、面；严格成功的UID并非在各检查点完全相同。不能宣布100条overfit成功，也不能凭有限预算失败宣布容量不足。

## 执行口径

- 同一个随机初始化模型：latent512、decoder hidden1024、Edge/Face评分各32维。μ重建，sampling关闭，KL=0；logvar参数冻结，四组fresh Adam。
- math00：Encoder和Decoder显式FP32 MATH、确定性Graph正反向、TF32/autocast关闭。manifest.args.precision的历史字符串不是实际backend判据；以manifest.backend、runtime安装和review_runtime源码为准。
- 每微批1条完整mesh；累积4条，各自(Edge Soft4+Face Soft4)/4，统一clip=1，然后一次Adam更新。每epoch100条无重复遍历，因此25更新/epoch。
- fully-differentiable Soft4：membership参与反传；每类软权重加权BCE/对应权重总量（epsilon=1e-8），四组固定平均。Edge全pair；Face固定positive+mixed池。公式及chunk归约见effective_functions.py.txt、loss_changes_exact_runtime.txt和common.py。
- E/μ LR第1步1e-6，第100步1e-5；D/head第1步1e-5，第100步1e-4，之后保持。无旧权重或Adam迁移；旧模型只用于离线Face负例池生成。
- 全量验收暂停更新，在同一checkpoint依次前向100条，不是packed100。Face来自当次预测Edge图；缺边导致未进入候选的GT Face仍计FN。阈值0。

## 材料索引

| 路径 | 内容 |
|---|---|
| training_evaluation.png | loss、实际错误、严格成功数曲线 |
| evaluation_trend.csv | 7轮全量验收汇总，分别记录训练pool loss与实际重建错误 |
| per_mesh_all_evaluations.csv | 700条逐mesh评估：规模、参与次数、loss、TP/FP/FN/F1、候选覆盖、margin与完整性 |
| per_mesh_final.csv | 末尾100条的完整汇总 |
| per_mesh_training_trace.csv | 20000条训练参与记录；变化batch上的训练loss不可当成固定集合的同口径趋势 |
| evidence/run/updates.jsonl | 全部5000条原始更新日志：四组梯度、clip、LR、实际位移、表示尺度 |
| evidence/run/eval-*.jsonl、eval-summary-*.json | 原始全量验收，不改变null或下界定义 |
| evidence/run/epoch-order-*.json | 每epoch固定保存的实际UID顺序 |
| evidence/run/manifest.json、complete.json、status.json | 运行配置、组定义、源文件哈希、预算完成证据 |
| evidence/runner_exit.json、train-console.log | 正常退出证据与完整控制台日志 |
| evidence/overfit100_manifest.csv、data_manifest.csv | 100条数据路径、规模、GT数目、哈希 |
| evidence/selection.json、excluded.csv、data_validation.json、pool_provenance.json | 预先选样规则、排除原因、完整数据/候选核验与池来源 |
| evidence/preflight/、preflight-console.log | 100条零更新资源预检、梯度累积核验 |
| evidence/train.py、runtime.py、evaluate.py、prepare.py | 本轮实际训练、数值路径、评估与pool准备入口 |
| evidence/source_archive/ | 与本次manifest记录逐项SHA核对通过的历史源文件快照 |
| evidence/review_runtime/ | 仅import运行时后解析出的依赖源码与有效函数；补充动态加载的代码依赖 |
| evidence/loss_changes_exact_runtime.txt、sampling_forward.py、graph_backward.py、C_graph_only/ | 实际动态替换公式与计算实现 |
| evidence/USER_PROTOCOL.md、README.md | 原协议与启动阶段说明；README是历史启动说明，当前完成状态看本文件和complete.json |
| checkpoint_inventory.json | 服务器checkpoint路径、大小及已记录的验收SHA |
| verification.json、SHA256SUMS.txt | 日志/计数/源文件一致性核验，包内逐文件哈希 |

## 范围与证据边界

这是用于审阅训练结果、目标函数及实际执行方式的评估材料包，不是离线重新运行推理的模型分发包。
大checkpoint（末尾含Adam约1.3GB）、原mesh二进制与177MB训练pool保留服务器，本包提供路径、已有数据哈希、生成规则和checkpoint索引；没有假装打入权重。
本轮原日志未保存所有候选逐条logit/ID、每顶点embedding或完整梯度张量，本包不凭空补造。margin为原日志中的各组最小值。
额外读取runtime只做import，不实例化模型、前向、反向或optimizer更新。训练时已有source/entry哈希全部重新核对；额外依赖的当前快照有独立SHA，但不能据此假装它们都有训练启动时的哈希。
原始记录保持原样，汇总CSV由build_review_package.py生成。CSV中空单元格对应原始null，不能当0。

服务器实验根目录：/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914
