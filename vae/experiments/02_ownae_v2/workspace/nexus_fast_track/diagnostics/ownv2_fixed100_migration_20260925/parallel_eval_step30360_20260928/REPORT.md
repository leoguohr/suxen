# OwnAE-v2 固定100条：step30360完整评价

已完成同一不可变checkpoint、真实Encoder→μ→Decoder的全部100条实际重建评价。相较step24920，联合严格成功12→23条，Edge和Face的FP/FN均减少；Face micro-F1仍未达到0.997阶段目标。

| 指标 | step24920 | step30360 |
|---|---:|---:|
| Edge micro-F1 | 99.90921933% | 99.96797578% |
| Edge FP / FN | 73 / 488 | 46 / 152 |
| 实际 Face micro-F1 | 99.39475980% | 99.56569901% |
| 实际 Face FP / FN | 1538 / 939 | 1487 / 293 |
| Edge严格成功 | 31/100 | 47/100 |
| Face严格成功 | 12/100 | 24/100 |
| Edge+Face联合严格成功 | 12/100 | 23/100 |

原12条联合成功保留10条、丢失2条、新增13条，不能描述为全部旧成功均保持。丢失UID：nexus_2k_000673, nexus_2k_001409。

Face FN=293，其中292个因缺Edge而未进入实际候选，1个已进入候选但判负。实际Face从本次预测Edge图完整枚举，判正logit>0，没有GT补边、候选截断或输出修复。

100条预测数组、分片SHA、所有Edge混淆计数、Face计数、三角形完整性及候选外GT FN均经独立CPU核对。评价没有optimizer更新，耗时103.37秒。

## 训练恢复

本次新实例GPU0从30360完整恢复模型、AdamW、RNG和遍历状态，未改变训练配方。旧日志中30361—30371的11次未保存更新已归档；恢复后重放的UID、负例哈希、loss及梯度范数全部一致，不重复计入模型步数。训练继续运行；GPU1只完成本次单独评价，没有持续监督器或评测循环。

## 证据与保存点

- 评价源：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_eval_step30360_20260928/source/checkpoint-step30360-original.pt`
- 大小：2960559104 bytes
- SHA256：`1dba1581dbc06d05c1a7c6652201b8494af9af0f723abeb8ba72ed6b3067d4da`
- 服务器评价目录：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_eval_step30360_20260928`
- 本地逐mesh表：`analysis/per_mesh.csv`
- 对比及UID变化：`analysis/comparison.json`、`analysis/per_mesh_changes.json`
- 原始数组核验：`analysis/verification.json`

按要求未打包ZIP。当前结果属于确定性拓扑AE，不是已通过sampling/KL的VAE，也不是顶点生成结果。
