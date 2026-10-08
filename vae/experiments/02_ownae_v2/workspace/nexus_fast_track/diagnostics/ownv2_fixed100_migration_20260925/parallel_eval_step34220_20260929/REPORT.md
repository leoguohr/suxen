# OwnAE-v2固定100条：step34220完整评测

2026-09-29已完成同一checkpoint的100条真实Encoder→μ→Decoder评价，以及预测数组和完整候选的独立CPU计数核验。新增optimizer更新为0，GPU评价耗时113.46秒。

相较step30360，联合严格成功23→28条；漏检明显减少，误报略增。Face micro-F1为99.60223878%，仍未达到99.7%阶段目标。

| 指标 | step30360 | step34220 |
|---|---:|---:|
| Edge micro-F1 | 99.96797578% | 99.98140287% |
| Edge FP / FN | 46 / 152 | 51 / 64 |
| 实际Face micro-F1 | 99.56569901% | 99.60223878% |
| 实际Face FP / FN | 1487 / 293 | 1508 / 123 |
| Edge严格成功 | 47/100 | 54/100 |
| Face严格成功 | 24/100 | 30/100 |
| Edge+Face联合严格成功 | 23/100 | 28/100 |

原23条联合成功保留19条、丢失4条、新增9条，不能声称旧成功全部保持。

- 丢失：nexus_2k_000074, nexus_2k_000142, nexus_2k_000766, nexus_2k_000840
- 新增：nexus_2k_000379, nexus_2k_000446, nexus_2k_000673, nexus_2k_000869, nexus_2k_001395, nexus_2k_001409, nexus_2k_001483, nexus_2k_001539, nexus_2k_001919

Face FN共123个，全部因缺少预测Edge而未进入实际候选；候选存在但判负为0。Face FP仍有1508个。Face依据当前预测Edge图完整、去重枚举，没有补入GT候选、截断或修复输出。

## 当前执行状态

用户此次提供的36910实例为`3i880tdt19oqq-0`，仅一张A100-80GB，GPU UUID `GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d`。进入时无GPU计算进程、无OwnAE训练进程，本轮完成一次独立评测，未执行训练更新。

共享训练日志最后更新于北京时间2026-09-28 19:55:08，到step34239；最新完整恢复点为step34220。状态文件中的training为旧记录，不能当作当前运行证据。step34221—34239没有对应的最新完整checkpoint记录，若恢复应从完整34220状态按原协议处理未落盘记录。

## 证据

- checkpoint：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_eval_step34220_20260929/source/checkpoint-step34220-original.pt`
- 大小：2960559104 bytes
- SHA256：`c9cd45dbd011c4adec9f3916bcdeb85dbe4bb2dcd3949db313356f1909f5ce95`
- 逐mesh指标：`analysis/per_mesh.csv`
- 比较及UID变化：`analysis/comparison.json`和`analysis/per_mesh_changes.json`
- 数组与候选完整性核验：`analysis/verification.json`
- 服务器评测目录：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_eval_step34220_20260929`

本轮没有ZIP交付，也未修改训练配置、数据或源码。结果属于确定性拓扑AE，不代表sampling/KL或顶点生成已通过。
