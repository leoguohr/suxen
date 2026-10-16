# topology: T2 gate 3

本地证据更新时间：2026-10-16T20:35:16+08:00。

## 来源与执行状态

| 项目 | 记录 |
| --- | --- |
| 源码 commit | 205c1b7d24b7321debe08feb8fec127d52d94a2b |
| Vertex VEVAL | /guohaoran/tmp/nexus_vertex_round3_batch2_20261015/round3_train/K2_J5_lr5e-6_d0.5/evals/u004500_raw |
| Vertex full_trees_exact | 92 |
| CPU gate | CPU_GATE_OK (14 tests) |
| 执行补丁 | train.py: serialize init_checkpoint Path in config JSON; no training semantics changed |

| 实际执行边界 | 记录 |
| --- | --- |
| GPU 分配 | {"0": "gt_reference then B; cooldown raw after completion", "1": "A; cooldown ema after completion", "2": "B cooldown single GPU", "3": "unused fallback"} |
| Cooldown 实际入口 | GPU 2 single GPU; T1 adapter does not accept --init-checkpoint; AdamW state retained |
| 执行上限 | 4 小时墙钟；最多 16 GPU 小时；截止 2026-10-17T00:04:04+08:00（新加坡时间） |
| 存储边界 | /guohaoran (GPFS)；单个新增完整检查点约 34.8 GiB，原子替换峰值约 69.6 GiB；个人配额：unknown; quota and mmlsquota unavailable |
| 检查点保留 | cooldown latest.pt then final.pt; one old copy retained until atomic replacement; parent checkpoints unchanged |
| VEVAL 文件清单 SHA256 | 0e75c07658cbb8a8106c93187dcaa8f9fdcbb453f980aa1132ac3244332e76c3 |
| 输入 checkpoint 重新哈希记录 | input_checkpoint_sha256.json |

生成使用 Vertex 预测 D9 cells 与原点云条件；strict 和 one_cell 均为生成后的匹配评价。每个 flow seed 包含 2 个 vertex seeds × 50 条，共 100 棵树；GT reference 为 50 条。exact/inexact 划分以批准的 Topology GT 为准，001825 使用 1277 点版本。

| 运行 | 状态 | 核验 |
| --- | --- | --- |
| gt_reference_B | 完成 | 行数、唯一 UID/seed、TP/FP/FN 与汇总一致 |
| cascade_B | 完成 | 行数、唯一 UID/seed、TP/FP/FN 与汇总一致 |
| cascade_A | 完成 | 行数、唯一 UID/seed、TP/FP/FN 与汇总一致 |
| B_cooldown_raw | 未开始或未拉取 | 缺少 results.json / 尚未完成; cascade_flow0: 缺少 summary.json; cascade_flow1: 缺少 summary.json |
| B_cooldown_ema | 未开始或未拉取 | 缺少 results.json / 尚未完成; cascade_flow0: 缺少 summary.json; cascade_flow1: 缺少 summary.json |

## gt_reference_B

状态：完成。

Checkpoint：`/guohaoran/nexus_topology_flow_overfit50_20261015/pair_resume_20261016/B_s0warm/checkpoint-u004000-finalstage.pt`；weights：`raw`；采样：euler 20 steps，bf16。

Vertex GT cells 差异：`[{"uid": "nexus_2k_001825", "vertex_gt_cells": 1279, "topology_gt_cells": 1277, "only_in_vertex_gt": 2, "only_in_topology_gt": 0}]`。

预设自检目标 Face TP/FP/FN：102811/945/79；以下是实际记录。

| 自检项 | Expected | Actual | Actual − expected |
| --- | --- | --- | --- |
| Face TP | 102811 | 102811 | 0 |
| Face FP | 945 | 945 | 0 |
| Face FN | 79 | 79 | 0 |

计数自检：完全一致。

| flow | 匹配 | 分组 | 树数 | Edge TP/FP/FN | Edge micro-F1 | Face TP/FP/FN | Face micro-F1 | Joint strict | Overflow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gt_reference_flow0 | strict | all | 50 | 154944/183/34 | 0.99930024 | 102811/945/79 | 0.99504467 | 11 | 0 |
| gt_reference_flow0 | strict | exact_vertex_trees | 50 | 154944/183/34 | 0.99930024 | 102811/945/79 | 0.99504467 | 11 | 0 |
| gt_reference_flow0 | strict | inexact_vertex_trees | 0 | 0/0/0 | 0.00000000 | 0/0/0 | 0.00000000 | 0 | 0 |
| gt_reference_flow0 | one_cell | all | 50 | 154944/183/34 | 0.99930024 | 102811/945/79 | 0.99504467 | 11 | 0 |
| gt_reference_flow0 | one_cell | exact_vertex_trees | 50 | 154944/183/34 | 0.99930024 | 102811/945/79 | 0.99504467 | 11 | 0 |
| gt_reference_flow0 | one_cell | inexact_vertex_trees | 0 | 0/0/0 | 0.00000000 | 0/0/0 | 0.00000000 | 0 | 0 |

## cascade_B

状态：完成。

Checkpoint：`/guohaoran/nexus_topology_flow_overfit50_20261015/pair_resume_20261016/B_s0warm/checkpoint-u004000-finalstage.pt`；weights：`raw`；采样：euler 20 steps，bf16。

| flow | 匹配 | 分组 | 树数 | Edge TP/FP/FN | Edge micro-F1 | Face TP/FP/FN | Face micro-F1 | Joint strict | Overflow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cascade_flow0 | strict | all | 100 | 309832/363/124 | 0.99921471 | 205565/1870/215 | 0.99495420 | 23 | 0 |
| cascade_flow0 | strict | exact_vertex_trees | 90 | 261867/277/54 | 0.99936840 | 174000/1566/122 | 0.99517284 | 23 | 0 |
| cascade_flow0 | strict | inexact_vertex_trees | 10 | 47965/86/70 | 0.99837645 | 31565/304/93 | 0.99375069 | 0 | 0 |
| cascade_flow0 | one_cell | all | 100 | 309838/357/118 | 0.99923406 | 205571/1864/209 | 0.99498324 | 23 | 0 |
| cascade_flow0 | one_cell | exact_vertex_trees | 90 | 261867/277/54 | 0.99936840 | 174000/1566/122 | 0.99517284 | 23 | 0 |
| cascade_flow0 | one_cell | inexact_vertex_trees | 10 | 47971/80/64 | 0.99850134 | 31571/298/87 | 0.99393958 | 0 | 0 |
| cascade_flow1 | strict | all | 100 | 309834/352/122 | 0.99923566 | 205574/1869/206 | 0.99497850 | 23 | 0 |
| cascade_flow1 | strict | exact_vertex_trees | 90 | 261866/266/55 | 0.99938747 | 174001/1557/121 | 0.99520133 | 23 | 0 |
| cascade_flow1 | strict | inexact_vertex_trees | 10 | 47968/86/67 | 0.99840773 | 31573/312/85 | 0.99375226 | 0 | 0 |
| cascade_flow1 | one_cell | all | 100 | 309840/346/116 | 0.99925501 | 205580/1863/200 | 0.99500754 | 23 | 0 |
| cascade_flow1 | one_cell | exact_vertex_trees | 90 | 261866/266/55 | 0.99938747 | 174001/1557/121 | 0.99520133 | 23 | 0 |
| cascade_flow1 | one_cell | inexact_vertex_trees | 10 | 47974/80/61 | 0.99853261 | 31579/306/79 | 0.99394111 | 0 | 0 |

cascade_flow0：strict Face FP+FN 最大的 10 棵树（并列按 UID、vertex seed 排序）。

| UID | Vertex seed | Exact vertices | Face FP | Face FN | FP+FN |
| --- | --- | --- | --- | --- | --- |
| nexus_2k_001415 | 97029001 | False | 161 | 7 | 168 |
| nexus_2k_001415 | 97029000 | True | 160 | 2 | 162 |
| nexus_2k_000800 | 97029001 | True | 118 | 15 | 133 |
| nexus_2k_000800 | 97029000 | True | 121 | 7 | 128 |
| nexus_2k_001317 | 97029001 | True | 100 | 2 | 102 |
| nexus_2k_001317 | 97029000 | True | 96 | 0 | 96 |
| nexus_2k_000872 | 97029001 | True | 94 | 0 | 94 |
| nexus_2k_000872 | 97029000 | True | 91 | 2 | 93 |
| nexus_2k_000083 | 97029001 | False | 49 | 9 | 58 |
| nexus_2k_000083 | 97029000 | True | 45 | 10 | 55 |

cascade_flow1：strict Face FP+FN 最大的 10 棵树（并列按 UID、vertex seed 排序）。

| UID | Vertex seed | Exact vertices | Face FP | Face FN | FP+FN |
| --- | --- | --- | --- | --- | --- |
| nexus_2k_001415 | 97029001 | False | 168 | 6 | 174 |
| nexus_2k_001415 | 97029000 | True | 158 | 0 | 158 |
| nexus_2k_000800 | 97029001 | True | 122 | 11 | 133 |
| nexus_2k_000800 | 97029000 | True | 117 | 8 | 125 |
| nexus_2k_001317 | 97029000 | True | 95 | 2 | 97 |
| nexus_2k_001317 | 97029001 | True | 97 | 0 | 97 |
| nexus_2k_000872 | 97029000 | True | 94 | 1 | 95 |
| nexus_2k_000872 | 97029001 | True | 92 | 0 | 92 |
| nexus_2k_000083 | 97029001 | False | 52 | 12 | 64 |
| nexus_2k_000242 | 97029001 | True | 58 | 0 | 58 |

## cascade_A

状态：完成。

Checkpoint：`/guohaoran/nexus_topology_flow_overfit50_20261015/pair_resume_20261016/A_fresh/checkpoint-u004000-finalstage.pt`；weights：`raw`；采样：euler 20 steps，bf16。

| flow | 匹配 | 分组 | 树数 | Edge TP/FP/FN | Edge micro-F1 | Face TP/FP/FN | Face micro-F1 | Joint strict | Overflow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cascade_flow0 | strict | all | 100 | 309810/607/146 | 0.99878621 | 205530/2243/250 | 0.99397175 | 16 | 0 |
| cascade_flow0 | strict | exact_vertex_trees | 90 | 261844/488/77 | 0.99892228 | 173960/1881/162 | 0.99416224 | 16 | 0 |
| cascade_flow0 | strict | inexact_vertex_trees | 10 | 47966/119/69 | 0.99804411 | 31570/362/88 | 0.99292342 | 0 | 0 |
| cascade_flow0 | one_cell | all | 100 | 309816/601/140 | 0.99880556 | 205536/2237/244 | 0.99400077 | 16 | 0 |
| cascade_flow0 | one_cell | exact_vertex_trees | 90 | 261844/488/77 | 0.99892228 | 173960/1881/162 | 0.99416224 | 16 | 0 |
| cascade_flow0 | one_cell | inexact_vertex_trees | 10 | 47972/113/63 | 0.99816896 | 31576/356/82 | 0.99311212 | 0 | 0 |
| cascade_flow1 | strict | all | 100 | 309817/612/139 | 0.99878946 | 205545/2259/235 | 0.99396979 | 15 | 0 |
| cascade_flow1 | strict | exact_vertex_trees | 90 | 261853/482/68 | 0.99895089 | 173978/1880/144 | 0.99421681 | 15 | 0 |
| cascade_flow1 | strict | inexact_vertex_trees | 10 | 47964/130/71 | 0.99790906 | 31567/379/91 | 0.99261053 | 0 | 0 |
| cascade_flow1 | one_cell | all | 100 | 309823/606/133 | 0.99880880 | 205551/2253/229 | 0.99399880 | 15 | 0 |
| cascade_flow1 | one_cell | exact_vertex_trees | 90 | 261853/482/68 | 0.99895089 | 173978/1880/144 | 0.99421681 | 15 | 0 |
| cascade_flow1 | one_cell | inexact_vertex_trees | 10 | 47970/124/65 | 0.99803389 | 31573/373/85 | 0.99279920 | 0 | 0 |

cascade_flow0：strict Face FP+FN 最大的 10 棵树（并列按 UID、vertex seed 排序）。

| UID | Vertex seed | Exact vertices | Face FP | Face FN | FP+FN |
| --- | --- | --- | --- | --- | --- |
| nexus_2k_001415 | 97029001 | False | 160 | 5 | 165 |
| nexus_2k_001415 | 97029000 | True | 154 | 0 | 154 |
| nexus_2k_000800 | 97029001 | True | 135 | 14 | 149 |
| nexus_2k_000800 | 97029000 | True | 126 | 9 | 135 |
| nexus_2k_001317 | 97029000 | True | 110 | 1 | 111 |
| nexus_2k_001317 | 97029001 | True | 105 | 4 | 109 |
| nexus_2k_000872 | 97029000 | True | 100 | 4 | 104 |
| nexus_2k_000872 | 97029001 | True | 101 | 2 | 103 |
| nexus_2k_000935 | 97029000 | True | 61 | 2 | 63 |
| nexus_2k_000935 | 97029001 | True | 57 | 4 | 61 |

cascade_flow1：strict Face FP+FN 最大的 10 棵树（并列按 UID、vertex seed 排序）。

| UID | Vertex seed | Exact vertices | Face FP | Face FN | FP+FN |
| --- | --- | --- | --- | --- | --- |
| nexus_2k_001415 | 97029001 | False | 159 | 6 | 165 |
| nexus_2k_001415 | 97029000 | True | 157 | 2 | 159 |
| nexus_2k_000800 | 97029001 | True | 138 | 8 | 146 |
| nexus_2k_000800 | 97029000 | True | 127 | 16 | 143 |
| nexus_2k_001317 | 97029001 | True | 112 | 4 | 116 |
| nexus_2k_000872 | 97029000 | True | 102 | 3 | 105 |
| nexus_2k_001317 | 97029000 | True | 97 | 3 | 100 |
| nexus_2k_000872 | 97029001 | True | 95 | 1 | 96 |
| nexus_2k_000083 | 97029001 | False | 61 | 11 | 72 |
| nexus_2k_000935 | 97029001 | True | 58 | 5 | 63 |

## B_cooldown_raw

状态：未开始或未拉取。

## B_cooldown_ema

状态：未开始或未拉取。

## Cooldown 训练阶段评价

状态：`{"state": "training", "update": 50, "last_loss": 0.08796004205942154, "max_updates": 1000}`。

| 初始化来源核验 | 实际配置记录 |
| --- | --- |
| parent_update | 4000 |
| optimizer_loaded | True |
| ema_from_raw | True |

尚无已拉取的 u500/u1000 评价结果。

训练阶段评价使用 GT 顶点，Gate 3 使用 Vertex 生成顶点，二者分开报告。未完成组不用于全量成功结论，joint strict 不等同于 micro-F1 达标。

小文件证据与 SHA256 清单：[`evidence/topology-t2-20261016/manifest.json`](../evidence/topology-t2-20261016/manifest.json)，共 25 个文件。
