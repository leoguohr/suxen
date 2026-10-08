# Decoder末端Edge续训：新增500次更新

71条联合成功的模型完整保留；独立分支沿原末端Edge step500的权重和Adam继续到tail step1000。每次100条共同参与，mean100(fully-diff Edge Soft4)，全84,669,234 pair；仅Decoder末块＋最终LayerNorm＋Edge head更新。LR=1e-5/1e-4，clip=1，wd=0，无warmup，无Face loss、KL或sampling。
Face head使用恢复后的Face-step500权重并冻结，没有加载其新Adam。初始尾部权重在Edge-step500与71条完整成功模型之间逐位相同；初始Adam一二阶矩、step及CPU/CUDA RNG与原Edge-step500完全一致。
所有训练步骤仍使用缓存的末块输入，不是最终hidden。起点与末尾均从真实输入重跑100条，核验缓存结果；实际Face在每个检查点从该点预测Edge图重新枚举，没有沿用旧候选图。

| 新增更新 | 末端累计更新 | Edge严格成功 | 联合严格成功 | Edge FP/FN | 实际Face FP/FN |
|---:|---:|---:|---:|---|---|
| 0 | 500 | 71 | 71 | 156784/1 | 49411/984 |
| 50 | 550 | 72 | 69 | 158754/1 | 59367/738 |
| 100 | 600 | 72 | 68 | 159441/1 | 67753/607 |
| 200 | 700 | 72 | 64 | 154475/1 | 78735/587 |
| 300 | 800 | 73 | 63 | 148671/1 | 87538/650 |
| 400 | 900 | 74 | 58 | 138134/1 | 92132/744 |
| 500 | 1000 | 75 | 55 | 130832/1 | 98872/744 |

## 固定组的Edge进展

| 组 | 条数 | 起点FP/FN | 末尾FP/FN | 起点/末尾严格成功 |
|---|---:|---|---|---|
| all100 | 100 | 156784/1 | 130832/1 | 71/75 |
| initial_failed29 | 29 | 156784/1 | 130832/1 | 0/4 |
| N_gt1500 | 33 | 151457/1 | 129170/1 | 7/9 |
| initial_success71 | 71 | 0/0 | 0/0 | 71/71 |

原29条失败样本中，29条FP减少、0条不变、0条增加；其中4条末尾Edge严格成功。

原71条Edge成功：末尾保留71、丢失0、新增4。UID清单见retention.json。
原71条联合成功：末尾保留55、丢失16、新增0。UID清单见retention.json。

共享Decoder末端更新后，Face embedding会变化，即使Face head权重不变。Edge收益不能自动继承原联合成功标签；本轮按实际Face重新验收。有限预算内仍有错误不构成容量不可能的证明。本轮已停止，没有刷新pool、扩大解冻范围或自动延长。

## 文件索引及核验范围

- run/updates.jsonl：500条更新、分组实际位移、梯度、clip和LR；run/edge_trace.jsonl：501个状态×100条全部Edge计数、loss及margin。
- run/actual-new*.json、actual_evaluations.jsonl、actual_per_mesh.csv：7次完整实际Edge/Face与GT候选覆盖；Face pool仅作诊断。
- edge_group_trace.csv：预先固定的原29条、>1500顶点组与原71成功组；edge_per_mesh_comparison.csv给出逐UID变化。
- restore_verification.json：恢复原Edge Adam，不使用Face Adam；run/verification.json：冻结、缓存及真实网络重接检查。
- cache/manifest.json保留最初epoch900导出的旧baseline字段，用于缓存来源核验；本轮真正step0见run/actual-new0000.json和edge_trace首条，不能把旧缓存baseline当作本轮起点。
- independent_audit.json：读取持久化起点/末尾checkpoint独立验证Adam、RNG和10个可变张量边界，并重算全部84,669,234个末尾Edge logits。
- run/checkpoint-new*.pt：尾部三模块及Adam/RNG；末尾为new0500-tail1000，不把Face训练更新计入Edge累计。
- actual候选枚举没有超时截断。最终Face候选逐条logit未全部导出；完整计数、评分表示和实际执行枚举代码保留。
- FullEvidence另含缓存末块输入、原mesh、末尾hidden/Edge/Face表示和全部Edge logits、中间尾部checkpoint。Review保留日志、代码、初末尾尾部权重/Adam和核验结果。
- 大型原完整模型与本轮派生完整推理模型不入ZIP；位置和SHA见EXCLUDED_FILES.json。原缓存与所有旧分支保持不变。

保留71条基线：`/guohaoran/nexus_fast_track/diagnostics/face_head_recovery_fixed100_20260918/run/model-tail500-face500-inference.pt`，SHA256 `8de6074700fafa02b3adcfafd352a4555ce0cbc88421667261f29495faa69640`。
本轮候选完整模型：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_edge_continue500_20260919/run/model-tail1000-face500-inference.pt`，SHA256 `342fffbb7c2aaea4082dbbbde577757bb985909b4c613c56c4c7709a51c1fa63`。
