# 三条固定大mesh：单一共享Edge head诊断

仅优化同一份32×1024权重与32维bias。三条hidden均固定，来自同一epoch900源checkpoint的Decoder末端LayerNorm输出。没有回归指定成功表示，没有每mesh独立权重，没有Face或KL。

每步三条共同参与，目标为 mean_mesh(Edge Soft4)/4；fresh Adam LR=0.0001，clip=1，wd=0，预算2000步。学习率是本轮预定候选设置，不代表最优。

| UID | 初始FP/FN | 最终FP/FN | 最终Edge Soft4 |
|---|---:|---:|---:|
| nexus_2k_000446 | 1337/1 | 0/0 | 2.96668e-05 |
| nexus_2k_001093 | 7063/0 | 0/0 | 0.00014818118 |
| nexus_2k_000898 | 13303/0 | 0/0 | 0.00035777537 |

首次三条同时严格成功：592；总共1409次；最长连续1409步；最后500步中500次。

成功仅证明当前固定hidden对这三条存在本次训练找到的共同Edge读出；不代表Face或100条联合任务通过。有限预算失败也不能证明特征或容量不足。

## 材料

- `snapshots/head_initial.npz`：唯一原始共享head权重与bias。
- `snapshots/<uid>/hidden.npy`：每条固定Decoder hidden；配套全部pair标签、原logits、原head输出、GT结构与有效评分源码。
- `run/updates.jsonl`：逐步三条同一head计数、margin、loss、参数实际更新、梯度、clip和权重范数。
- `run/checkpoint-*.pt`：每个检查点仅一份head和Adam；配套三条NPZ保存实际读出与全部pair logits。
- `run/baseline_verification.json`、`run/verification.json`：基线复现与保存后重载全量验收。
- `export_complete.json`：源网络未更新；未向原网络checkpoint写入新head。

不含上游大网络checkpoint或100条完整训练数据。源checkpoint：
`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_face_hardneg_round2_20260917/run/checkpoint-update22500.pt`
SHA256: `428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79`

原网络重载并安装新head的完整推理未纳入本诊断；这里核验固定hidden上的单一共享读出。
