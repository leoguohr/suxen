# 三条固定大mesh：自由32维Edge表示诊断

仅学习每条mesh自己的每顶点32维表示，不更新共享网络。保留原16+16评分、scale、逐mesh中心化、全部无向pair、fully-diff Soft4，目标为 Edge Soft4 / 4；fresh Adam LR=1e-3，clip=1，wd=0，每条独立2000步。

| UID | 顶点 | 起点FP/FN | 末尾FP/FN | 首次全对step | 全对更新数 | 最长连续 | 末500全对 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nexus_2k_000446 | 1519 | 1337/1 | 0/0 | 125 | 1876/2000 | 1876 | 500/500 |
| nexus_2k_001093 | 1964 | 7063/0 | 0/0 | 251 | 1750/2000 | 1750 | 500/500 |
| nexus_2k_000898 | 2547 | 13303/0 | 0/0 | 279 | 1722/2000 | 1722 | 500/500 |

成功只能证明这些独立32维表示在当前评分下存在并可被本次优化找到；不证明共享网络能够输出它们，不包含Face，也不代表100条联合重建通过。有限预算失败也不是维度不足的证明。

## 材料索引

- `snapshots/<uid>/`：网络原始head输出、中心化表示、原始顶点/GT面/边、全部pair标签/logits、直接表示梯度、当前有效评分与loss源码、父状态逐mesh评价。
- `runs/<uid>/updates.jsonl`：step0及2000次更新，完整TP/FP/FN/TN、F1、loss、margin、梯度、clip、实际位移。
- `runs/<uid>/checkpoint-*.pt/.npz`：固定检查点的表示、Adam及全部pair logits；若成功包含first-perfect。
- `runs/<uid>/baseline_verification.json`、`saved_checkpoint_verification.json`：网络→导出基线复现及首个成功/最终保存结果重载核验。
- `comparison.csv/json`、`curves.png/pdf`：三条比较。
- `export_complete.json`：原网络参数/缓冲/元数据与源checkpoint不变，零网络更新。

## 大文件与边界

包内包括三条原始mesh和表示探针checkpoint；不包含上游大网络checkpoint、100条完整数据、其他两条100-mesh分支checkpoint。源网络仍在服务器：
`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_face_hardneg_round2_20260917/run/checkpoint-update22500.pt`
SHA256: `428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79`

Control与第二轮难负例网络均未被覆盖；三条探针是独立变量，不可拼接成共享模型。

## 复现

`python run_probe.py --snapshot snapshots/<uid> --output <new-output>`，然后 `python verify_saved.py --snapshot snapshots/<uid> --run <new-output>`。需CUDA/PyTorch，具体版本见各manifest。

导出阶段曾因state_dict含非Tensor元数据使哈希检查报错；修正检查器后重试。该次错误发生在导出及任何优化前，原日志保留。
