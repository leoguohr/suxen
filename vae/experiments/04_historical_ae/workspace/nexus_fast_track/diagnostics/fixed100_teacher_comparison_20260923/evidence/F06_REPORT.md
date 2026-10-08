# 共享 Edge head 接回网络：100条无更新验收

源模型为第二轮固定Face难负例epoch900/update22500；替换为三条共享head诊断step2000的同一份32×1024权重与32维bias。仅修改两个state_dict键，保留所有其他权重、buffer、metadata。新增optimizer update=0，backward=0。

先检查指定三条完整Encoder→μ→Decoder前向，再对固定100条逐条完整验收。所有Face都由各自预测Edge图枚举；旧的30秒枚举时间限制取消以完成全部候选，枚举、scoring及chunk顺序未变。

| 指标 | 源模型 | 安装共享head后 |
|---|---:|---:|
| Edge严格成功 | 50 | 5 |
| Edge＋实际Face严格成功 | 47 | 2 |
| Edge FP | 209400 | 59492 |
| Edge FN | 2 | 17635 |
| 实际Face FP | 10322 | 6768 |
| 实际Face FN | 980 | 30614 |

## 指定三条

| UID | Edge FP/FN 前→后 | 实际Face FP/FN 前→后 |
|---|---|---|
| nexus_2k_000446 | 1337/1 → 0/0 | 56/5 → 0/5 |
| nexus_2k_001093 | 7063/0 → 0/0 | 200/4 → 0/4 |
| nexus_2k_000898 | 13303/0 → 0/0 | 180/46 → 0/46 |

三条真实网络Edge logits与探针逐位一致、重复forward逐位一致。全部100条μ、logvar、Decoder hidden、Face embedding和固定Face训练pool的logits在替换前后均逐位不变。原模型重跑的100条loss、硬计数与margin复现归档。

原联合成功47条：保留2条，丢失45条，新增0条；安装后联合成功2/100。具体UID在comparison.json。

Face训练pool分数不变不等于实际Face结果不变：新Edge图改变候选进入资格。完整的FP/FN和缺失GT Face候选数保留在逐mesh记录。

## 判读边界

本次验证固定三条的共同读出可以接回真实网络；对其余97条的效果以本次全量记录为准。该head仅在预先指定三条上训练，并非重新优化100条。保留原模型和所有历史分支；本次不自动将派生模型设为主线，不新增训练预算。

## 包含与未包含

- run/before.jsonl、after.jsonl：两份同源模型各100条完整Edge/实际Face计数、loss、margin、候选覆盖和特征哈希。
- run/three_verification.json、installation_verification.json、complete.json：三条接回、模型重载及零更新核验。
- per_mesh_comparison.csv、comparison.json：逐mesh变化、原成功保留/丢失/新增UID。
- run/three-features-*.npz：三条当前μ、hidden、Face embedding以及替换前后Edge embedding。
- head/checkpoint-step2000.pt：同一成功共享head及其诊断历史状态，仅供溯源；本次未使用优化器。
- source_archive、review_runtime、runtime_dependencies、effective_scoring及入口：归档源码、运行时替换与评分公式。
- 不包含完整源网络、派生完整网络、100条原始mesh及Face pool二进制；其路径和哈希在EXCLUDED_FILES.json、manifest和数据清单中。当前head和源网络可重建派生模型。

源checkpoint：`/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_face_hardneg_round2_20260917/run/checkpoint-update22500.pt`
源SHA256：`428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79`
派生checkpoint：`/guohaoran/nexus_fast_track/diagnostics/shared_edge_head_reintegrate100_20260917/run/model-source22500-head2000-inference.pt`
派生SHA256：`a732eaf2fb1cccce7489d0f4cb42402ba0a6d170ef220ddcff2ba58c47198231`

派生checkpoint是仅推理的网络副本，没有拼接不匹配的旧Adam状态；正式续训需要单独约定。
