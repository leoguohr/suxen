# Decoder末端接回真实网络：100条Edge＋实际Face无更新验收

使用原第二轮难负例epoch900/update22500的独立副本，安装Decoder末块＋最终LayerNorm＋Edge head诊断step500权重。仅替换指定10个state_dict张量；其余参数、buffer、metadata逐位不变。Face head权重没有改变。
源SHA256：`428aeddbc6ea03ae166ba22aa431fe40f4298ca83e0e5008302a3c5eed3ceb79`；输入末块checkpoint SHA256：`23522f92141dd10857edf38340ab0962cc0a2f6a653ec3227ecd50848d10686a`。没有加载任何Control模型、三条专用head或head-only step2000。
本次没有构造optimizer，没有backward或optimizer update。每次都从真实mesh输入执行Encoder→μ→全部Decoder；没有用缓存hidden代替网络推理。

先完成全部100条Edge gate：真实网络的raw/centered Edge表示及全部84,669,234个pair logits逐位复现缓存诊断；71/100、FP156784、FN1全部复现。μ、logvar和末块输入在安装前后逐位不变。7条指定大mesh额外重复forward逐位相同。
随后固定每份模型，逐条完整评估100条；Face从各自当次预测Edge图枚举，不替换成训练pool，不截断候选。替换前100条计数、loss、margin复现源epoch900归档。

| 指标 | 源模型 | 安装末块＋LN＋head后 |
|---|---:|---:|
| Edge严格成功 | 50 | 71 |
| 实际Face严格成功 | 51 | 50 |
| Edge＋实际Face联合成功 | 47 | 50 |
| Edge FP | 209400 | 156784 |
| Edge FN | 2 | 1 |
| 实际Face FP | 10322 | 59755 |
| 实际Face FN | 980 | 1651 |
| 缺边导致未进入候选的GT Face | 3 | 1 |

原联合成功47条，保留45，丢失2，新增5；安装后同一模型联合成功50/100。全部UID在comparison.json。

## 7条新成功大mesh的实际Face

| UID | 顶点数 | 新Edge FP/FN | 原Face FP/FN | 新实际Face FP/FN | 新联合严格成功 |
|---|---:|---|---|---|---|
| nexus_2k_000446 | 1519 | 0/0 | 56/5 | 0/30 | False |
| nexus_2k_001957 | 1556 | 0/0 | 65/0 | 7/32 | False |
| nexus_2k_000249 | 1585 | 0/0 | 27/0 | 3/13 | False |
| nexus_2k_000766 | 1617 | 0/0 | 52/1 | 0/6 | False |
| nexus_2k_001539 | 1700 | 0/0 | 26/0 | 0/1 | False |
| nexus_2k_001969 | 1722 | 0/0 | 20/3 | 3/39 | False |
| nexus_2k_000815 | 1735 | 0/0 | 21/0 | 0/6 | False |

## 判读边界

100条中有100条Face embedding发生变化。冻结Face head参数不等于冻结Face分数：共享Decoder末端变化会影响评分；Edge图变化还会改变Face候选。两种影响同时存在，本轮没有另外做因果拆分。
Edge诊断接回成功不等于Edge＋实际Face成功。联合结论仅以本次同一推理副本、各自当次预测Edge候选的完整计数为准。不拼接不同模型/不同forward的成功。
原源模型与step500诊断checkpoint均保留，派生模型是没有Adam状态的推理副本，本轮没有自动选择新主线或追加训练。

## 材料索引

- run/installation_verification.json：权重映射、实际改变的10个键、其他状态不变、独立副本SHA与重载检查。
- run/edge_gate.json、edge_gate_summary.json：100条真实网络与缓存结果逐位对齐，hidden/Face表示的变化量。
- run/before.jsonl、after.jsonl：两份模型各100条的实际Edge/Face、候选覆盖、FN分解、训练pool诊断loss、margin和特征哈希。
- comparison.json、per_mesh_comparison.csv、seven_large_meshes.csv：汇总、成功保留/丢失/新增UID及7条大mesh结果。
- runtime.py、sampling_forward.py、evaluate.py、run_evaluation.py、effective_scoring及runtime_dependencies：实际执行代码、运行时替换、枚举和评分规则。
- input_tail/checkpoint-step0500.pt：输入三模块权重及此前500步诊断的历史状态；本次只加载其中tail权重，未使用其Adam。
- run/features/*.npz（FullEvidence包）：全部100条μ/logvar、安装前后hidden、Edge/Face表示、全部末尾Edge logits及GT mesh数组。
- 大型源网络、派生完整网络及Face pool二进制不入包；路径与SHA在EXCLUDED_FILES.json和source_manifest.json。Review包另省略大型features，逐文件路径/哈希明确列出。
- 没有导出全部实际Face候选的逐条logit；保留完整枚举计数、代码与评分表示，不能把未导出称为不存在。

推理副本：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_reintegrate100_20260918/run/model-source22500-tail500-inference.pt`
推理副本SHA256：`d95454bc973895afccd845964724b3412bd6f489fd933e3033220bb7d06a127f`
