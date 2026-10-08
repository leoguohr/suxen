# 固定100条hidden：共享Face head恢复实验

已完成500次更新并按预算停止。每次全部100条分别计算完整固定训练pool的Face Soft4，除100后累积，统一clip=1，再执行一次fresh Adam更新。只有一个32×1024线性Face head及bias可训练，共32800参数。
起点为安装Decoder末端后的完整推理副本；没有使用安装前的旧hidden。LR=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0，无warmup。训练只含fully-differentiable Face Soft4，内部固定除4，无额外诊断系数，无Edge loss、KL、sampling或MSE。
源文件：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_reintegrate100_20260918/run/model-source22500-tail500-inference.pt`。SHA256：`d95454bc973895afccd845964724b3412bd6f489fd933e3033220bb7d06a127f`。
共缓存435,507,200字节FP32 hidden；原两轮扩充训练pool共750,215候选，固定预测Edge图产生554,923个实际Face候选。两者没有混用或自动合并。所有GT中1个Face因缺边不可进入实际候选，验收始终计FN。

| 更新 | Edge严格成功 | Face严格成功 | 联合严格成功 | 实际Face FP | 实际Face FN | Face Soft4均值 | 原50条保留/丢失/新增 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 71 | 50 | 50 | 59755 | 1651 | 0.419265649 | 50/0/0 |
| 50 | 71 | 67 | 67 | 43889 | 2321 | 0.155297932 | 50/0/17 |
| 100 | 71 | 69 | 69 | 55486 | 1454 | 0.139254568 | 50/0/19 |
| 200 | 71 | 71 | 71 | 65019 | 968 | 0.121197358 | 50/0/21 |
| 300 | 71 | 71 | 71 | 63506 | 860 | 0.109078603 | 50/0/21 |
| 400 | 71 | 71 | 71 | 55969 | 925 | 0.100569988 | 50/0/21 |
| 500 | 71 | 71 | 71 | 49411 | 984 | 0.094521536 | 50/0/21 |

末尾联合严格成功71/100；原50条保留50、丢失0、新增21。step300、400、500三个检查点共同成功71条。这是检查点保持证据，没有把未评估步骤宣称为全部成功。

Edge在7次全量检查均为71/100、FP156784、FN1。Face FP由59755变为49411，FN由1651变为984。较低训练pool loss不保证池外候选改善，实际FP/FN仍是独立验收。固定Edge图下，联合成功数上限71；其余Edge不完整样本不可能仅靠Face head成为完整mesh。

## 7条新增Edge成功大mesh

| UID | 起点Face FP/FN | 末尾Face FP/FN | 末尾联合成功 |
|---|---:|---:|---|
| nexus_2k_000446 | 0/30 | 0/0 | True |
| nexus_2k_001957 | 7/32 | 0/0 | True |
| nexus_2k_000249 | 3/13 | 0/0 | True |
| nexus_2k_000766 | 0/6 | 0/0 | True |
| nexus_2k_001539 | 0/1 | 0/0 | True |
| nexus_2k_001969 | 3/39 | 0/0 | True |
| nexus_2k_000815 | 0/6 | 0/0 | True |

## 验证和边界

起点100条真实网络前向与缓存路径的Face embedding、完整pool loss和head梯度逐位相同；两次完整100条backward梯度逐位相同。全部冻结权重、buffer与metadata哈希不变；缓存hidden、训练pool文件和实际候选集合哈希不变。
末尾重载Face head checkpoint，从真实mesh输入重跑100条Encoder→μ→Decoder，hidden、Edge embedding、Face embedding和完整实际重建逐位/逐项复现缓存验收。没有用GT图替代预测Edge图，没有安装独立mesh专属head。
保存的末尾逐候选logits与label已经独立重算全部100条Face及训练pool计数；保存的预测Edge ID与GT ID也独立重算，和记录完全一致。完整源模型文件SHA未变，派生模型另存。
只证明本轮有限预算下同一个Face head的结果，不证明100条全部通过，不将不同checkpoint的成功拼接，不自动追加预算。Face head之外参数没有更新；bias的数值变化仍记录，逐mesh中心化使bias对精确实数评分无影响。

## 材料索引

- run/updates.jsonl：完整500条更新，逐mesh Face loss、梯度、clip和实际参数更新。
- run/evaluation-step*.json、evaluations.jsonl：7次×100条实际Edge/Face、候选覆盖、margin和固定pool指标。
- trend.csv、per_mesh_evaluations.csv、retention.json、seven_large_meshes.csv：趋势与UID级保留/丢失/新增。
- run/checkpoint-step*.pt：7份共享Face head和对应新Adam状态；只训练Face head，不是完整网络的500轮训练。
- run/verification.json、final_real_network.json、audit.json、saved_logits_recount.json：冻结核验、真实网络重接与独立计数复核。
- cache/manifest.json：缓存和原两轮pool哈希。FullEvidence内cache/*.npz含安装后hidden、Edge表示、预测边图、实际Face/训练候选ID和label、原始mesh数组。
- run/final_outputs/*.npz（两种包均含）：末尾Face embedding、完整训练pool/实际候选/全部GT Face logits，与FullEvidence缓存逐行对齐。
- runtime.py、face_core.py、run.py、effective_face_scoring.py.txt、runtime_dependencies：实际执行代码与运行时替换。
- EXCLUDED_FILES.json：大型完整网络源/派生checkpoint及Review省略缓存的服务器路径、大小和SHA。Review用于快速审阅；FullEvidence包含全部缓存，原始大型网络权重另存服务器。

派生推理模型：`/guohaoran/nexus_fast_track/diagnostics/face_head_recovery_fixed100_20260918/run/model-tail500-face500-inference.pt`。SHA256：`8de6074700fafa02b3adcfafd352a4555ce0cbc88421667261f29495faa69640`。
