# 804-only μ 诊断材料与20-mesh趋势归档

这是一份已有训练结果与固定 checkpoint 的诊断导出，不是新训练。
主快照：only804_mu/checkpoint-update1000.pt；本分支新增1000步，四mesh阶段标记累计4400步。
SHA256：a622df076a2e4c6c5ea9dc2d93740b47fba90dcd559654291471efd377346fa5
补充计算没有创建optimizer，没有optimizer update。权重、RNG和原训练代码哈希均核验未改变。
固定前向与原已保存 step1000 的 μ、logvar、Edge logits、Face logits逐元素完全一致。

## 阅读顺序

1. A_only804_mu/manifest.json、verification.json、complete.json：起点、实际配置、冻结与完成核验。
2. A_only804_mu/comparison.json：与已有sampling+KL分支按相同 μ 验收口径比较。
3. A_only804_mu/updates.jsonl：完整连续1—1000更新，无抽样截断。
4. C_snapshot804_step1000/summary.json、logit_gradient_groups.csv：完整pool上的logit梯度及候选对应统计。
5. C_snapshot804_step1000/effective_code/effective_loss_and_scoring.py：可直接阅读/导入的实际生效公式与归约代码。
6. B_twenty_mesh/*.csv：20条逐mesh训练和硬重建趋势。

## A：804-only μ分支

包含所有已有μ检查点0/200/400/600/800/1000（覆盖所需0/400/800/1000）、完整日志、原运行入口、运行时forward/Graph补丁、完成及冻结核验。
训练目标严格为 `(EdgeSoft4_804(mu)+FaceSoft4_804(mu))/4`。
四条完整mesh仍进行一次packed forward；只有804点的重建项参与反传；β=0；logvar专属参数冻结。
日志中的另外三条parts、KL、σ是诊断值，不进入本分支目标。
未打包旧Control/BF16运行目录；comparison.json已包含所需旧对照计数。

## B：20-mesh统计的时间和口径

导出时间UTC：2026-09-14T01:43:46.235496+00:00。
逐步训练统计截至第2807次完整更新；硬验收至step2800；50噪声验收至step2000。
这是正在运行任务的截面，不是最终结果。

- per_mesh_training_updates.csv：每步每条的Edge/Face训练loss、KL、σ及贡献系数。loss发生在该次更新之前，因此另列state_step_before_update。
- per_mesh_hard_reconstruction.csv：μ、固定诊断噪声、已有监控噪声的实际TP/FP/FN、缺失GT候选、覆盖率与严格成功记录。
- per_mesh_noise_summary.csv：每个监控检查点各mesh的成功次数、均值/最小/最大错误与loss。
- manifest.json是实际运行manifest；recommended_config.json保留准备时的历史标记，不能拿其中not_launched字段判断当前未启动。

每个optimizer update包含20条完整mesh：固定顺序逐条forward/backward，各自完整目标先除20，累积20份梯度之后仅做一次global clip与一次Adam更新。
没有子图sampling。20条的实际执行布局是20个单mesh microbatch，不是同时packed20。
本20条阶段step=t时，每条都参与t次更新；父checkpoint之前各条历史不同，不能把4400统一加给全部20条。
硬指标只在原有检查点记录。strict_perfect_count_so_far_in_mode是“已记录验收forward中的累计次数”，不是声称每个训练step都进行了硬验收。
所有Face都由该次预测Edge图枚举；没有用训练pool替代。

## C：804点固定快照数据字典

所有ID均使用该mesh的0基局部顶点编号。无向pair满足i<j，ID=i*804+j；Face canonical ID=(i*804+j)*804+k，三元组排序后编码。

### edge_all_pairs.npz
完整322806对，pairs/ids/labels/logits/in_training逐行对齐。
`grad_loss_wrt_logit`是原完整目标对该实际logit的导数，已经包含外层除4。
`membership_TP_TN_FP_FN`形状[4,N]，顺序固定TP/TN/FP/FN；另存soft_group_mass与numerator。

### face_training_rows.npz
完整训练pool的4012行，保留原训练顺序：全部1604个GT正例，然后2408个mixed负例。
`union_row`映射到face_union；此文件的logit梯度来自完整pool的一次原目标反传，不是对子集重新计算的Soft4。
这里实际采用positive+mixed，**原NPZ中名为pool/heldout/original的其他数组不等于本轮全部参与训练的行**。

### face_union.npz
训练pool ∪ 预测Edge candidates ∪ GT faces，共5618条；实际Edge candidates为3100条。
`in_training_pool`和`in_actual_edge_candidates`是两个独立布尔值。
`actual_predicted_face=in_actual_edge_candidates & (logits>0)`。
GT未进入实际候选，即使诊断logit>0仍然计FN，不绕过Edge门控。
`has_direct_loss_term=False`时，`grad_loss_wrt_logit_sum=NaN`；这表示没有直接loss项，不能当成参加训练但导数0。
训练行全部唯一；仍保留training_occurrences及union映射，以明确集合与逐行归约关系。

本快照实际Face FP=800，其中139在训练pool，661不在训练pool；实际Face FN=851，其中557因缺边未入候选，294入候选后判负。
训练pool的GT logit FN=623与实际Face FN=851不同，原因包括上述Edge门控；不能混用这两个口径。

### representations_and_gradients.npz

- μ、z、logvar：[804,64]，本快照z=μ。
- decoder_hidden_before_final_ln、decoder_hidden_after_final_ln：[804,1024]。后者就是两个线性head实际接收的H。
- edge_head_raw、face_head_raw：[804,32]，中心化前的线性head输出。
- edge_embedding_scoring、face_embedding_scoring：[804,32]，实际进入评分的中心化表示。
- 所有对应grad_*为同一个完整目标的loss梯度；没有保存全网络参数梯度。
- vertices：[804,3]；face_centroids：[1604,3]；输入投影后的vertex/face features分别是[804,512]和[1604,512]。
- encoder_faces_in_input_order是Encoder真正输入的面顺序；gt_faces是监督候选顺序；两者集合一致但行序不同，用encoder_face_to_gt_candidate_row对齐。
- packed_vertex_row对应只含顶点的Decoder序列；encoder_packed_*_node_row对应含顶点和面节点的Encoder序列。
- incidence_index_local_vertex_then_face保存实际输入图关系。

input_examples.json给出输入张量的简短样例；input_posterior_output_head_weights.npz保存输入/后验/输出head与终端LN权重，可结合表示检查评分。

## 生效代码与loss语义

effective_code/effective_loss_and_scoring.py包含当前执行的Soft4、边/面评分、chunk和最终/4调用；独立导入复算已与运行时目标bitwise一致。
Soft4使用p=sigmoid(logit)，τ=1，membership不detach，分母为各组完整质量和+1e-8，四组固定平均。
`backend00`通过运行时替换移除detach。导出的精确替换源可能保留历史注释“Membership must not contribute”，但执行语句已经没有detach；不要按旧注释判断梯度定义。
当前Edge chunk=13000000，804全部pair在一个训练chunk；Face训练4012条在一个评分调用中；外层完整目标再除4。
scales、clamp、normalize、LayerNorm位置见effective_code/runtime_contract.json。
spacetime embedding不做输出RMS归一化；每mesh每channel中心化；评分scale和面积因子保留。评分用到的面积clamp等细节完整保留在导出函数中。
sampling_forward.py文件名沿用历史，但本分支a.training=False选择μ路径；不能按文件名判定sampling开启。
math00的wrapper/core与recompute精度上下文见source_closure中的backend00.py及effective_code。
source_closure只包括本次导入的依赖源码，不含旧实验日志；source_index.json记录原路径及哈希。

## 如何看logit梯度

signed_margin=(2y-1)*logit。
margin_gain_per_unit_eta=-(2y-1)*grad_loss_wrt_logit。
它>0表示把该logit视为独立变量时，负梯度方向有利于该margin；<0表示不利；=0保持数值零。
这不等于网络一次参数更新后每个logit必然沿同方向移动：网络的logits由共享表示生成，并非独立自由变量。
summary.json另列abs_gradient<=1e-12的数量，阈值仅用于阅读，不参与训练。

## 读取示例
```python
import numpy as np
from pathlib import Path
p=Path('C_snapshot804_step1000')
edges=np.load(p/'edge_all_pairs.npz')
faces=np.load(p/'face_union.npz')
representations=np.load(p/'representations_and_gradients.npz')
unsupervised_fp=(~faces['labels']) & faces['actual_predicted_face'] & (~faces['in_training_pool'])
print(int(unsupervised_fp.sum()))  # 661
```

本包不含1.34GB完整checkpoint；包含其SHA、来源、实际快照和生效代码，不需要加载大checkpoint即可检查候选/表示/logit梯度。也未打包20条候选NPZ。
export_scripts是本次导出脚本；SHA256SUMS.txt用于传输校验。
