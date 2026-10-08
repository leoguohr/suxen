# 已归档：Medium Topology AE overfit20 代码阅读指南

> 本文对应历史 Medium 实验，参数和入口已经过时，不应用于当前
> paper-aligned overfit20。当前阅读顺序见项目根目录
> `TOPOLOGY_AE_CODE_PATH.md`。

这份指南只解释服务器当前正在运行的 `topology_ae_overfit20_medium_fp32_centered_v2`。服务器代码与本地代码在开始注释前逐文件 SHA-256 一致；本次只增加本地教学注释，不改变任何计算。正在运行的服务器任务没有被覆盖。

## 先看完整调用链

```text
scripts/run_topology_ae_overfit20_medium.sh
  ├─ 激活 scripts/server_env.sh
  ├─ 校验 20 条 manifest 的 SHA-256
  └─ 启动 scripts/train_topology_ae_overfit4.py
       ├─ mini_nexus/data_2k.py
       │    └─ 读取并审计 Stage-2/3/4 数据，组成 Nexus2KBatch
       ├─ mini_nexus/negative_candidates.py
       │    └─ 为 face 读取 mixed_medium 固定负 triplet
       └─ mini_nexus/training_2k.py
            ├─ 调用 mini_nexus/topology.py 的 TopologyAutoencoder
            ├─ 对全部 V(V-1)/2 顶点对计算 edge loss
            ├─ 对全部正 face + 固定负 face 计算 face loss
            └─ 加上 1e-4 × KL loss
```

推荐按上述顺序阅读。不要一上来从 `topology.py` 第一行死磕到最后；先知道每个文件为何存在，再进入模型内部。

## 这次实验真正使用了什么数据

每个对象都从 manifest 得到四个文件：condition point、Stage-3 training mesh、octree 和 topology。loader 会把四者全部读取并交叉校验，但当前 Topology AE 前向传播实际使用的是：

```text
vertices       [V,3]     Stage-3 归一化顶点坐标
faces          [F,3]     有向三角面索引，也是 encoder 的图结构
edge_index     [2,E]     从 faces 精确推导的 GT 无向边
face_set       [F,3]     每个 face 排序后的无向 triplet
negative_faces [N,3]     mixed_medium sidecar 中固定的负 triplet
```

`condition[8192,6]` 和九层 octree 在 batch 中存在，但本轮 Topology AE 没有消费它们。这不是完整的 point-cloud-to-mesh 生成；它是在给定 GT mesh 的前提下测试 Topology AE 能否记住 topology。

## 一个训练 step 究竟发生什么

启动脚本设置 `micro_batch_size=20`，所以一个 step 看完固定的全部20个对象，只调用一次 `optimizer.step()`：

1. `optimizer.zero_grad()` 清除上一步梯度。
2. `Nexus2KTopologyAESystem.forward()` 逐对象去掉 padding。
3. `TopologyAutoencoder.encode()` 用 vertex node、face node 和 incidence graph 得到每顶点 `mu/log_variance[V,64]`。
4. VAE 重参数化：`z = mu + exp(0.5*logvar)*epsilon`。
5. decoder 对 `concat(z, vertex_xyz)` 做12层 self-attention，输出两套 `[V,32]` embedding。
6. Edge 使用全部无序顶点对；Face 使用全部正例加固定负例。
7. 20个对象的 loss 取算术平均并反向传播。
8. 记录的是裁剪前 gradient norm；随后裁剪到1，再由 AdamW 更新参数。
9. 每500 step 保存一次 checkpoint，每步向 `train.jsonl` 写一行。

当前总目标是

\[
L=L_{edge}+L_{face}+10^{-4}L_{KL}.
\]

### 三种容易混淆的“batch”

本项目有两个训练入口，不能只看到变量名 `sample_index` 就把它们当成同一种
batch。先记住：`sample_index` 只是一个整数下标，例如7；真正决定 collate
对象数的是传入列表的长度。

```text
正式分阶段训练 train_nexus2k_single.py

stream.next() -> sample_index=7               一个 int
train_dataset[7] -> sample                     一个对象
collate_nexus2k_samples([sample]) -> B=1        单次 forward 看1个对象
重复 gradient_accumulation=4 次 backward
optimizer.step()                               有效 batch=4
```

```text
当前 overfit20 训练 train_topology_ae_overfit4.py

读取固定 samples[0:20]
collate_nexus2k_samples(samples) -> B=20        统计上的 batch=20
Nexus2KTopologyAESystem.forward(batch)
  for sample_index in 0..19:                    内部逐对象循环
    autoencoder(vertices_i, faces_i)            一次仍只算1个mesh
20个对象 loss 求平均
backward + optimizer.step()                     有效 batch=20
```

所以当前 overfit20 的结论是：一个参数更新确实使用20个对象，但并没有把20个
变长mesh同时送进Transformer。`pair_chunk_size=262144`又是第三个概念：它表示
处理一个mesh时一次计算多少个候选顶点对，与mesh batch size无关。

这里不是普通“把全部 BCE 相加再平均”。Edge 和 Face 都先按照当前预测被分成 TP、TN、FP、FN：

\[
L_{balanced}=\frac{1}{|G|}\sum_{g\in G}
\frac{1}{|g|}\sum_{i\in g}\operatorname{BCEWithLogits}(s_i,y_i),
\]

其中 `G` 是当前非空组。这样数量巨大的 TN 不会完全淹没真边和错误预测。某一组为空时怎样处理并未由论文公开；当前独立实现只平均非空组。

## 模型内部的 shape

Medium 配置是：encoder width 384、12个“单独层”即6组 GraphSAGE+Transformer，decoder width 768、12个 attention-only block、latent 64、Spacetime embedding 32。

```text
vertices [V,3] ──Linear──> vertex_features [V,384]
faces [F,3] ──取三顶点中心──Linear──> face_features [F,384]
拼接                                nodes [V+F,384]
6 × (GraphSAGE + Transformer)       nodes [V+F,384]
只取前 V 个 vertex nodes
  ├─ mu                             [V,64]
  └─ log_variance                   [V,64]
重参数化 z                          [V,64]
concat(z, vertices)                 [V,67]
Linear + 12 × self-attention        [V,768]
  ├─ edge_embedding                 [V,32]
  └─ face_embedding                 [V,32]
```

encoder 使用 GT faces 建立的 vertex↔face incidence graph。因此它是在压缩已知 topology，不是在无条件猜 topology。decoder 不接收 faces，只接收每顶点 latent 与坐标，必须从压缩表示中恢复边和面关系。

## Spacetime embedding 如何变成边和面

把32维 embedding 等分成16维 space 与16维 time。两个顶点的一阶 interval 是

\[
s_{uv}=\|x_u-x_v\|^2-\|t_u-t_v\|^2.
\]

`s_uv > 0` 预测为 edge，`s_uv < 0` 预测为 non-edge。训练中这个连续值直接作为 BCE logit，不先做 sigmoid，因为 `binary_cross_entropy_with_logits` 内部会以数值稳定方式完成 sigmoid+BCE。

Face 使用三个顶点在 space/time 中张成的平行四边形面积平方：

\[
s_{uvw}=A_{space}^2-A_{time}^2.
\]

同样以零为分类阈值。推理时必须先由一阶 interval 建 edge graph，再枚举3-cycle，最后只对3-cycle做二阶 interval 判断。

## 为什么 decoder 输出要减均值

interval 只依赖顶点之间的差，所以给所有顶点 embedding 加同一个向量不会改变任何精确数学结果。这是一个 loss 无法约束的“公共平移自由度”。旧 BF16 运行中公共分量长到数百、有效顶点差异不足1，最终舍入破坏了 interval 符号。

当前代码在每个 mesh 内执行：

```python
embedding = embedding - embedding.mean(dim=0, keepdim=True)
```

这叫固定 gauge：它移除无意义的共同偏移。该操作与 FP32 是本独立复现的数值稳定措施，不是论文公开的实现细节。

## 逐文件学习入口

1. `scripts/run_topology_ae_overfit20_medium.sh`：实验参数、隔离环境、manifest hash、自动续训。
2. `scripts/train_topology_ae_overfit4.py`：20条数据如何进入GPU、一个step、反向传播、梯度裁剪、日志与checkpoint。
3. `mini_nexus/data_2k.py`：四类文件如何保持同UID以及如何组成变长batch。
4. `mini_nexus/negative_candidates.py`：mixed_medium face负样本如何确定性选取。
5. `mini_nexus/training_2k.py`：全顶点对 edge loss、固定 face loss、逐对象平均。
6. `mini_nexus/topology.py`：GraphSAGE/Transformer encoder、VAE、attention decoder、Spacetime interval。

阅读每个函数时都回答四个问题：输入 shape 是什么、输出 shape 是什么、它是否参与梯度、它属于论文明确内容还是独立工程选择。能自己回答完这四个问题，才算真正看懂，而不是只看懂 Python 语法。

## 论文明确内容与独立实现选择

论文明确：vertex/face graph nodes、GraphSAGE 与 Transformer encoder、每顶点64维 latent、pure-attention decoder、一阶/二阶 Spacetime Interval、edge 全顶点对监督、face 全正例加采样负 triplet、TP/TN/FP/FN 平衡 BCE。

独立实现选择：Medium 容量、head 数、GraphSAGE 的具体写法、attention block 不带 FFN、逐对象变长循环、chunk size、mixed_medium 的负 face 比例与分布、空 TP/TN/FP/FN 组处理、FP32、每 mesh embedding 去均值、gradient clip=1。

因此这份代码可以作为论文约束下的可运行独立复现学习，但不能称为作者官方实现。
