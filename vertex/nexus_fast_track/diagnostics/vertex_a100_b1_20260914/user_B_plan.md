**这次已经把问题缩小了：数据和八叉树基础链路基本通过检查，当前卡在“DiT 能记住固定答案，却没有学好对随机噪声的正确响应”，同时伴随明显的残差特征放大。**

**下一步不要扩数据，也不要进入 C/D。建议只做一个实验阶段：`1 个 mesh + depth 9 + 固定 t=0.5 + 随机噪声`，并且从头初始化模型，不再默认接着 A 的权重训练。**

我重新计算了保存数组中的指标、检查了源码，并在本地重跑了相关测试：**28 项测试通过，20 个样本的空间对齐核验通过**。包内没有模型权重，因此下面关于完整模型噪声响应和中间特征的数值，来自你提供的前向诊断，而不是我独立重跑了大模型。[本地测试记录](sandbox:/mnt/data/vertex_ab_local_tests.txt) · [空间核验结果](sandbox:/mnt/data/vertex_ab_alignment_recomputed.json)

## 一、这轮究竟通过了什么、没有通过什么？

**A 阶段确实通过了。** 从 `probe-000200.npz` 重新计算，速度 MSE 约为 **`1.26e-5`**，64 个占据判断全部正确，最终 8 个整数顶点坐标与 GT 集合完全一致。这个结果证明了：至少对这个固定输入，模型、反向传播和目标解码能够共同完成任务。[A 原始数组](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/run/A/probe-000200.npz)

**B 阶段不是“接近成功，只是验收太严格”。** 第 2000 步的结果如下：

| 时间步     | 新噪声上的平均速度 MSE | 精确恢复坐标集合 |
| ------- | ------------: | -------: |
| `t=0.1` |         0.881 |      0/4 |
| `t=0.3` |         0.875 |      0/4 |
| `t=0.5` |         0.882 |      0/4 |
| `t=0.7` |         0.884 |      1/4 |
| `t=0.9` |         0.955 |      4/4 |

这里实际是 **4 个噪声种子 × 5 个时间步**，不是 20 份彼此独立的噪声。所谓“5/20 恢复成功”，其中 4 次集中在接近 GT 的 `t=0.9`；但这些例子的速度误差仍然很高，不能据此认定去噪已经学会。[B 最终评估](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/run/B/evaluation-002000.json)

更直接的是：**给了正确的 GT parents，从纯噪声采样这一层，4 个种子分别生成 26、20、17、23 个顶点，目标只有 8 个，而且坐标集合全部不对。** 所以目前的问题已经发生在单层内部，还不需要用“多层误差累积”来解释。[重算结果](sandbox:/mnt/data/vertex_ab_recomputed.json)

## 二、最重要的判断：A 学会的很可能是固定答案，而不是去噪规则

对你当前的路径：

$$
x_t=(1-t)\epsilon+t\,y
$$

在**固定 mesh、固定父格、固定目标 \(y\)** 的这个特殊测试中，正确速度满足：

$$
v^*(x_t,t)=y-\epsilon=\frac{y-x_t}{1-t}
$$

因此，当 `t=0.5` 时：

$$
\boxed{v^*(x_t,0.5)=2y-2x_t}
$$

也就是说，输入噪声状态变化 \(\Delta x\)，正确输出应该变化：

$$
\boxed{\Delta v=-2\Delta x}
$$

这不是需要模型发现的复杂几何规律，而是这个固定目标测试下的一条线性关系。

但你保存的 FP32 前向诊断显示：

| 检查点        | 输出对输入扰动的幅度增益 | 此测试的正确值 |
| ---------- | -----------: | ------: |
| A，第 200 步  |  **0.00788** |       2 |
| B，第 2000 步 |  **0.76127** |       2 |

A 换新噪声后，`t=0.5` 的两例 MSE 又升到了约 **1.81、2.06**。结合极低的扰动响应，我更倾向于判断：**A 主要利用固定位置等信息，记住了那一份固定噪声对应的速度答案；B 虽然开始响应噪声，但还远没有学对。** 这不等于证明输入被 `detach()`，也不能只凭增益大小确认响应方向正确。[前向诊断](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/stage_B_forward_diagnostics.json)

这里要调整上一轮的训练建议：

> **A 是训练正确性的单元测试，不必成为 B 的预训练阶段。**
> 现在应该增加“从头初始化直接做随机噪声学习”的对照，而不是把 A 权重作为唯一入口。

作为额外核验，我用实际的 GT 占据标签，加上固定 parent ID，拟合了一个很小的线性回归对照；它在 **128 份新噪声上全部恢复了正确占据**。这只是固定父格任务的可解性检查，**不是 Vertex DiT 或条件生成的成功成绩**。[对照结果与复核脚本](sandbox:/mnt/data/vertex_ab_reaudit.py)

## 三、最值得优先排查的异常：残差放大和通道失衡

固定诊断输入下，你的中间特征变化很明显：

| 诊断项                          |  A 结束 |        B 结束 |
| ---------------------------- | ----: | ----------: |
| 第一个 block 输出 RMS             |  2.22 |  **309.55** |
| 最后一个 block 输出 RMS            | 75.89 | **3617.50** |
| 第一个 block 的 `scale_sa` 最大绝对值 |  1.89 |    **50.0** |

与此同时，B 结束时，噪声输入的 `data_embedding` RMS 只有约 **0.065**，位置 embedding 约 **0.202**，深度 embedding 约 **0.983**。这组相对尺度，加上较弱的噪声响应，使得“有效噪声信息没有被稳定保留和利用”成为当前最值得验证的解释。[A 末次评估](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/run/A/evaluation-000200.json) · [B 末次评估](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/run/B/evaluation-002000.json)

**还有一个比“RMS 很大”更具体的异常。**

B 的 FP32 诊断中，最终 `output_norm` 后最大通道绝对值约为 **37.16**。你的这一层是宽度 1536、无仿射参数的 LayerNorm；按每个 token 归一化后均方值约为 1 推算：

$$
\frac{37.16^2}{1536}\approx 0.899
$$

**这意味着至少一个 token，约 90% 的平方能量集中在单个通道上。** 注意，这是“至少一个 token”，不是已经证明所有 token 都如此。它说明最终 LayerNorm 虽然把整体尺度归一化了，但并没有消除严重的通道失衡。[前向诊断](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/stage_B_forward_diagnostics.json) · [归一化实现](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/code/mini_nexus/vertex.py)

**这仍然是强线索，而不是已经证明的唯一根因。** 不能仅凭它就删除 LayerNorm、RoPE 或 depth embedding；应该用初始化和训练设置的对照，检查这些异常是否随学习改善而消失。

## 四、源码里优先动哪里？

### 1. 先保留 flow 路径和八叉树编码

当前 `flow.py` 中的训练路径、速度目标、Euler 时间方向是相互一致的；空间和标签核验也已通过。我不会优先改速度符号、占据阈值或子格 bit 顺序。[Flow 实现](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/code/mini_nexus/flow.py) · [空间核验结果](sandbox:/mnt/data/vertex_ab_alignment_recomputed.json)

### 2. 把初始化作为第一组模型侧对照

你的 `_VertexBlock` 已经对 self-attention 和 FFN 使用了零初始化的调制分支，**但 cross-attention 是直接加到残差上的，其输出投影没有零初始化**。因此，初始化时整个 block 并不是恒等映射；源码注释也明确说明了这一点。[`vertex.py` 第 272–326 行](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/code/mini_nexus/vertex.py)

“无 gate 的 cross-attention”本身不能判定为错误。不过，作为对照，PixArt 官方实现也采用直接相加的 cross-attention，同时将其输出投影初始化为零。([GitHub][1])

我建议增加下面这个**稳定化初始化候选**，而不是把它当成已经验证的修复：

```python
# 在 VertexDiT.__init__ 的通用初始化之后执行。
# 保留原有 self/FFN modulation 和最终 output 的零初始化。

nn.init.normal_(self.depth_embedding.weight, std=0.02)

for layer in (self.time_embedding[0], self.time_embedding[2]):
    nn.init.normal_(layer.weight, std=0.02)
    nn.init.zeros_(layer.bias)

for block in self.blocks:
    nn.init.zeros_(block.cross_attention.output.weight)
    nn.init.zeros_(block.cross_attention.output.bias)
```

其中，time MLP 的 `std=0.02` 可参考官方 DiT 初始化；**将 depth embedding 缩到这个尺度，是针对你当前输入尺度失衡提出的工程对照，不是声称原论文要求如此。** ([GitHub][2])

这个修改必须在**新模型初始化**时测试，不能随后加载旧 checkpoint，把新初始化覆盖掉。cross 输出置零后，VecSet 梯度可能需要多几个更新才开始出现；原来要求第二次反向传播立即有 VecSet 梯度的测试，也需要相应调整，不能把这种预期的延迟误判为断梯度。

### 3. 暂时不要把 `1000*t` 或 BF16 定为唯一根因

你包里的固定时间对照，在第 200 步仍未通过；但它只包含到第 225 步左右的快照，配对的随机时间组还没有完成。因此，**还不能得出“时间随机化是根因”或“时间编码一定写错了”的结论**。[固定时间对照](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/noise_time_controls/fixed_time/evaluation-000200.json) · [快照状态](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/noise_time_controls/status.json)

同样，FP32 前向没有解决问题，只说明**切换推理精度不能直接救回当前权重**，不等于已经排除了 BF16 训练过程的影响。

## 五、下一轮具体怎么跑？

### 第一步：增加独立的 B1，不再强制先训练 A

保持 **`nexus_2k_000105`、depth 9、GT parents、实际固定点云与法向**，只随机噪声，时间固定为 `0.5`。

建议先用下面这组设置。它们是排错起点，不是已验证的最优超参数：

| 项目    | 下一轮建议                               |
| ----- | ----------------------------------- |
| 模型起点  | **从头初始化，不加载 A/B 权重**                |
| 时间与噪声 | `t=0.5`；每次独立采样噪声                    |
| 每次更新  | 平均 **8 份噪声实例**，仍然只有 1 个 mesh        |
| 优化器   | 新建 AdamW，`lr=1e-5`，`weight_decay=0` |
| 稳定设置  | warmup 100，梯度裁剪 1                   |
| 首轮预算  | 500 步，每 50 步评估；持续改善再延长到 1000        |

显存不够时，可以用 8 次独立 forward/backward 做梯度累积，loss 除以 8，最后才裁剪和更新。

**同一协议做两组：R0 使用原始初始化，R1 使用上面的稳定化初始化候选。** 两组都从头开始，使用相同训练噪声序列、学习率和评估噪声。R1 包含几个初始化变化，因此它通过只能说明“组合有效”，不能立即归因于某一个改动。

现有 `train_vertex_staged.py` 会从 A 接着使用同一个模型和优化器进入 B，所以需要新增独立 B1 入口，**不是单纯把 `--stage-steps` 调大**。[训练脚本](sandbox:/mnt/data/vertex_ab_review/vertex_A_B_review_20260914_221614/code/scripts/train_vertex_staged.py)

### 第二步：用“新噪声表现＋响应方向”验收 B1

固定一组验证噪声观察曲线；候选达标后，再用 **64 份没有用于调参的新噪声**检查。建议继续要求每例速度 MSE 不超过 `0.01`，且整数坐标集合完全正确。

另外，在 FP32 前向中检查：

$$
\text{signed gain}
=
\frac{\langle \Delta v,\Delta x\rangle}{\|\Delta x\|^2}
$$

在这个固定目标、`t=0.5` 的测试里，它应接近 **−2**，而不只是“输出变化的幅度变大”。同时记录：

$$
\frac{\|\Delta v+2\Delta x\|}{2\|\Delta x\|}
$$

看响应误差是否真正下降。**这个 −2 的关系只适用于当前固定目标测试，不是多物体生成的一般验收标准。**

训练过程中继续记录逐层 RMS，但增加**每个 token 的 top-1/top-8 通道平方能量占比**。这比只记录一个最大 hidden RMS 更容易判断，是否又出现了单通道主导。

**只训练 `t=0.5` 的 B1，不要求已经能从 `t=0` 积分到 `t=1`。** 那是下一阶段的任务，不能混在一起验收。

### 第三步：根据结果决定后续，而不是直接加训练量

**R0 通过**：说明原结构能学会这个任务，应优先排查旧的 A warm-start、`1e-4` 学习率和单噪声更新组合。

**只有 R1 通过**：保留这个结果，再逐项拆分初始化改动，确定真正必要的部分。

**两组都不通过**：先把随机噪声映射在更小的 DiT 上做对照，继续定位网络内部问题，不扩数据、不盲跑更多步数。

B1 通过后，再恢复随机时间做 **B2**，并要求 GT parents 下的纯噪声单层采样通过；然后才是 **C：单 mesh 全树生成 → D：2–4 个形状的条件切换 → 10/20 个 mesh**。

---

**我的优先级判断是：先解决“随机噪声映射能否稳定学会”，再解决“不同物体条件能否被利用”。当前单 mesh 测试即使不依赖 VecSet，也应该能学会前者，所以此时不宜继续把 VecSet 当作唯一主攻方向。**

完整操作细节整理在 [下一轮执行方案](sandbox:/mnt/data/vertex_ab_next_steps.md)，数值在 [复核结果 JSON](sandbox:/mnt/data/vertex_ab_recomputed.json)，可用 [复核脚本](sandbox:/mnt/data/vertex_ab_reaudit.py) 重算。

**下一轮最重要的成功信号，不是 A 的 loss 再低一点，而是换了新噪声以后，模型仍能给出正确、方向一致的去噪响应。**

[1]: https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha/master/diffusion/model/nets/PixArt.py "https://raw.githubusercontent.com/PixArt-alpha/PixArt-alpha/master/diffusion/model/nets/PixArt.py"
[2]: https://raw.githubusercontent.com/facebookresearch/DiT/main/models.py "https://raw.githubusercontent.com/facebookresearch/DiT/main/models.py"
