# Graph归一化顺序与激活函数：CPU机制核验

2026-10-03。已执行CPU前向及自动微分，没有optimizer更新，没有使用GPU或连接服务器。未加载训练权重，未修改任何原实验源码或数据。

**结论：已经验证“这些操作改变了特征与梯度”及“投影后LN可抑制投影整体缩放带来的消息尺度变化”；尚未验证其中哪一项改善实际Edge/Face重建。** 这份报告不能作为新增训练消融结果。

## 范围与输入

- 直接只读导入归档 `own512_v2_recipe_pair_20260923/native_models.py`。
- 使用已保存 `teacher_cad50_00.npz` 的FP32坐标与原face编号：68顶点、136面，vertex-face图共204节点。
- 固定seed=0，新建两个3→512输入投影与单个真实512宽EncoderBlock。没有经过已经训练的完整Encoder或Decoder。
- 对照时固定同一输入、全部权重、Graph邻居bias、Attention及同一标量线性探针，只改变指定操作。
- PyTorch 2.8.0，CPU FP32；激活导数检查为CPU FP64。复用现有本地环境，没有安装或修改依赖。

脚本、日志和原始数值分别为 `verify.py`、`execution.log`、`results.json`。JSON包含源码与数据SHA256；运行结束复核它们未变。

## 1. 先确认公式确实对应有效代码

原A：`h + SiLU(Graph(LN(h)))`，后接原Attention与ReLU FFN。

V2：`h + GELU(LN(Graph(h)))`，后接原Attention与GELU FFN。

两种手写展开分别与各自原生EncoderBlock比较：输出最大绝对差=0，固定探针对输入的梯度最大绝对差=0。

另确认一个此前讲解未展开的差异：原A的neighbor_projection有bias，V2没有。后续隔离测试将bias保持相同，避免把它混入顺序或激活的作用。

## 2. 归一化位置确实改变消息尺度

这是一个人为控制的缩放测试：同时把Graph投影权重与bias乘0.1、1、10；其他参数相同，两边均使用GELU。记录每节点激活前特征标准差，再对204节点平均。

| Graph投影整体倍率 | 先LN后Graph | 先Graph后LN |
|---:|---:|---:|
| 0.1 | 0.07654 | 0.99547 |
| 1 | 0.76537 | 0.99995 |
| 10 | 7.65366 | 1.00000 |

在本次LN初始gamma=1、beta=0的条件下，Graph之后的LN将激活输入尺度保持在接近1；Graph之前的LN无法直接消除后续投影引入的整体缩放。0.1档的微小偏差来自LN的epsilon。

该测试没有展示实际训练中Graph权重是否如此缩放，也没有证明后置LN总是更稳定。训练后的LN仿射参数会改变尺度，归一化还会改变信息和梯度，不能只凭这一表选择结构。

## 3. 三项操作都改变梯度，但没有“越变越好”的排序

实际执行了2×2×2种组合：LN位置(pre/post)、Graph激活(SiLU/GELU)、Encoder FFN激活(ReLU/GELU)。全部使用相同权重和输入。

下面以post/GELU/GELU为参照，每次只撤回一项。相对变化定义为 `||x_other-x_reference|| / ||x_reference||`。

| 只改变一项 | block输出相对变化 | 对输入的VJP相对变化 | 对全部block参数的VJP相对变化 |
|---|---:|---:|---:|
| LN移到Graph之前 | 23.671% | 23.530% | 26.254% |
| Graph GELU改SiLU | 12.476% | 13.429% | 14.575% |
| FFN GELU改ReLU | 9.019% | 11.544% | 20.884% |

VJP使用固定随机线性探针 `sum(output * probe)`，用来比较各运算的局部导数。**它不是Edge/Face loss，以上百分比不是性能提升、不是梯度质量分数。** 所有输出和梯度有限，block权重和buffer在测试前后逐元素相同。

## 4. 对上一条解释的补充

ReLU在负输入区域导数为0；GELU在部分负输入上保留非零导数，实际数值与固定函数的中心有限差分一致（ReLU在0不可微，该点不作有限差分验证）。

例如x=-1时：ReLU导数=0，GELU导数约-0.08332。这个负导数说明“仍有梯度”不自动意味着沿任何下游目标更容易优化。非常负的输入仍可能产生很小梯度。原Graph的SiLU本来也是平滑激活。残差与其他通道仍然提供梯度路径，不能称ReLU使整个网络断梯度。

## 5. 已有实验支持的范围

已复核历史 `own512_v2_recipe_pair_20260923/repro_outputs/COMPARABILITY.md` 及恢复分支的 `delivery/CAD50_REPORT_ZH.md`：同CAD50、同配方、同19356有效更新，A/B实际Face F1为0.686776634/0.995264898。

那轮同时变化Fourier输入、Graph顺序/激活/bias、Encoder FFN激活、16个Decoder FFN和参数量。它支持整套V2有效，没有单独回答本次两个问题。在本次读取的相关对照报告中，没有找到已完成的仅LN顺序或仅Encoder激活训练对照；`C_graph_only`条目实际是deterministic index_add后端检查，并非归一化顺序消融。

## 6. 验证实际重建收益还需要什么

可在同一套V2基础上设四支：原V2；只将Graph LN移到聚合/投影前；只将Graph GELU改SiLU；只将Encoder FFN GELU改ReLU。保留相同Fourier、Decoder FFN、参数量和neighbor_bias=False。每支从匹配的随机参数独立起步，固定数据、顺序、负例、优化器、训练预算与真实重建验收；多随机种子可检验结果的稳健性。

这个设计检验的是各项在V2背景下的条件收益；若要量化三项之间的交互，还需完整因子对照。本次没有创建或启动这些训练，没有凭空指定服务器或预算，也没有改动现有VAE训练。

**当前结论：机制解释获得数值验证，实际重建收益的单项因果归属仍未确定。**
