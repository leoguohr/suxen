# Nexus topology diffusion：50-CAD overfit 实验

已在本地 RTX4090 上完成三组小数据训练。第三组按用户指定使用 Nexus 默认 **Spacetime Interval，式5**，不是式9的 Minkowski loss 消融项，也不是一般的 Lp 距离。

## 数据

真实使用 `ricemonster/qwen2.5-3B-SFT` 推理生成 CAD 代码，再经 CadQuery 导出 STEP/OBJ/STL。
共尝试 57 条公开文本，7 条未通过预设的执行/几何/顶点预算检查，接受 50 个 mesh；OBJ 文件哈希不同者 50 个，全部 watertight：True。
顶点数范围 8–274，面数范围 12–548。
统一尺度归一化并舍入至6位后，canonical mesh哈希不同者 50 个；有 7 个零件包围盒长宽比超过100。极细长外观来自模型原始输出，未人工替换；本次未评估文本语义或尺寸准确性。
没有使用数据集配套的目标代码冒充推理输出，也没有人工补写生成几何。

![全部50个CAD](../cad50_gallery.png)

## 实验设置

同一批50个mesh、seed=20260913、batch=5；每组VAE训练 3000 步，flow训练 4000 步，AdamW学习率均为0.0003。
VAE约89.8万参数，flow约155.7万参数；latent维数64。VAE包含vertex–face图卷积、全局attention、KL bottleneck、分离edge/face embedding；flow包含3D RoPE与时间AdaLN，给定顶点位置条件。
三组采用相同训练预算，不按结果分别调参。架构、阈值、温度、负采样、标准化和与论文的差异见项目README。
最终协议采用逐步负三元组重采样和低噪声后验初始化（logvar=-6）；之前固定负采样的诊断试跑单独保存在 `results/pilot_fixed_negatives`，不混入此处曲线。

## 测量结果

| 指标 | Flow固定噪声MSE：初始→最终 | 降幅 | 最终VAE balanced BCE | VAE完整面F1 | 噪声生成完整面F1 |
|---|---:|---:|---:|---:|---:|
| Cosine similarity | 2.31695 → 0.14824 | 93.6% | 0.57039 | 0.1007 | 0.0813 |
| Euclidean distance | 2.29275 → 0.21543 | 90.6% | 1.21665 | 0.0920 | 0.0560 |
| Minkowski / Spacetime（式5） | 2.31272 → 0.14270 | 93.8% | 0.80702 | 0.4946 | 0.3112 |

![三条diffusion loss同图](diffusion_loss.png)

![VAE及diffusion两阶段](loss_curves.png)

曲线为原始记录的评估点连线，未拟合、未人为修改。Flow指标来自**训练集**的50个mesh，每个使用4个固定t和独立固定Gaussian噪声；在线随机训练loss也保存在CSV。VAE BCE使用论文的TP/TN/FP/FN分组，空组计零，因分组变化可能有波动。

完整面F1通过全部预测边枚举全部三角环，再用面指标判断；不是训练抽样候选集F1。生成F1使用50步Euler从Gaussian噪声积分，给定真实顶点位置，不输入真实拓扑或样本ID。

## 解释边界

这是单种子、缩小网络的**训练集overfit实验**，不是原论文32×A100、2B DiT及百万mesh训练的完全复刻，也不是泛化评估。训练完成不意味着每个方法都已达到近零loss或完美重建；上表F1直接反映剩余拓扑误差。
不同方法的VAE学习了不同latent，虽然逐通道标准化，flow MSE仍不能单独证明某一种拓扑度量优越。必须结合VAE重建和真实采样F1；不能把这个实验当作论文Table9的复现数值或统计显著性结论。

## 复现与文件

`config.json` 保存配置及数据manifest哈希；各方法目录保存 `autoencoder.csv`、`diffusion.csv`、`vae.pt`、`flow.pt`、`latent_normalization.pt`、`reconstruction_metrics.json`。全部50个重建的面索引保存为JSON，前10个非空重建另存OBJ。
数据位于 `data/cad50`，每次尝试均有prompt、模型原始输出、执行日志；成功样本还有STEP/OBJ/STL。权重revision和SHA256保存在 `models/text-to-cadquery-3b`。

来源：[Nexus全文](https://arxiv.org/html/2607.13563v1)、[Text-to-CadQuery官方实现](https://github.com/Text-to-CadQuery/Text-to-CadQuery)、[生成模型](https://huggingface.co/ricemonster/qwen2.5-3B-SFT)。
