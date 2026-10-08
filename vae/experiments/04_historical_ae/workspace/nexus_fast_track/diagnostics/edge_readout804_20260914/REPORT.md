# 804点固定Decoder hidden的线性读出诊断

结果：离线FP32读出成功，原网络临时替换同一head权重后也严格成功。所有检查均未执行optimizer update。

## 设置

- H：only804_mu step1000导出的Decoder末端LayerNorm后特征，804×1024。
- 目标E*：自由Edge表示拟合step2000的已中心化表示，804×32，按保存值使用。
- 解FP64最小二乘：min_W ||center(H W^T)-E*||_F²；CPU SVD Moore-Penrose最小范数解。
- 截断规则：max(m,n)×eps64×最大奇异值；未做ridge、温度、LR或秩阈值sweep。
- FP32验收：保留原bias，F.linear(H,W,b)后按mesh中心化，原16+16拆分、原评分scale与threshold=0。
- bias在精确算术中心化后消去；验收仍保留原bias，未把它当作新可调参数。
- 原完整checkpoint SHA256：a622df076a2e4c6c5ea9dc2d93740b47fba90dcd559654291471efd377346fa5。

| 路径 | FP | FN | Edge F1 | 最小GT margin | 最小non-GT margin |
|---|---:|---:|---:|---:|---:|
| 原网络 | 1645 | 332 | 67.722449% | -0.870247 | -6.810190 |
| 自由表示step2000 | 0 | 0 | 100.000000% | 6.388382 | 5.705134 |
| 离线FP32线性读出 | 0 | 0 | 100.000000% | 6.087714 | 5.360884 |
| 原网络packed前向＋新head | 0 | 0 | 100.000000% | 5.962943 | 5.126428 |

## 可实现性与数值条件必须分开解释

- 中心化H在FP64默认阈值下秩：803，中心化后的最大行秩为803。
- 在常用FP32相对阈值下的有效秩：475；这个阈值只用于报告，没有用于求解。
- 最大／最小保留奇异值：530.793978／1.37765746e-05。
- 保留子空间条件数：38528734.1。
- 原head权重Frobenius范数：3.7573278。
- 新head权重范数：689716.718，是原来的183566倍。
- 新权重最大绝对值：22298.252。
- FP64表示相对残差：1.73214598e-08；MSE=1.22794732e-15。
- 离线FP32表示相对残差：0.00697238799；MSE=0.0001989637。
- 实际packed网络表示相对残差：0.0102574435。

这份目标的精确拟合依赖小奇异值方向，并产生很大的最小范数权重。
因此“存在严格正确读出”不等于“读出权重尺度合理、容易通过现有训练学到、或对扰动稳健”。
也不能反推所有其他正确Edge表示都必须使用同样巨大的权重；本次只拟合了指定的E*。

## 完整网络核验

原四mesh packed布局不变，H、μ和Face embedding与原网络逐位一致；仅临时替换autoencoder.edge_embedding.weight。
原baseline H、Edge embedding及全部logits首先与归档逐位核对通过。
新head完整前向重复两次一致，804点FP=FN=0；结束时恢复原weight，全部参数哈希、恢复后的forward、RNG核验通过。
原checkpoint文件哈希未变化，没有保存带新head和旧Adam混用的训练checkpoint。

804行离线GEMM与4299行packed原head GEMM的数值结果不完全一样：
表示最大差异0.140625，logit最大差异3.09729004。
二者仍均严格成功。这也是必须做实际网络核验而不能只看离线MSE的原因。

## 结论范围

当前固定Decoder特征能经原形状线性Edge head读出804点的严格正确边图；成功不需要改变Encoder、Decoder或32维评分。
这使“当前特征完全不含可用信息”不再是充分解释；训练目标、参数更新尺度、联合优化与可读出性的数值条件仍需区分。
本轮未验证Face和其他mesh是否成功。Face embedding保持不变并不意味着实际Face图不变，因为Edge候选会变。
没有把单条离线解当作最终共享模型，也没有开始任何后续训练。

文件：offline/edge_head_candidate.pt是单独候选head；offline/readout.npz含H、目标、FP64/FP32权重与全部pair评分；
network_verification/actual_network_readout.npz保存实际网络结果；singular_values.csv及图保存完整奇异值。
