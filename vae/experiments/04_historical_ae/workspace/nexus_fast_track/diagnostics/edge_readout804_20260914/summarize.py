"""Build the report and singular spectrum from saved results only."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
fit=json.loads((root/'offline/result.json').read_text())
actual=json.loads((root/'network_verification/result.json').read_text())
s=np.load(root/'offline/singular_values.npz')['centered_h']
with (root/'singular_values.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['index_1based','centered_H_singular_value'])
    w.writerows(enumerate(s,1))
fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
ax.semilogy(np.arange(1,len(s)+1),s,color='#286e92',label='Singular values of centered H')
ax.axhline(s[0]*fit['spectrum']['fp32_relative_cutoff'],color='#c8643d',ls='--',label='Conventional FP32 rank threshold (diagnostic only)')
ax.axhline(fit['spectrum']['fp64_absolute_cutoff'],color='#577747',ls=':',label='FP64 solve rank threshold')
ax.set(xlabel='Singular-value index',ylabel='Singular value',title='Fixed Decoder hidden: 804 vertices x 1024 features')
ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(root/'singular_spectrum.png',dpi=180)
fig.savefig(root/'singular_spectrum.pdf')

table=['| 路径 | FP | FN | Edge F1 | 最小GT margin | 最小non-GT margin |',
       '|---|---:|---:|---:|---:|---:|']
for label,item in [('原网络',actual['baseline']),('自由表示step2000',fit['target']),
                   ('离线FP32线性读出',fit['fitted_fp32']),('原网络packed前向＋新head',actual['fitted'])]:
    table.append(f"| {label} | {item['fp']} | {item['fn']} | {100*item['f1']:.6f}% | {item['min_margin_gt']:.6f} | {item['min_margin_non_gt']:.6f} |")
report=f'''# 804点固定Decoder hidden的线性读出诊断

结果：离线FP32读出成功，原网络临时替换同一head权重后也严格成功。所有检查均未执行optimizer update。

## 设置

- H：only804_mu step1000导出的Decoder末端LayerNorm后特征，804×1024。
- 目标E*：自由Edge表示拟合step2000的已中心化表示，804×32，按保存值使用。
- 解FP64最小二乘：min_W ||center(H W^T)-E*||_F²；CPU SVD Moore-Penrose最小范数解。
- 截断规则：max(m,n)×eps64×最大奇异值；未做ridge、温度、LR或秩阈值sweep。
- FP32验收：保留原bias，F.linear(H,W,b)后按mesh中心化，原16+16拆分、原评分scale与threshold=0。
- bias在精确算术中心化后消去；验收仍保留原bias，未把它当作新可调参数。
- 原完整checkpoint SHA256：{actual['original_checkpoint_sha256']}。

{chr(10).join(table)}

## 可实现性与数值条件必须分开解释

- 中心化H在FP64默认阈值下秩：{fit['spectrum']['centered_rank_fp64']}，中心化后的最大行秩为803。
- 在常用FP32相对阈值下的有效秩：{fit['spectrum']['effective_rank_fp32_tolerance']}；这个阈值只用于报告，没有用于求解。
- 最大／最小保留奇异值：{fit['spectrum']['largest_singular']:.9g}／{fit['spectrum']['smallest_retained_singular']:.9g}。
- 保留子空间条件数：{fit['spectrum']['condition_retained']:.9g}。
- 原head权重Frobenius范数：{fit['weights']['original_frobenius']:.9g}。
- 新head权重范数：{fit['weights']['fit_fp32_frobenius']:.9g}，是原来的{fit['weights']['norm_ratio']:.6g}倍。
- 新权重最大绝对值：{fit['weights']['fit_max_abs']:.9g}。
- FP64表示相对残差：{fit['fp64_residual']['relative_frobenius']:.9g}；MSE={fit['fp64_residual']['mse']:.9g}。
- 离线FP32表示相对残差：{fit['fp32_residual']['relative_frobenius']:.9g}；MSE={fit['fp32_residual']['mse']:.9g}。
- 实际packed网络表示相对残差：{actual['target_relative_frobenius']:.9g}。

这份目标的精确拟合依赖小奇异值方向，并产生很大的最小范数权重。
因此“存在严格正确读出”不等于“读出权重尺度合理、容易通过现有训练学到、或对扰动稳健”。
也不能反推所有其他正确Edge表示都必须使用同样巨大的权重；本次只拟合了指定的E*。

## 完整网络核验

原四mesh packed布局不变，H、μ和Face embedding与原网络逐位一致；仅临时替换autoencoder.edge_embedding.weight。
原baseline H、Edge embedding及全部logits首先与归档逐位核对通过。
新head完整前向重复两次一致，804点FP=FN=0；结束时恢复原weight，全部参数哈希、恢复后的forward、RNG核验通过。
原checkpoint文件哈希未变化，没有保存带新head和旧Adam混用的训练checkpoint。

804行离线GEMM与4299行packed原head GEMM的数值结果不完全一样：
表示最大差异{actual['offline_embedding_max_abs_difference']:.9g}，logit最大差异{actual['offline_logits_max_abs_difference']:.9g}。
二者仍均严格成功。这也是必须做实际网络核验而不能只看离线MSE的原因。

## 结论范围

当前固定Decoder特征能经原形状线性Edge head读出804点的严格正确边图；成功不需要改变Encoder、Decoder或32维评分。
这使“当前特征完全不含可用信息”不再是充分解释；训练目标、参数更新尺度、联合优化与可读出性的数值条件仍需区分。
本轮未验证Face和其他mesh是否成功。Face embedding保持不变并不意味着实际Face图不变，因为Edge候选会变。
没有把单条离线解当作最终共享模型，也没有开始任何后续训练。

文件：offline/edge_head_candidate.pt是单独候选head；offline/readout.npz含H、目标、FP64/FP32权重与全部pair评分；
network_verification/actual_network_readout.npz保存实际网络结果；singular_values.csv及图保存完整奇异值。
'''
(root/'REPORT.md').write_text(report)
print(report)
