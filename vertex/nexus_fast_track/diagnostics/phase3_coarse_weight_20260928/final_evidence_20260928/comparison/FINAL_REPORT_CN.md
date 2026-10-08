# Phase3-A final comparison

同一10个seen、2个unseen对象及原4个种子。纯噪声开始、自己的父格、D15、20 Euler steps/layer、阈值0.5。
这是加权续训2000步后的固定回归比较；没有等预算未加权续训对照，改善不能单独归因为loss权重。

| split | Phase2完整树 | Phase3完整树 | Phase2首错2–5 | Phase3首错2–5 |
|---|---:|---:|---:|---:|
| seen | 16/40 | 7/40 | 19 | 20 |
| unseen | 0/8 | 0/8 | 8 | 8 |

全部15层occupancy accuracy/F1/TP/FP/FN与首错见对应CSV；accuracy分母为GT与预测父格并集的8个子格。
点数与XYZ诊断、原生整数顶点和每层完整数组保留在post_training_final_014000。
本轮预算结束，不自动继续训练。最终权重身份见run/checkpoint_identity.json；权重张量不打包。
