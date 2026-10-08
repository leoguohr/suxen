# 与昨天代码的实质对照

由 gpt-6-astra/xhigh 亲自完成结构、公式、固定输入与实际采样判断。机械 hashes 只是来源核验。昨天包位于 `/Users/luthier/Downloads/Nexus_teacher_full_reconstruction`；对应 hashes 记录在验证 JSON 和上级 `DERIVED_RECOVERY_EVIDENCE.json`。

| 昨天 | 本次 | 一致之处 / 实际实现差异 | 证据 |
|---|---|---|---|
| `teacher_ae.fourier_positions` | `position_features` | 坐标主序6频带、π、拼接相同 | 原39输入维+承接已验证约定；全50 AE最终embedding相同 |
| `GraphBlock` | `VertexFaceMessage` | 顶点–面双向均值聚合、self/neighbor投影、LN/GELU残差相同；独立编写索引构造 | 原graph键形状+逐mesh前向 |
| 手写 `TransformerBlock` 的QKV/SDPA | 原生 `nn.TransformerEncoderLayer` | 4heads、pre-norm、GELU、FF512相同；本次调用标准模块，全部state键保持identity | 新AE244项strict加载；50/50 edge/face embedding最大差0 |
| `ReconstructedTeacherAE` | `TopologyAutoencoder` | 相同8层、128宽、64latent、两个32维输出；仍用mu重放 | 原件结构+全50重建 |
| `time_embedding` / `rope_3d` | `time_features` / `rotary_xyz` | 数学约定相同，reshape与浮点算术结合顺序不同 | 固定同输入对照；不据此声称原公式唯一恢复 |
| `DiTBlock` | `ConditionalBlock` | qkv、out、FF、6路AdaLN相同；本次广播布局为[B,1,D] | 112/206项strict；共享输入velocity最大差2.3842e-6 |
| `TopologyFlow` | `VertexConditionedFlow` | 10层、144宽、64通道、位置条件相同 | 原件config/state；实际50步采样 |
| `PointFlow` | `TextConditionedPointFlow` | 18层、有序274slots、count275、512宽先验、scale相同 | 原件206项；当前全50 count正确 |
| `sample_points` / `sample_topology` | `generate_points` / `generate_topology_latent` | 相同Euler公式；dt乘法写法改为除步数，未调阈值、温度或权重 | 2样本相同噪声：点最大差1.8626e-9，latent最大差9.5367e-7，面集合相同 |
| `indicators.py` | `topology_scores.py` | 三种logit公式和`>0`规则相同；dict字段改为edges/faces/candidates | 全50 AE离散面与昨日50/50相同 |
| `hard4_bce` | `grouped_bce` | 四组均值、空组0、固定除4相同 | 固定输入值和有限梯度检查 |
| `kl_standard_normal` | `standard_normal_kl_elements` | 每元素公式相同；本次不预先替用户固定mean归约，调用`.mean()`与昨天数值相同 | 原KL归约未恢复；loss JSON对照 |
| 其余3种训练loss组件 | 对应新组件 | 公式同义，参数名/返回字段不同；count权重仍显式要求 | 七项固定输入loss比较差值全0 |
| `replay.py` | `verify_recovery.py` / `replay_cascade.py` | 本次分别提供全50 AE与完整串联入口；串联先完成全部生成，再载入GT和旧预测评价 | 命令日志、50个新预测、结果JSON |

“重新编写”没有要求故意改变网络。相同的原权重与已验证约定理应给出相同功能；这里可复查的是来源、独立实现、完整严格加载和实测。未把代码换名、字节差异或浮点差异说成新的网络发现。

原训练器、负例精确采样、训练dropout、后验clamp、KL归约、阶段初始化、原点缓存构造仍未恢复。昨天README/REPORT已经明确提过Flow优化器归属、7个prior槽、step语义及258000/267500差异；本次从原件重新确认，并扩展为全部22个checkpoint的证据表。
