# 本次从原始 ZIP 重新恢复的可运行代码

作者模型：gpt-6-astra；reasoning effort：xhigh；日期：2026-09-23；optimizer 更新：0。

本目录是本次重新编写、实际验证的实现。先重新读取原 ZIP 的全部 22 个 PyTorch 文件，取得张量结构、配置和训练状态；对不能由这些资料唯一确定的前向约定，显式沿用昨天已验证的候选。我们已经看过昨天代码，所以这不是盲逆向，也不是老师源码恢复。详见 `CONFIG_PROVENANCE.json` 的 A/B/C/U 证据等级。

- `recovered_networks.py`：AE、顶点条件拓扑 Flow、文本条件点网络/先验、严格加载与采样。
- `topology_scores.py`：边/面评分、完整预测边图三角候选、集合评价。
- `recovered_objectives.py`：可恢复的损失组件；不包含原训练循环。
- `verify_recovery.py`：全部 50 条 AE、新旧候选同运行时对照和少量实际点/拓扑采样。
- `verify_objectives.py`：固定输入下损失及梯度检查。
- `replay_cascade.py`：固定种子、顺序的全 50 条点到拓扑 CPU 串联。
- `verification_input.json`：原件 hashes、全部 state shapes、原保存 AE 面；后两者仅供加载验证/评价。
- `yesterday_cascade_reference.json`：昨天串联输出及来源 hashes，仅用于生成结束后的比较。

运行需要已有 PyTorch、NumPy、SciPy；不自动安装依赖。权重、文本条件和训练缓存来自原 ZIP，不在此目录重复复制。

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python verify_recovery.py \
  --teacher-root /path/to/original_assets \
  --baseline-root /path/to/yesterday_candidate \
  --out /path/to/empty_verification_output

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python replay_cascade.py \
  --teacher-root /path/to/original_assets \
  --out /path/to/empty_cascade_output
```

`--baseline-root` 可省略；老师原 ZIP 至少需要这六个原件，路径保持不变：

```text
data/point50/training.pt
results/point_diffusion/text_conditions.pt
results/point_diffusion/latest.pt
results/minkowski_target_099/vae.pt
results/minkowski_target_099/best_flow.pt
results/minkowski_target_099/latent_normalization.pt
```

本次验证在 CPU、单线程、torch `2.3.0a0+6ddf5cf85e.nv24.04` 执行，显式关闭 MHA fastpath。AE 用标准 `TransformerEncoderLayer`，所有 state key 与老师原件一一同名，无忽略项、无 key 重命名。

小样本 `verify_recovery.py` 每个样本分别重置点 seed34567、拓扑 seed12345，GT 顶点先恢复老师原始顺序；全量 `replay_cascade.py` 每个 generator 仅初始化一次，按00→49顺序消耗噪声，拓扑使用新生成点的顺序。这两种检查不能用同名 seed 直接混作同一实验。

损失组件只表达可检验公式。KL 函数返回每元素值，原归约需要另行选择；point count 系数必须显式传入；`cached_prior_mse` 的原缓存生成办法未知。没有新训练，没有恢复到“从零训练必然达到老师指标”的程度。
