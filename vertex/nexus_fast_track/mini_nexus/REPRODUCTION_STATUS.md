# mini-Nexus 复现状态

2026-09-08 Vertex 更新：当前 `VertexStageSystem` 已切换到新点云 VecSet +
Vertex DiT，结构与实现边界见 [`VERTEX_DIFFUSION.md`](VERTEX_DIFFUSION.md)。
下文为 2026-08-21 的历史小模型记录，其中的 toy 指标、测试数量及后续计划
不代表新 Vertex 或当前 Topology AE 状态。

更新日期：2026-08-21

## 当前结论

算法工作已经可以开始，而且第一版核心闭环已经实际运行。当前完成的是**独立的 mini-Nexus 核心复现**，不是论文规模训练，也不是作者官方复现。

## 已完成并验证

- 单独算法环境：`.envs/nexus-algo`，未修改 `.envs/nexus-data32`。
- 官方参考代码快照：Hunyuan3D-2.1 与 MeshFlow，commit 已写入 `REFERENCE_CODE_MAP.md`。
- depth-wise octree、多父节点 8-child multi-hot 标签与 leaf center 解码。
- FPS + Fourier embedding + cross-attention VecSet condition encoder。
- 带 3D RoPE、learnable octree depth embedding 的 flow Transformer。
- 线性 flow matching、velocity MSE、Euler ODE。
- topology KL autoencoder、独立 edge/face spacetime embeddings。
- 一阶/二阶 Spacetime Interval。
- all-pair edge supervision、all-positive + sampled-negative face supervision。
- sparse edge graph 3-cycle enumeration 与 face recovery。
- topology latent flow。
- variable-length vertex padding/mask、每对象 faces 与同层 octree batch。
- 三阶段 batch 训练、checkpoint/resume 与 `torchrun` DDP 入口。
- 13 个自动测试全部通过。
- 一个真实 pilot 样本完成三个模块的一次 forward/backward。

## Toy 闭环的真实指标

命令：

```bash
../.envs/nexus-algo/bin/python scripts/train_toy.py \
  --vertex-steps 10000 \
  --topology-steps 1500 \
  --latent-flow-steps 4000 \
  --ode-steps 100 \
  --output outputs/toy_final
```

结果来自 `outputs/toy_final/metrics.json`：

| 指标 | 结果 |
|---|---:|
| vertex cell IoU | 1.0 |
| 生成顶点数 / 目标顶点数 | 8 / 8 |
| topology AE edge F1 | 1.0 |
| topology AE face F1 | 1.0 |
| topology flow edge F1 | 1.0 |
| topology flow face F1 | 1.0 |
| topology flow loss | 12.5355 → 0.0639 |
| 导出 OBJ | 8 vertices / 12 faces，face index 最大 7 |

这只证明模型能在一个立方体上过拟合并生成合法 mesh，不能外推为真实生成质量。

## 当前真实 pilot 状态

32 条数据已全部处理：Stage 1 为 27 accepted / 5 quarantined，Stage 2 为
26 accepted / 6 quarantined。进一步做 depth-9 算法审计后，只有
`nexus_pilot_0032` ready（572 vertices、1140 faces、8192 point+normal）；
其余 31 条因上游 quarantine 或 vertex quantization collision 被拦截。

`nexus_pilot_0032` 已完成真实 shape 的 one-step smoke：vertex flow、topology AE、topology flow 均可 forward、计算 loss 和 backward。它不是 overfit 结果。

## 尚未完成

- 尚未对 `nexus_pilot_0032` 做完整单样本过拟合。
- DINOv3 image branch、TP/TN/FP/FN balanced edge loss 尚未实现。
- all-pair edge loss 尚未做大 mesh 的 pair chunking。
- 当前服务器只有一张 A100；多进程 DDP 尚不能在本机做真实多卡通信验证。
- 量化碰撞如何修复尚未获批，不能擅自 merge vertex 并重映射 topology。
- 没有 Nexus 作者代码和权重，论文级 2B 模型指标未验证。

## 下一 Gate

只做一件事：在 A100 上对 `nexus_pilot_0032` 做单样本 overfit，并分别保存：

1. Vertex Stage checkpoint 与每层 occupancy 指标；
2. Topology AE checkpoint、edge/face F1；
3. 冻结 AE 后的 topology-flow checkpoint、最终 mesh；
4. config、seed、输入/输出 hash 和显存峰值。

单样本 Gate 通过以后，需要先冻结量化碰撞规则，才能形成真正的多对象训练集；
当前不应声称已经启动论文规模训练。
