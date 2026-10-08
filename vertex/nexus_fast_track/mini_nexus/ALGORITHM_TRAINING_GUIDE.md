# mini-Nexus 算法训练讲义：3D RoPE、VecSet、batch 与 DDP

## 1. 先看清当前数据 Gate

32 条 pilot 完成数据处理后：

- Stage 1：27 accepted，5 quarantined；
- Stage 2：26 accepted，6 quarantined；
- depth-9 算法审计：1 ready，31 quarantine。

最后一项严格得多。Nexus Vertex Stage 把一个 depth-9 cell 当作一个顶点。如果两个 canonical vertices 落进同一个 cell，Vertex Stage 只能输出一个 cell，而 Topology Stage 仍可能引用两个不同 vertex identity。当前规则禁止静默 merge，因此真实训练集暂时只有 `nexus_pilot_0032`。

这不妨碍验证 batch/DDP 代码，却意味着当前实验是单样本过拟合，不是论文规模训练。

## 2. 正式 VecSet 条件编码器

输入是：

```text
condition: [B,8192,6] = xyz + normal
```

编码数据流：

```text
8192 个点
  → 对 xyz 做 FPS（最远点采样）
  → 选择 K 个输入相关 anchor
  → xyz Fourier embedding，并保留 normal 属性
  → anchor 作 query，全部 8192 点作 key/value
  → cross-attention + FFN
  → VecSet condition tokens [B,K,C]
```

旧教学版使用 K 个可学习 query，不依赖输入点的位置。现在的实现与公开 3DShape2VecSet encoder 的关键结构一致：anchor 来自输入点的 FPS，而不是凭空学习的一组 query。

对应代码：

- `farthest_point_sample`：逐步选择离已选集合最远的点；
- `PointFourierEmbedding`：把 xyz 变成多频率 sin/cos 特征；
- `VecSetConditionEncoder`：FPS + cross-attention + FFN。

注意：Nexus 作者没有公开代码和完整超参数，因此这是基于论文与公开 3DShape2VecSet 的独立实现，不能称为作者源码逐行复刻。

## 3. 3D RoPE 到底放在哪里

Transformer self-attention 先得到：

```text
Q,K,V: [B,H,N,D_head]
position: [B,N,3]
```

把每个 head 的一部分通道分给 x、y、z 三轴，每轴两两成对旋转：

```text
[q_even,q_odd]
  → [q_even*cosθ - q_odd*sinθ,
     q_even*sinθ + q_odd*cosθ]
```

其中 θ 由该 token 的 x/y/z 坐标与不同频率共同决定。Q 和 K 同样旋转，V 不旋转。因此 attention score 会自然包含相对三维位置。

在 Vertex Stage：

- token 是 occupied parent；
- position 是 parent 的三维 octree code；
- 另加 learnable depth embedding，告诉统一网络当前在第几层。

在 Topology Flow：

- token 是 vertex latent；
- position 是 normalized vertex coordinate；
- 不使用 octree depth embedding。

对应代码：`apply_3d_rope`、`RotarySelfAttention` 和 `ConditionalFlowTransformer`。

## 4. batch 怎么容纳不同大小的 mesh

同一批对象可能分别有 572、1000、5000 个顶点，不能直接 `torch.stack`。当前合同是：

```text
vertices:    [B,Vmax,3]   不足位置补 0
vertex_mask: [B,Vmax]     真顶点为 True，padding 为 False
condition:   [B,8192,6]   点数固定，可以直接 stack
faces:       tuple([F1,3], [F2,3], ...)
```

Vertex Stage 在同一 depth 把每个对象的 parent tokens pad 到 `Nmax`，flow loss 只计算 mask=True 的 token。

Topology AE 因每个对象 graph 和 face index 不同，在一次 batch forward 内逐对象计算图消息和 loss，最后取平均。它保持 vertex identity 正确，代价是没有完全向量化。

对应代码：`PilotBatch`、`collate_pilot_samples`、`collate_octree_level`。

## 5. 为什么训练拆成三个命令

### 5.1 Vertex Stage

学习：

```text
condition + parent position + depth + noisy 8-way occupancy
  → 8-way occupancy velocity
```

```bash
python scripts/train_pilot.py \
  --stage vertex \
  --steps 1000 \
  --batch-size 1 \
  --output outputs/pilot_training/vertex
```

### 5.2 Topology Autoencoder

学习把真实 edges/faces 压到 per-vertex latent，再用一阶、二阶 Spacetime Interval 恢复 topology：

下面是 pilot 教学命令，不是当前末端 LayerNorm、输出仅中心化的 20-mesh 实验入口。
当前调用链见 `TOPOLOGY_AE_CODE_PATH.md`。

```bash
python scripts/train_pilot.py \
  --stage topology-ae \
  --steps 1000 \
  --batch-size 1 \
  --output outputs/pilot_training/topology-ae
```

### 5.3 Topology Latent Flow

它需要已经训练好的 topology AE checkpoint：

```bash
python scripts/train_pilot.py \
  --stage topology-flow \
  --topology-ae-checkpoint outputs/pilot_training/topology-ae/checkpoint-001000.pt \
  --steps 1000 \
  --batch-size 1 \
  --output outputs/pilot_training/topology-flow
```

`--batch-size` 是每张 GPU 的 batch size。全局 batch 为：

```text
per_gpu_batch × GPU 数量
```

## 6. 多卡 DDP 怎么启动

四张 GPU 的例子：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 scripts/train_pilot.py \
  --stage vertex \
  --steps 1000 \
  --batch-size 2 \
  --output outputs/pilot_training/vertex-ddp4
```

`torchrun` 会启动四个进程。每个进程绑定一张 GPU，`DistributedSampler` 分配不同数据，DDP 在反向传播时同步梯度。

当前服务器只有一张 A100，可以真实测试 DDP 包装、sampler 和 checkpoint 路径：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 scripts/train_pilot.py \
  --force-ddp \
  --stage vertex \
  --steps 1 \
  --batch-size 1 \
  --output outputs/ddp1-smoke
```

这能证明代码进入 DDP 路径，但不能证明多张 GPU 的 NCCL 通信已经在这台机器上验证。

## 7. checkpoint 与断点续训

训练目录包含：

```text
config.json       数据 UID、设备、world size、超参数
train.jsonl       每一步 loss、gradient norm、depth 和 UID
checkpoint-*.pt   模型、优化器、step、参数与 ready UID
```

断点续训示例：

```bash
python scripts/train_pilot.py \
  --stage vertex \
  --steps 2000 \
  --resume outputs/pilot_training/vertex/checkpoint-001000.pt \
  --output outputs/pilot_training/vertex
```

这里的 `steps=2000` 表示最终走到第 2000 步，不是再额外跑 2000 步。

## 8. 当前能说和不能说的结论

完成单元测试、A100 三阶段 smoke 和单进程 DDP 后，可以说：

- 3D RoPE、VecSet、padding/mask batch 和 DDP 训练入口可执行；
- 一条真实 pilot 能完成 forward/backward/checkpoint；
- 脏数据和量化碰撞被显式阻断。

仍不能说：

- 已复现论文质量；
- 20 亿参数模型已经训练；
- 32 张 A100 配置已经实测；
- 量化碰撞已经解决。
