# mini-Nexus：Nexus 核心算法的独立教学复现

这是依据 Nexus 论文公开方法编写的**独立实现**，不是作者官方代码，也不是论文中约 20 亿参数模型的等价替代品。

当前目标是先把四条核心链路做成可运行、可测试的 PyTorch 代码：

1. 把顶点量化为八叉树，并逐层构造每个父节点的 8 维 multi-hot 标签；
2. 用带 3D RoPE 和 learnable depth embedding 的条件 DiT 学习 flow-matching 速度场；
3. 用一阶、二阶 Spacetime Interval 从逐顶点 embedding 恢复边和面；
4. 用 topology autoencoder 与 topology latent flow 组成 Nexus 第二阶段的最小闭环。

2026-09-08 的点云 Vertex 入口已接入独立的 8 层 VecSet 与 Vertex DiT，默认
配置为条件宽 2048、DiT 36 层宽 1536；结构选择、缩小配置及尚未接入的
DPM-Solver 见 [`VERTEX_DIFFUSION.md`](VERTEX_DIFFUSION.md)。可变顶点数 batch、mask、checkpoint/resume 和
`torchrun` DDP 入口见 [`ALGORITHM_TRAINING_GUIDE.md`](ALGORITHM_TRAINING_GUIDE.md)。

## 它与两份参考代码的关系

- Hunyuan3D-2.1 提供了“条件编码器 → latent flow DiT → ShapeVAE 解码 → 表面提取”的成熟参考。mini-Nexus 借鉴条件注入、时间编码和 flow 训练骨架，但不使用它的 ShapeVAE/隐式场输出。
- MeshFlow 提供了“高斯噪声 → 坐标速度场 → triangle soup”的直接 mesh 生成参考。mini-Nexus 借鉴直接在结构数据上做坐标/占用流的思路，但不采用 triangle soup 表示。
- Nexus 自身使用“八叉树顶点流 → topology latent 流 → Spacetime 解码”。新 Vertex 提供完整尺寸配置，toy 使用同一 Vertex 实现的缩小配置。

## 目录

```text
mini_nexus/
├── mini_nexus/              # 核心算法
│   ├── flow.py              # flow-matching 与 Euler ODE
│   ├── octree.py            # 八叉树标签、展开与顶点恢复
│   ├── vertex.py            # 新点云 VecSet 与 Vertex DiT
│   ├── models.py            # Topology Flow 原型使用的小网络
│   └── topology.py          # topology AE、Spacetime Interval、边面恢复
├── scripts/
│   ├── train_toy.py         # 在一个立方体上做最小过拟合实验
│   ├── train_pilot.py       # 真实 pilot 的 batch/DDP 三阶段训练入口
│   ├── train_topology_ae_overfit_packed.py
│   ├── run_topology_ae_overfit20_packed.sh
│   └── run_topology_ae_overfit20_fp32_flash_bf16.sh
├── tests/                   # 数学、shape 与训练 smoke tests
├── requirements.txt         # 本机隔离环境依赖
└── environment-a100.yml     # A100 服务器建议环境
```

当前 Topology AE 的唯一阅读路径见
[`TOPOLOGY_AE_CODE_PATH.md`](TOPOLOGY_AE_CODE_PATH.md)。废弃 Medium、旧教学和
转发别名脚本已移出当前源码；历史实验需要使用对应完整快照。
当前 Topology AE 保留模块内部 LayerNorm，在 Encoder/Decoder 末端增加
LayerNorm；输出 embedding 只中心化，不使用 RMS 归一化。Edge/Face 独立
logit scale 校准与 Face 面积因子仍保留。

## 本机运行

```bash
cd /Users/luthier/Documents/sophomore/nexus_fast_track/mini_nexus

../.envs/nexus-algo/bin/python -m pytest -q
../.envs/nexus-algo/bin/python scripts/train_toy.py \
  --vertex-steps 10000 \
  --topology-steps 1500 \
  --latent-flow-steps 4000 \
  --ode-steps 100 \
  --output outputs/toy_final
```

输出会写入 `outputs/toy/`。这些输出只证明算法闭环与代码接口成立，不代表论文质量。

## A100 上的下一步

服务器的持久化目录布局、独立环境安装和逐条验证命令见
[`SERVER_A100_GUIDE.md`](SERVER_A100_GUIDE.md)。每次登录后先执行：

```bash
source /guohaoran/nexus_fast_track/mini_nexus/scripts/server_env.sh
```

先运行 toy 和 32 条 pilot 的单样本过拟合，确认以下 Gate，再扩数据：

- 八叉树 round-trip 完全一致；
- vertex-flow 的 occupancy IoU 明显高于随机基线；
- topology AE 能在训练样本上恢复绝大多数边和面；
- topology-flow loss 能稳定下降；
- 同一个 UID 的 condition、vertex 与 topology 使用同一 canonical vertex identity。

在读取 32 条 pilot 前先运行：

```bash
../.envs/nexus-algo/bin/python scripts/audit_pilot_for_algorithm.py
```

它只读检查 Stage-2 输出，并把 depth-9 量化冲突或上游 quarantine 明确拦下，不会悄悄 merge 顶点。

对通过审计的一条真实数据执行全链路 forward/backward smoke test：

```bash
../.envs/nexus-algo/bin/python scripts/smoke_real_pilot.py --uid nexus_pilot_0032
```

这一步只证明真实 tensor 能接入每个模块，不代表训练收敛或生成质量。
