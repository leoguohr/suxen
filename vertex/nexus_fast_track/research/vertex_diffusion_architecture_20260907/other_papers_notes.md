# Nexus Vertex Diffusion 结构调研：额外论文

核对日期：2026-09-07。只读论文、官方代码与本地已有参考仓库；没有下载 PDF、clone、改模型源码或训练。

## 结论

最有针对性的补充是 **Meta MeshFlow**：它实际比较了 Shape2VecSet 条件 cross-attention 与 3D RoPE 条件注入。其官方 DiT 确实采用 SA → CA → FFN 与 AdaLN-Zero，能为 Nexus 未公开的 block 细节提供有依据的候选。

**OctFusion**用于比较“八叉树逐层连续化生成”，而不是用其 U-Net 或隐式场解码替换 Nexus。目录里已有的 **qiisun/MeshFlow** 是另一篇同名论文，没有点云条件编码器。

## 1. Meta / HKUST MeshFlow，CVPR 2026

论文全名：MeshFlow: Efficient Artistic Mesh Generation via MeshVAE and Flow-based Diffusion Transformer。

- [官方论文 v1 §3.4](https://arxiv.org/html/2606.04621v1#S3.SS4)
- [附录 F：条件注入与模型规模](https://arxiv.org/html/2606.04621v1#S6)
- [官方项目页](https://mesh-flow.github.io/)
- 官方代码固定 commit：`55f56f60e1bbf98d1c1991670ac998094d5f59ae`

### 表示及输出路线

它生成 MeshVAE latent，解码得到连续顶点、法向、有效性 mask 和 adjacency embedding，再按边关系恢复三角面。不是八叉树孩子占用，也不使用 Marching Cubes。其主要生成问题更接近“联合几何拓扑 latent 生成”，不是 Nexus 独立 Vertex 阶段。

[论文 §3.1–3.3](https://arxiv.org/html/2606.04621v1#S3.SS1)；[代码 meshflow_vae.py L725–755](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_vae.py#L725-L755)

### 点云条件：作者明确比较两种路径

1. 最初设计：32768 个表面采样点 → 预训练的 Shape2VecSet 类编码器 → 2048 个 shape tokens → DiT cross-attention；顶点数量进入时间条件。
2. 最终路径：用训练 mesh 顶点的 XYZ 形成 3D RoPE 空间条件；作者观察到更快收敛，但真实顶点与推理表面点分布不同，因此先以 32³ 体素量化，减少分布差异。

以上是该文报告的经验，不能推导出“点云 CA 普遍不好”，也不能证明 Nexus 应去掉 VecSet。Nexus 规定联合训练 VecSet，而这里初版使用预训练编码器；生成目标也不同。

[附录 F 原文](https://arxiv.org/html/2606.04621v1#S6)；[pipeline.py L260–296：点云重采样和 voxel index 条件](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/pipelines/meshflow_pipeline.py#L260-L296)

### DiT block：官方源码确定内容

```text
hidden
  → Norm + time shift/scale → Self-Attention → time gate + residual
  → Norm                   → Cross-Attention → time gate + residual
  → Norm + time shift/scale → GELU FFN (4×)  → time gate + residual
```

- `adaLN_modulation` 输出 7×hidden：SA 的 shift/scale/gate、FFN 的 shift/scale/gate、CA 的 gate；零初始化。
- Self-attention 的 Q/K 均可施加 3D RoPE；head 维度等分 XYZ，每轴使用独立旋转频率。
- CA 中位置旋转是可选，只旋 Q；官方点云路线的 CA condition 是可选图像特征，点云本身通过 RoPE 传入。
- 有可选 U-Net 式跨 block skip connections、QK norm、DropPath，不能因为代码支持就断言论文 checkpoint 全部开启。
- 时间是 sinusoidal → MLP → 每块 AdaLN，输入点数还可投影加入时间特征。

[block L511–607](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L511-L607)；[3D RoPE L92–197](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L92-L197)；[SA Q/K L438–453](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L438-L453)；[CA Q-only L351–364](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L351-L364)

论文附录 F 明确：DiT 16 blocks、hidden 1536、895M 参数。公开源码构造函数的默认值是 24 blocks、hidden 1024，但实际入口从 checkpoint YAML 读取配置；两者不可混称。`facebook/meshflow/resolve/main/config.yaml` 本次返回 401，需要模型授权，因此本次未核实 checkpoint 的全部 flags。

[论文规模](https://arxiv.org/html/2606.04621v1#S6)；[按 YAML 建模 L859–886](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/models/meshflow_dit.py#L859-L886)；[模型页面的授权要求](https://huggingface.co/facebook/meshflow)

### Flow / sampler

论文用 `x(t)=(1-t)x_data+t epsilon`，目标速度 `epsilon-x_data`；从 t=1 向 0 采样。末期用 logit-normal 时间采样，推理 timestep shift=3.0。方向与此前 Nexus 草案的 noise→data 时间定义相反，不能直接混用变量或速度符号。

公开 pipeline 使用 `FlowMatchEulerDiscreteScheduler` 安排时间；真正更新在 `flow_sample` 中是 `latents -= distance[i] * prediction`，没有 DPM-Solver。默认 pipeline 为 28 步，实际读取模型 YAML；没有完整训练代码可核对时间分布实现。

[论文 Eq.(5)](https://arxiv.org/html/2606.04621v1#S3.SS4)；[scheduler 构造 L191–205](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/pipelines/meshflow_pipeline.py#L191-L205)；[Euler 更新 L198–225](https://github.com/facebookresearch/meshflow/blob/55f56f60e1bbf98d1c1991670ac998094d5f59ae/meshflow/pipelines/utils.py#L198-L225)

### 对 Nexus 的判断

可借：SA → CA → FFN 的时间调制布局；明确的 3D RoPE 维度拆分；条件坐标训练/推理一致性的验证。

不能直接搬：用表面点位置替代生成 latent 的 RoPE anchors 并删除 VecSet；MeshVAE 的 token/几何拓扑定义；16×1536 的规模；Euler 代替 Nexus 论文 DPM-Solver 后仍称完全一致。

Nexus 父节点位置来自八叉树历史，条件点云来自外部输入，两者不是同一组位置。仅给父节点 SA 加 RoPE，并不能自动替代从点云读取条件的通路。

## 2. OctFusion，SGP 2025

- [官方论文 v2 §3.3.2](https://arxiv.org/html/2408.14732v2#S3.SS3.SSS2)
- [附录 A.2.2 网络结构](https://arxiv.org/html/2408.14732v2#A1.SS2.SSS2)
- 官方代码固定 commit：`5fa85fbc4f3aa8fb2686fdc7619512a8004e2716`

OctFusion 对八叉树 splitting signal 与叶节点 latent 加噪。其八通道描述孩子是否进一步分裂，能一次推进两个深度；语义不同于 Nexus 每父节点的8-child“包含目标 mesh 顶点”占用。前4层固定全满，使用共享、嵌套多尺度 U-Net，按阶段训练时冻结已训练浅层部分。

网络是卷积/dual-octree graph U-Net 加注意力，不是 DiT。低分辨率三级通道64/128/256；高层通过dual-octree图卷积下采样接入低层 U-Net。时间 embedding 进入 residual blocks，类条件可与时间 embedding 相加。

最终表示为 octree latent → VAE/local SDF → MPU 全局连续 SDF → Marching Cubes mesh。论文展示无条件、类别、文字、sketch 条件；点云构建 VAE 输入不是一个已经证明的 Nexus 式 VecSet 条件分支。

可借其二值八叉树信号连续去噪和逐层闭环验证。不能把八通道形状相同视作目标相同，不能把33M U-Net作为Nexus约2B DiT的直接等价替代。

[表示 §3.2](https://arxiv.org/html/2408.14732v2#S3.SS2)；[时间注入 L105–114、L228–234](https://github.com/octree-nn/octfusion/blob/5fa85fbc4f3aa8fb2686fdc7619512a8004e2716/models/networks/diffusion_networks/graph_unet_hr.py#L105-L114)；[采样实现 L293–350](https://github.com/octree-nn/octfusion/blob/5fa85fbc4f3aa8fb2686fdc7619512a8004e2716/models/octfusion_model_union.py#L293-L350)

采样注意：代码参数名是 `ddim_steps`，但 x0 分支包含随机后验噪声，eps 分支才是相应确定性更新。不能只看参数名就称整个实现为DDIM，更不能称flow matching；论文Eq.(4–5)是扩散加噪与clean-data预测。

## 3. 本地 qiisun/MeshFlow：同名但不同论文

全名：MeshFlow: Mesh Generation with Equivariant Flow Matching，SIGGRAPH 2026。

- [作者项目](https://qiisun.github.io/MeshFlow/)
- 本地仓库：`external_references/MeshFlow`
- 本地固定 commit：`b78e2f6ee317056ec820e8d7cecc61a8b62fb441`

原生输出三角形 soup；EquiDiT 在顶点上编码，三顶点均值为面特征，面之间 self-attention，再广播回顶点。为保留集合置换等变，不用序列位置编码；输入坐标仍有 Fourier 特征，不能说完全没有坐标编码。

它是无点云条件生成，代码的条件是可选面数量。可参考 AdaLN-Zero 与线性 flow 接口，但不能借它确定 VecSet、点云 CA 或八叉树 RoPE。其嵌套最优传输匹配针对面/面内顶点置换，不应额外塞进已有空间对应的 Nexus 八叉树目标。

[EquiDiT L201–222](https://github.com/qiisun/MeshFlow/blob/b78e2f6ee317056ec820e8d7cecc61a8b62fb441/models/equidit.py#L201-L222)；[位置与面数条件 L247–264](https://github.com/qiisun/MeshFlow/blob/b78e2f6ee317056ec820e8d7cecc61a8b62fb441/models/equidit.py#L247-L264)；[线性 flow L20–42](https://github.com/qiisun/MeshFlow/blob/b78e2f6ee317056ec820e8d7cecc61a8b62fb441/flow_matching.py#L20-L42)

## 给下一版结构讨论的建议

1. 维持 Nexus 明确规定的点云 VecSet + CA、父节点 3D RoPE、depth embedding、共享 DiT、8通道速度输出。
2. 将 SA → CA → FFN + AdaLN-Zero 升级为“有多个公开实现可核查的复现候选”，仍不声称 Nexus 作者原结构。
3. CA 是否需要门控、是否对Q/K加入空间位置、归一化类型，都应单独写出选择，不能以“参考某模型”隐藏差异。
4. 优先验证条件使用：相同父节点/噪声换点云、变换坐标后的空间对应、真实父节点预测与自由展开分别评估。Meta 的训练/推理位置分布问题为此提供直接动机。
