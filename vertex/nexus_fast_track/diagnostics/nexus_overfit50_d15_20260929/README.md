# NEXUS 固定50物体 D15 等权续训

用户授权：从 OwnAE-v2 固定100条 UID 中均衡选50条；双A10080GB；新增10000次optimizer更新。只开展Vertex Generation，不修改V2。

## 固定协议

来源Phase2 checkpoint-012000.pt，SHA256 `44ca719f853f064e60024fd09a9c9d021a9455013af09577145ae139841b5501`。恢复R1模型、907项Adam状态和RNG，D15、VecSet、DiT、velocity MSE、lr1e-5、WD0、clip1、BF16不变。

每个全局更新8个microbatch，两卡各4个，DDP均值归约；不扩大有效batch。rank0按全局micro顺序生成噪声和时间并广播，保留一条可恢复的全局随机流。10000更新合计80000份microbatch，每个对象×层级106或107份。均衡层级调度从新50物体列表起点开始。

## 数据选择

使用实际OwnAE run/data_manifest.json的100个UID，只读取其清单及数据，不修改AE工程。按D15唯一顶点数、全部层父格总数、UID排序，分10个等数量组，各取5个分散位置；碰撞样本在同组内替换，保留同组内已有Phase2训练UID。

选中50个对象，18–2547个D15唯一顶点，共53200点；与Phase2训练集重合2个UID：001045、000084。7个原始候选有D15量化碰撞，全部记录在pool100_audit.json中，本轮未选。原始重复顶点记录数单列；使用现有D15集合编码，不把重复记录数当生成点数。

22个入选对象的D15点数大于OwnAE的D9点数。这是从Stage2原始浮点几何重新编码得到，不使用V2的D9顶点做D15伪标签。未来接AE时仍需单独处理顶点身份对应，本轮不宣称已串联AE。

## 运行

服务器工作根目录：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_overfit50_d15_20260929`。

```bash
CUDA_VISIBLE_DEVICES=GPU-b3c81e4b-1632-ceea-f4f4-97e3e5f84d4e,GPU-0cb18edf-c41d-02e8-3b12-14d55a283546 OMP_NUM_THREADS=8 UCX_VFS_ENABLE=n PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /guohaoran/envs/nexus-algo/bin/python -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_overfit50_d15_20260929/runtime/scripts/overfit50_pipeline.py
```

训练最多新增10000更新，累计至22000；每200更新及首步保存完整状态。NVME写入后校验SHA，再复制SSD、回读SHA、原子提交身份清单。仅在本新目录中保留最近两份完整权重，所有历史身份JSON和逐步日志保留；原Phase2迁移起点不删除。

## 训练后评估

训练期间不评估。成功完成并校验最终checkpoint后，两卡分别评估97029000、97029001，50对象×2新种子=100条完整树。每层20 Euler、15层最多300次DiT调用；阈值0.5；原4096父格保护阈值保持，超出记capacity_abort，不截断或修补。

生成先保存原生整数点、XYZ及逐层数组并哈希，然后独立读取GT评分。输出完整树、逐层occupancy、首错层、50×50条件匹配矩阵与XYZ指标。另做全部50对象depth2–5的GT-parent纯噪声生成诊断，这不是完整生成成绩。

原始重复GT和显式唯一几何GT的XYZ指标分别报告；不称未见物体泛化、CAD50成绩或老师同条件对比。最终自动生成不含大权重的结果ZIP。
