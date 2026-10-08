# 老师CAD50剩余网络的可运行复建

这是根据用户提供的老师ZIP、权重形状、配置与输出所构造的候选源码，**不是老师原训练源码，也不是原论文完整octree系统**。已执行CPU前向、噪声采样与三种指标的重放；没有重新训练老师模型。

原老师包：`69e9a5d3-44dc-4773-9713-f4ab473b475d.zip`（与更早老师ZIP字节相同）。
SHA256：`982686feb58c207932d9311786cabb3ffd7a1397627b83c44df3d5c8ab2189e5`。

## 1. 文件

- `models.py`：复建的拓扑Flow、最终点Flow/文本先验、count head、3D RoPE、AdaLN、两个Euler采样器。
- `teacher_ae.py`：沿用上一轮实际测试过的AE候选；2层早期/8层最终权重可加载。
- `indicators.py`：cosine/euclidean/spacetime三种边/面评分与完整边图三角候选恢复。
- `replay.py`：点生成、给定顶点拓扑生成、完整串联的统一入口。
- `replay_three_ae.py`：三方法早期AE重放。
- `objectives.py`：四组BCE、KL和两个flow matching目标的可微函数。**不是原始完整训练器**；缺失loss系数要求显式传入，训练采样/调度仍未精确恢复。
- `smoke_test.py`：strict加载与前后向梯度冒烟测试，不做optimizer更新。
- `REPORT.md`：本次结果、公式及边界。
- `evidence/`：实际预测、每条指标、有限前向假设测试、checkpoint结构与冷启动记录。
- `source_notes/`：直接复制老师包中的报告/配置，区分原始说明和本次推断。

## 2. 环境和输入

在已有包含PyTorch、NumPy、SciPy的隔离环境运行；不要为本任务重建正在训练的主环境。
CPU为本次实测设备，GPU可由`--device cuda:0`指定；CPU/GPU随机数序列和数值细节不保证逐位一致。

先将原老师ZIP解压到`TEACHER_ROOT`；该目录直接包含`data`、`results`。
本包不重复携带几百MB原权重。已有老师ZIP足够运行以下重放。

```bash
python smoke_test.py --teacher-root /path/to/TEACHER_ROOT --out runs/smoke.json

# 三种指标：都是自己的AE权重、自己的阈值
python replay_three_ae.py --teacher-root /path/to/TEACHER_ROOT --out runs/three_ae

# 最终AE重建：
python replay_three_ae.py --teacher-root /path/to/TEACHER_ROOT --out runs/final_ae --phase final

# 最终拓扑生成：只给真实顶点，不给真实拓扑；高斯噪声经50步Flow再解码
python replay.py --teacher-root /path/to/TEACHER_ROOT --out runs/topology \
  --stage topology --topology-seed 12345

# 点生成：输入缓存文本向量和噪声，点数由count head预测
python replay.py --teacher-root /path/to/TEACHER_ROOT --out runs/points \
  --stage points --point-seed 34567

# 完整串联：不输入真实顶点/GT面；每条先采样点，再采样拓扑
python replay.py --teacher-root /path/to/TEACHER_ROOT --out runs/cascade \
  --stage cascade --point-seed 34567 --topology-seed 12345

# 复用本程序已经生成的点以降低重复计算；会记录其文件哈希
python replay.py --teacher-root /path/to/TEACHER_ROOT --out runs/cascade_second_noise \
  --stage cascade --point-dir runs/points --topology-seed 23456

# 早期三种指标对应的拓扑Flow，同形状不等于同一套权重
python replay.py --teacher-root /path/to/TEACHER_ROOT --out runs/early_cosine_flow \
  --stage topology --phase early --method cosine
```

`--limit 2`仅用于接口检查；完整结果默认50条。输出目录必须不存在或为空，避免覆盖。
三种早期AE的联合入口是`replay_three_ae.py`。

## 3. 重要区别

- 三方法实验是三种**拓扑指标**对应的三套AE+Topology Flow，不是三个不同的点扩散网络；报告中的Flow loss均为MSE。
- 后期只续训spacetime，不能把最终spacetime F1当作与早期cosine/euclidean同预算的比较。
- 最终点checkpoint是18层，不是目录里早期config.json的6层。
- 最终点模型有`coordinate_prior`和`residual_scale≈1.219824e-5`；其高精度主要来自文本坐标先验。不是原始去噪器单独达到同样精度的证据。
- 缓存2048维文本向量作为模型输入；本包不包含原Text-to-CAD预训练模型或原生成代码，因此没有重新执行raw text→文本编码器→50个CAD代码生成。使用新文本需另行恢复同修订文本编码器、tokenizer和masked-mean pooling，不可用样本ID查表替代。
- 老师的点模型是有序点槽位与坐标生成，不是论文的分层octree占据扩散。
- 采样结果不是老师保存数组的精确拷贝；原噪声张量、所有原始实现细节和随机数流未全部提供，CPU重放的F1可能略高或略低。
- AE加载旧权重达到高F1不等于从零训练复现；本轮没有任何训练更新。

## 4. 权重和续训边界

`results/minkowski_target_099/latest.pt`保存的是Flow阶段optimizer；不是最终AE的optimizer。
`results/point_diffusion/latest.pt`是最终推理state_dict，没有完整optimizer/RNG。
`results/point_prior_candidate/candidate.pt`含与最终点网络逐项相同的state_dict；其中optimizer只覆盖6个coordinate_prior张量+1个residual_scale标量，**不覆盖整个18层点去噪器，且没有RNG**。
checkpoint source_step=267500与final frozen_denoiser_source_step=258000存在元数据差别；不能静默拼成唯一训练历史。

## 5. 本次目的

交付一个能实际载入和运行老师网络的对照起点，而不是再提出几轮loss猜测。
先冷启动重放，再将“从随机初始化训练”作为独立任务。原100条与现有CAD512训练目录不动。

## 6. 三种指标训练时的额外边界

AE的indicator thresholds沿用上一版作为buffer以重放现有值；cosine/euclidean原始训练时是可训练参数还是其它更新规则，state_dict自身不能唯一证明。当前实现不会自动训练这些buffer。要独立重训这些变体，需明确设置阈值参数化、初始化及temperature，并标注为训练实现选择。`cosine_scale=10`来自早期config；仅离散预测不能验证其它分支未知的正比例温度。
