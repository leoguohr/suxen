# CAD50 Face-Finish 训练方法（逆向重建包）

**只含训练方法。网络结构不在本包范围内** —— 按下面 §2 的接口契约把你已还原的网络接上即可。

来源：`CAD50_Face47_and_fixed100_preflight_01.zip`（sha256 `e2ba3b93…`）
重建方式：静态源码分析 + numpy 独立重推导验证 + 训练日志考古

---

## 1. 这是什么训练方法（30 秒版）

在一个**已收敛的父模型**上做的**局部微调**，不是从零训练。

| 项 | 值 |
|---|---|
| 可训练参数 | **32,800**（`face_embedding.weight` 1024×32 + `face_embedding.bias` 32） |
| 冻结 | 其余全部（父代 412 个可训练张量中的 410 个） |
| 优化器 | AdamW `lr=1e-4, betas=(.9,.999), eps=1e-8, wd=0.01`，**继承父状态、不重置** |
| 批 | 每次更新 **5 个完整 mesh**；10 次更新 = 1 epoch |
| 损失 | **Hard4** —— 按 TP/TN/FP/FN 四组，**每组等权 1/4**，组均值再平均 |
| 负样本 | `ceil(1.5×|gt_faces|)` 无状态随机负样本 ∪ **固定 47 个 hard negative** |
| 预算 | 500 步，每 50 步存盘+评估，目标 face micro-F1 ≥ 0.997 |

**核心提醒**：本方法里的 `edge_logits` / `face_logits` 是**固定解析公式**，不含可学习参数。
真正被训练的只有那个 1024→32 的线性投影。详见 `TRAINING_METHOD.md` §2。

---

## 2. 接口契约

你的网络必须提供：

```python
model.face_embedding : nn.Linear(decoder_width, 32)   # ← 唯一可训练
model.edge_embedding : nn.Linear(decoder_width, 32)   # ← 冻结

model.forward(vertices, faces, *, sample_latent=False, graph=None) -> dict
    "edge"            [V+F, 32]            去均值嵌入
    "face"            [V+F, 32]            去均值嵌入
    "decoder_hidden"  [V+F, decoder_width] ← 缓存边界
```

数据契约（每个 mesh 一个 dict）：

```python
{
  "uid"         : str,
  "vertices"    : float32 [n, 3],
  "faces"       : int64   [m, 3],
  "gt_faces"    : int64   [k, 3],   # 排序去重的真值三元组
  "pairs"       : int64   [P, 2],   # 上三角顶点对
  "edge_labels" : float32 [P],      # 1.0 = 真实边
}
```

自检：

```bash
python code/trainer_reference.py --check-contract
```

---

## 3. 快速验证（只需 numpy）

先跑这个确认数学重建无误 —— 它不依赖 torch：

```bash
python code/verify_math.py
```

预期输出：`face_logits` 与"16 维半空间三角形面积平方之差 × 0.398" **MATCH**；
`edge_logits` 与"‖Δ‖² 之差 × 0.9306" **MATCH**；Lagrange 恒等式成立。

---

## 4. 目录结构

```
CAD50_训练方法_v1/
├── README.md                      ← 本文件
├── TRAINING_METHOD.md             ← 训练方法完整规格（超参全表、公式推导、调度、协议）
├── FAILURE_ANALYSIS.md            ← 为什么这套方法实测没修好 + 修改建议
├── MANIFEST.txt
├── code/
│   ├── objective.py               ← 解析分类器 + Hard4 损失（网络无关）
│   ├── schedule.py                ← 无状态种子调度 + 负采样
│   ├── determinism.py             ← 可复现性配置（位级一致所必需）
│   ├── trainer_reference.py       ← 训练主循环参考实现
│   └── verify_math.py             ← numpy 独立验证（无 torch 依赖）
└── evidence/                      ← 原始运行证据（未改动）
    ├── updates.jsonl              ← 500 步训练日志（真实记录）
    ├── PROTOCOL.json              ← 训练协议与冻结哈希
    ├── TRAIN_COMPLETE.json        ← 收尾回执与最终检查点哈希
    ├── TRAIN_FAILURE_attempt1.json← 第一版崩溃的真实 traceback
    └── eval_trajectory.md         ← 11 次评估的完整轨迹
```

---

## 5. 三个必须知道的坑

### 5.1 分类器不是 head

`face_logits` / `edge_logits` 是固定公式：

```
logit_edge(i,j)   = 0.9306077080970389 · ( ‖Δ_ij‖²_space − ‖Δ_ij‖²_time )
logit_face(i,j,k) = 0.39804385828730726 · ( Area_space² − Area_time² )
```

32 维嵌入切两半（前 16 "space" / 后 16 "time"）。`(‖u‖²‖v‖²−(u·v)²)` 是 Gram 行列式
= `‖u×v‖²` = `(2·Area)²`，乘 `FACE_FACTOR=0.25` 后即 `Area²`。

**face 分类器没有 bias 项** —— 阈值被硬钉在 0。这是本方法最大的结构性缺陷，见 `FAILURE_ANALYSIS.md`。

### 5.2 续训从 epoch 中途开始

`epoch, batch_index = divmod(total_update − 1, 10)`，用的是**全局** update 序号。
父模型停在 19356，所以第一次新更新的 `batch_index = 6` —— **不是 epoch 起点**。

第一版因为断言了错误的参与度分布而在第 50 步崩溃（见 `evidence/TRAIN_FAILURE_attempt1.json`）。

### 5.3 优化器建在 412 个张量上，只训 2 个

AdamW 覆盖父代全部可训练张量（Adam 状态完整继承），随后冻结 410 个。
`opt.step()` 对 `grad is None` 的参数直接跳过，所以冻结项**状态保留、永不推进**。
`trainer_reference.py` 用 `frozen_hash` + `adam_frozen_hash` 每 50 步双重校验。

---

## 6. 实测结局（重要）

**这套方法没有达成目标。** 500 步预算耗尽，最佳 face F1 = **0.995265 @ new_updates=0**
—— 就是父模型本身，**一步都没超越**。目标 0.997 差 0.0017。

病征：`face_fp` 47→41（↓）、`face_fn` 6→17（↑），边界单侧漂移。

根因：Hard4 的等权分组使**每个 FP 的权重是每个 TN 的 178.7 倍**，梯度被极少数稀有样本主导。

完整机理与改法见 `FAILURE_ANALYSIS.md`。**若你要复现此配方，建议先看那一份。**

---

## 7. 复现所需外部依赖（不在本包内）

```
<PARENT>/B_v2_teacher_blocks/checkpoint-19356.pt              # 2.96 GB，sha256 6f965d38…
<PARENT>/B_v2_teacher_blocks/evaluations/eval-19356.json
<PARENT>/B_v2_teacher_blocks/evaluations/*/<uid>.npz          # 父预测，用于提取 47 个 hard negative
<DATA>/manifest.json                                          # sha256 4742e72b…
<POOL>/pool_manifest.json                                     # sha256 86e7895f…
```

数据集规模硬断言（原配方要求）：

```
meshes=50, vertices=2872, edges=8364, faces=5576, pairs=235741
```

运行环境：Python 3 + PyTorch(CUDA) + numpy。
