# CAD50 Face-Finish 训练方法完整规格

> 本文件只描述**训练方法**。网络结构不在范围内（委托方已自行还原）。
> 所有数值均为原配方精确值，未经"整理"或近似。

---

## 目录

1. [训练范式](#1-训练范式)
2. [目标函数](#2-目标函数)
3. [分类器](#3-分类器解析式)
4. [可训练参数与冻结](#4-可训练参数与冻结)
5. [优化器](#5-优化器)
6. [数据调度](#6-数据调度)
7. [负样本采样](#7-负样本采样)
8. [缓存机制](#8-缓存机制)
9. [梯度处理](#9-梯度处理)
10. [检查点与评估协议](#10-检查点与评估协议)
11. [可复现性配置](#11-可复现性配置)
12. [超参全表](#12-超参全表)

---

## 1. 训练范式

**局部微调（local fine-tune）/ 有界续训**，不是从零训练。

父模型 `checkpoint-19356.pt` 已完成 19,356 次更新。本次训练：

- 从父模型**无缝续起**（Adam 状态、RNG 状态、数据调度位置全部继承）
- 只解冻最后一层 face 嵌入投影
- 硬上限 500 次新更新
- 达标即停（face micro-F1 ≥ 0.997）

**目标**：把 face 分支的 47 个 FP + 6 个 FN 压下去。

**约束**（README 明示）：`upstream / final LN / Edge head 全程冻结`。

---

## 2. 目标函数

### 2.1 总体结构

每个 mesh 的损失 = **face Hard4 + 常量 edge 项**，整个 batch 取均值：

```python
for uid in batch:                              # batch = 5 个完整 mesh
    loss, row = face_objective(uid, epoch)
    ((loss + edge_constants[uid]) / 5).backward()
```

### 2.2 edge 常量的语义

`edge_constants[uid]` 在训练前一次性算好（`torch.no_grad()` 下），是 **Python float**：

```python
el, _ = hard4_chunks(edge_logits, rows['edge'], data['pairs'], data['edge_labels'], 32768)
edge_constants[uid] = float(el)
```

由于 edge 分支冻结，这一项**对可训练参数的梯度恒为 0**。

> **它的唯一作用是让记录的 loss 数值与原配方逐位一致。** 实际梯度路径 = face Hard4 only。

复现时若不需要 loss 值保真，可以省略这一项（不影响训练动力学）。

### 2.3 Hard4

原始实现（`data_objective.py:126-142`）：

```python
def hard4_sums(logits, labels):
    positive, truth = logits.detach()>0, labels == 1
    masks = (truth & positive,      # TP
             ~truth & ~positive,    # TN
             ~truth & positive,     # FP
             truth & ~positive)     # FN
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction='none')
    return torch.stack([bce[m].sum(dtype=torch.float32) for m in masks]), \
           torch.stack([m.sum() for m in masks])

def hard4_chunks(logits_fn, embedding, ids, labels, chunk):
    numerator = embedding.new_zeros(4)
    counts = torch.zeros(4, dtype=torch.long, device=embedding.device)
    for start in range(0, len(ids), chunk):
        logits = logits_fn(embedding, ids[start:start+chunk])
        n, c = hard4_sums(logits, labels[start:start+chunk])
        numerator, counts = numerator + n, counts + c
    loss = (numerator / counts.clamp_min(1)).sum() / 4
    return loss, {...}
```

**数学形式：**

$$L_{\text{Hard4}} = \frac{1}{4} \sum_{g \in \{TP,TN,FP,FN\}} \frac{1}{|g|} \sum_{i \in g} \mathrm{BCE}(z_i, y_i)$$

### 2.4 三个必须注意的语义点

| # | 语义 | 说明 |
|---|---|---|
| 1 | **分组用 `logits.detach()`** | 硬指派，梯度**不穿过分组边界**。BCE 本身可导。这是"Hard"的来源 |
| 2 | **每组等权 1/4，与组大小无关** | 本方法最具破坏性的设计。见 `FAILURE_ANALYSIS.md` |
| 3 | **`counts.clamp_min(1)`** | 空组不除零，贡献 0 |

### 2.5 chunk 化

候选集可能很大（pair 池 235,741；face 三元组数千）。按 `chunk` 分块累加：

- `pair_chunk = 32768`
- `face_chunk = 32768`

分块**不改变数学**（组内求和/计数都是可结合运算），但需要 `float32` 累加以匹配原实现的数值。

---

## 3. 分类器（解析式）

**这是本方法最容易被忽略的部分：分类器不是学习出来的 head，是固定公式。**

### 3.1 edge

```python
def edge_logits(embedding, pairs):
    delta = embedding[pairs[:,0]] - embedding[pairs[:,1]]
    space, time = delta.chunk(2, dim=-1)
    return (space.square().sum(-1) - time.square().sum(-1)) * EDGE_SCALE
```

$$\text{logit}_{\text{edge}}(i,j) = \text{EDGE\_SCALE} \cdot \left(\|\Delta_{ij}\|^2_{\text{space}} - \|\Delta_{ij}\|^2_{\text{time}}\right)$$

$$\text{EDGE\_SCALE} = 0.9306077080970389$$

### 3.2 face

```python
def face_logits(embedding, triples):
    first, second, third = [embedding[triples[:,i]] for i in range(3)]
    areas = []
    for a,b,c in zip(first.chunk(2,-1), second.chunk(2,-1), third.chunk(2,-1)):
        u, v = b-a, c-a
        areas.append((u.square().sum(-1)*v.square().sum(-1)
                      - (u*v).sum(-1).square()).clamp_min(0))
    return FACE_SCALE*(FACE_FACTOR*(areas[0]-areas[1]))
```

### 3.3 推导：为什么这是"面积平方差"

`u.square().sum(-1)*v.square().sum(-1) - (u*v).sum(-1).square()` 展开即：

$$\|u\|^2\|v\|^2 - (u \cdot v)^2$$

这是 **Gram 行列式**，由 Lagrange 恒等式等于 $\|u \times v\|^2$。

> 注意：$\|u \times v\|$ 在维度 ≠ 2,3 时无独立定义，但 Gram 行列式在任何维度都有定义，
> 且恒等于所有坐标对叉积平方和。**本包的 `verify_math.py` 用 496 个坐标对验证了这一点（n=32）。**

而 $\|u \times v\| = 2 \cdot \text{Area}$，所以：

$$\text{areas}[k] = (2 \cdot \text{Area}_k)^2$$

乘 `FACE_FACTOR = 0.25`：

$$0.25 \cdot (2A)^2 = A^2$$

最终：

$$\text{logit}_{\text{face}}(i,j,k) = \text{FACE\_SCALE} \cdot \left(\text{Area}^2_{\text{space}} - \text{Area}^2_{\text{time}}\right)$$

$$\text{FACE\_SCALE} = 0.39804385828730726$$

### 3.4 两个分类器的共同结构

```
32 维嵌入 ──切两半──► 前 16 维 ("space")  vs  后 16 维 ("time")
                              ↓
              比较同一几何量（长度² 或 面积²）在两半上的取值
                              ↓
                    差值 × 常数  =  logit
```

**所以整个训练任务被压缩成：**

> 学一个 1024→32 的线性映射，使真实面在"space"半空间的"面积²"大于"time"半空间；
> 非真实面则相反。

### 3.5 关键缺陷：没有 bias

**`logit_face` 没有偏置项，阈值被硬钉在 0**（即 `Area_space == Area_time` 处）。
而训练只能改 1024→32 的投影，**无法独立调节这个阈值**。

模型想"少判正"时，唯一手段是整体缩放/旋转嵌入空间 —— 会连带影响所有样本。
这是实测中 FP↓/FN↑ 单侧漂移的**结构性成因**。详见 `FAILURE_ANALYSIS.md` §3。

### 3.6 去均值

嵌入在两处做了中心化，**这是设计，不是冗余**：

```python
# 网络前向出口
edge = e - e.mean(0, keepdim=True)
face = f - f.mean(0, keepdim=True)

# 训练路径（face_objective 内，线性层之后再做一次）
z = model.face_embedding(h)
z = z - z.mean(0, keepdim=True)
```

---

## 4. 可训练参数与冻结

```python
original_names = [n for n,p in model.named_parameters() if p.requires_grad]
assert original_names == cp['config']['optimizer_parameter_names']   # 与父协议一致
params = dict(model.named_parameters())
opt = torch.optim.AdamW([params[n] for n in original_names], ...)
opt.load_state_dict(copy.deepcopy(cp['optimizer']))
equal(opt.state_dict(), cp['optimizer'])                             # 逐字段校验

model.requires_grad_(False)
model.face_embedding.requires_grad_(True)
model.train()
assert [n for n,p in model.named_parameters() if p.requires_grad] == FACE_NAMES
```

```
FACE_NAMES = ['face_embedding.weight', 'face_embedding.bias']

face_embedding: Linear(1024, 32)  →  32×1024 + 32 = 32,800 个参数
```

### 4.1 冻结的双重校验

每 50 步断言一次：

```python
assert frozen_hash(model) == frozen                          # 参数值未变
assert adam_frozen_hash(opt, original_names) == frozen_adam  # Adam 槽未变
```

`frozen_hash` 对**除 face 外的所有参数张量**做顺序敏感的 SHA256；
`adam_frozen_hash` 对**除 face 外的所有 Adam 槽**同样处理。

---

## 5. 优化器

```python
torch.optim.AdamW([params[n] for n in original_names],
                  lr=1e-4, betas=(.9,.999), eps=1e-8,
                  weight_decay=.01, foreach=True)
```

| 项 | 值 |
|---|---|
| 算法 | AdamW |
| lr | `1e-4` |
| betas | `(0.9, 0.999)` |
| eps | `1e-8` |
| weight_decay | `0.01` |
| foreach | `True` |
| 状态 | **继承父 checkpoint，不重置** |

### 5.1 关键点：优化器建在 412 个张量上，只训 2 个

父代 `param_groups["params"] = [0..411]` —— 412 个可训练张量。

本方法：

1. 在这 **412 个张量**上构建 AdamW
2. 载入父代的完整 Adam 状态
3. 冻结其中 410 个（`requires_grad_(False)`）

`opt.step()` 对 `grad is None` 的参数直接跳过 → **冻结项状态完整保留、永不推进**。

这与"只建一个 2 参数的优化器"语义不同：后者会丢失父代的 Adam 动量映射。

### 5.2 Adam step 计数器对齐

```python
assert [int(opt.state[p]['step']) for p in active] == [step, step]
```

其中 `step = 19356 + new_update`。

**这一步确认了本次微调被视为父训练的延续**，Adam 的 `step` 与全局更新序号严格对齐 ——
这对 Adam 的偏差修正（bias correction）至关重要，否则前几步的等效学习率会严重失准。

---

## 6. 数据调度

### 6.1 种子配方（无状态）

```python
def seed_for(kind, epoch, uid='', seed=0):
    text = f'own512-v2-v1/{kind}/{seed}/{epoch}/{uid}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')

def epoch_batches(uids, epoch, seed=0):
    ordered = np.asarray(uids)[np.random.default_rng(seed_for('order', epoch, seed=seed))
                               .permutation(len(uids))].tolist()
    return [ordered[i:i+5] for i in range(0, 50, 5)]
```

| kind | 配方串 |
|---|---|
| 批顺序 | `own512-v2-v1/order/0/{epoch}/` |
| 负样本 | `own512-v2-v1/negative/0/{epoch}/{uid}` |

**设计属性**：所有随机决策由 SHA256 派生，**不依赖环境 RNG 状态**。
因此任意一个更新都能从 `(kind, epoch, uid)` 单独复现 —— 这是断点续训能保证位级一致的前提。

### 6.2 更新 → 批次映射

```python
step = 19356 + new
epoch, batch_index = divmod(step - 1, 10)
batch = epoch_batches(uids, epoch)[batch_index]
```

| 项 | 值 |
|---|---|
| 数据集 | 50 个 mesh（`teacher_cad50_00` .. `_49`） |
| batch | 5 个完整 mesh |
| 每 epoch 更新数 | 10 |
| **1 epoch = 10 次更新** | |

### 6.3 ⚠ 续训从 epoch 中途开始

第一次新更新：`step = 19357` → `divmod(19356, 10) = (1935, 6)`

**`batch_index = 6`，不是 0。** 续训从父训练的中途批次接续，而不是新 epoch 起点。

这个事实有一个直接后果（第一版因此崩溃）：

50 次更新覆盖：

```
  epoch 1935: batch 6,7,8,9        →  4 次更新（20 个 UID 各 +1）
  epoch 1936..1939: 4 个完整 epoch → 40 次更新（50 个 UID 各 +4）
  epoch 1940: batch 0..5           →  6 次更新（30 个 UID 各 +1）
                                     ─────────
                                     50 次更新 = 250 次 mesh 访问
```

每个 UID 的访问次数 = `4 + (0 或 1) + (0 或 1)` ∈ **{4, 5, 6}**，而 `new//10 = 5`。

**所以不能断言 `set(participation.values()) == {new//10}`。** 正确写法（修复版）：

```python
assert sum(new_participation.values()) == new*5
assert min(new_participation.values()) >= new//10 - 1
assert max(new_participation.values()) <= new//10 + 1
```

### 6.4 参与度账本

每个 mesh 的累计访问次数单独记账（`participation` / `new_participation`），
写入 checkpoint，并每 50 步校验。实测 500 步后每个 UID 落在 49~51（期望 50）。

---

## 7. 负样本采样

负样本集 = **无状态随机负样本 ∪ 固定 hard negative**，去重。

### 7.1 来源一：无状态随机负样本

```python
def negative_faces(item, epoch, seed=0):
    n = len(item['vertices'])
    gt = {tuple(row) for row in item['gt_faces'].cpu().tolist()}
    available = math.comb(n,3) - len(gt)
    wanted = min(math.ceil(1.5*len(gt)), available)          # ← 1.5 倍正样本
    rng = np.random.default_rng(seed_for('negative', epoch, item['uid'], seed))
    if wanted == available:
        chosen = {x for x in itertools.combinations(range(n),3) if x not in gt}
    else:
        chosen = set()
        while len(chosen) < wanted:
            draws = np.sort(rng.integers(n, size=(max(128,4*(wanted-len(chosen))),3)), axis=1)
            for row in draws:
                key = tuple(int(x) for x in row)
                if key[0] < key[1] < key[2] and key not in gt:
                    chosen.add(key)
                    if len(chosen) == wanted: break
    result = np.asarray(sorted(chosen), dtype=np.int64).reshape(-1,3)
    assert len(result) == wanted
    return torch.from_numpy(result), array_sha(result)
```

| 项 | 规则 |
|---|---|
| 数量 | `ceil(1.5 × |gt_faces|)`，上限为全部非 GT 三元组 |
| 采样 | 拒绝采样（排序去重、剔除 GT、要求严格递增） |
| 刷新 | **每 epoch 重新生成**（种子含 epoch 与 uid） |
| 来源 | **不复用父代的 pool 负样本**（manifest 明确记录） |

父代 manifest 中的原话：

```
training_negatives = 'fresh stateless uniform non-GT triples; old pool negatives not used'
```

### 7.2 来源二：固定 hard negative（47 个）

```python
for uid, ref in zip(manifest['uids'], reference['meshes']):
    with np.load(OLD/ref['prediction_path']) as a:
        negatives = a['actual_face_candidate_ids'][~a['actual_face_candidate_gt']].astype(np.int64)
        assert len(negatives) == ref['face']['fp']
    hard[uid] = torch.from_numpy(negatives)
assert sum(x['count'] for x in hard_records) == 47
```

从父模型的**预测文件**里取出被误判为面的候选三元组（= 47 个 face FP），
作为**永久 hard negative** 钉在训练集里。

```python
write(OUT/'HARD_NEGATIVES.json', dict(total=47, gt_positives=5576,
      records=hard_records, refresh=False))
```

**`refresh=False` —— 全程不刷新。**

### 7.3 合并

```python
merged = torch.unique(torch.cat((neg, hard[uid])), sorted=True, dim=0)
```

### 7.4 正负样本配比

| 类别 | 数量 |
|---|---|
| 正样本 | `|gt_faces|`（全体 5,576，每 mesh 平均 ~112） |
| 随机负样本 | `ceil(1.5 × |gt_faces|)`（每 mesh ~168） |
| 固定 hard negative | 47（全体，不平均分布） |

---

## 8. 缓存机制

### 8.1 缓存边界

**缓存 `decoder_output_norm` 的输出 `decoder_hidden`。**

因为解码器（含 `decoder_output_norm`）冻结，整个 batch 只需前向一次即可复用。

```python
with torch.no_grad():
    rows = model(data['vertices'], data['faces'], sample_latent=False)
    cache[uid] = rows['decoder_hidden'].detach()      # ← 只缓存冻结路径的输出
```

**原则：绝不缓存任何可训练参数下游的输出。** 协议字段原话：

```
cache_boundary = 'frozen decoder_output_norm output; no trainable output cached'
```

### 8.2 缓存正确性审计（强制门槛）

训练前必须在**最大网格**上验证缓存路径与真实前向路径**位级一致**：

```python
check_uid = max(manifest['uids'], key=lambda u: len(items[u]['vertices']))

l, _ = face_objective(check_uid, 1935, real=True)      # 真实前向
g     = torch.autograd.grad(l, active)

c, _ = face_objective(check_uid, 1935)                 # 缓存路径
cg    = torch.autograd.grad(c, active)

assert torch.equal(l, c)                                       # 损失位级相同
assert all(torch.equal(x, y) for x, y in zip(g, cg))           # 梯度位级相同
```

**不通过就不训练。** 审计通过后写 `CACHE_AUDIT.json`（`passed=True`）。

这是本仓库工程质量最高的部分 —— 缓存复用最容易引入静默数值偏差，用位级比对把它堵死。

---

## 9. 梯度处理

```python
assert all(p.grad is None for n,p in model.named_parameters() if n not in FACE_NAMES)
norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True, foreach=True))
opt.step()
```

| 项 | 值 |
|---|---|
| `zero_grad` | `set_to_none=True`，每次更新开头 |
| 冻结断言 | 更新前断言非 face 参数无梯度 |
| 裁剪 | `clip_grad_norm_(active, 1.0)` |
| 非有限值 | `error_if_nonfinite=True`（NaN/Inf 直接抛错） |
| 更新后断言 | `all(torch.isfinite(p).all() for p in active)` |

### 9.1 每步记录的诊断量

| 字段 | 含义 |
|---|---|
| `gradient_norm_before_clip` | 裁剪前梯度范数 |
| `clip_coefficient` | `min(1, 1/(norm+1e-6))` |
| `parameter_displacement_l2` | 更新前后参数的 L2 位移 |
| `lr` | 当前学习率 |
| `adam_face_step` | Adam step 计数器（= 全局 update） |

时序约定：`timing = 'loss before update; parameter displacement and Adam step after update'`

---

## 10. 检查点与评估协议

| 项 | 规则 |
|---|---|
| 检查点频率 | 每 **50** 步 |
| 评估频率 | 每 50 步，**完整 50 网格原生前向** |
| 达标判据 | `face micro-F1 >= 0.997` |
| 达标后 | 再跑一次 **cold 评估**（全新进程）验证位级一致，然后停 |
| 硬上限 | **500** 步，绝不延长（README 原话：`Never extend this budget`） |

### 10.1 评估在独立子进程

```python
before = rng_state()
rc = subprocess.run(argv, cwd=ROOT, ...).returncode
assert rc == 0
equal(before, rng_state())        # 评估不得扰动父进程 RNG
```

评估还断言：`optimizer_updates_added_by_evaluation = 0`（评估不推进优化器）。

### 10.2 评估内容

对每个 mesh：

1. 完整前向 → edge / face 嵌入
2. edge：全部 `pairs` 打分，`logit > 0` 判正 → TP/FP/FN/TN
3. face：**由预测边构成的团（clique）枚举候选三元组**（不是全枚举 C(n,3)）
4. face 打分 → TP/FP/FN/TN
5. 存 `.npz`：`vertices / gt_edges / gt_faces / all_edge_pair_logits / predicted_edge_ids / actual_face_candidate_ids / ...`

**face 的候选集来自预测图的团枚举** —— 这是评估与训练负采样语义不同的地方（训练负样本是随机+固定，评估候选是预测团）。

### 10.3 跨评估一致性断言

冷/暖两次评估必须**位级一致**：

```python
assert result['counts'] == previous['counts']
assert result['perfect_uids'] == previous['perfect_uids']
for a, b in zip(earlier_meshes, rows):
    with np.load(...) as x, np.load(...) as y:
        assert x.files == y.files
        for k in x.files:
            assert x[k].dtype == y[k].dtype and x[k].tobytes() == y[k].tobytes()
```

---

## 11. 可复现性配置

**这不是装饰** —— 上文的缓存位级审计、冷评估位级一致，都依赖这些开关。

```python
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k] = '1'

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.mha.set_fastpath_enabled(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)          # 强制 MATH backend
torch.set_float32_matmul_precision('highest')
```

外加：

- **注意力手写实现**，绕开 `nn.MultiheadAttention` 的融合路径
  （手写 QKV 投影 → reshape → SDPA(MATH) → 输出投影）
- **确定性邻域聚合**：`DeterministicGather`（backward 用 `index_add_` 替代 CUDA atomics）
- **激活重计算**：`torch.utils.checkpoint`，`preserve_rng_state=False`
- RNG 状态完整存取（python / numpy / torch / cuda 四路）

---

## 12. 超参全表

```yaml
# ===== 起点 =====
parent_completed_updates: 19356
parent_sha256: 6f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2

# ===== 可训练 =====
trainable: [face_embedding.weight, face_embedding.bias]
trainable_elements: 32800              # 32*1024 + 32
frozen_everything_else: true
parent_trainable_tensor_count: 412     # 优化器建在全部 412 个上

# ===== 优化器 =====
optimizer: AdamW
lr: 1.0e-4
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.01
foreach: true
state: inherited_from_parent           # 不重置
adam_step_counter: "== global update index (19356 + new)"
grad_clip: 1.0
grad_clip_error_if_nonfinite: true

# ===== 数据 =====
dataset_size: 50                       # teacher_cad50_00 .. _49
batch_size: 5                          # 完整 mesh
updates_per_epoch: 10
epoch_seed_recipe: "own512-v2-v1/order/0/{epoch}/"
negative_seed_recipe: "own512-v2-v1/negative/0/{epoch}/{uid}"

# ===== 负样本 =====
random_negatives: "ceil(1.5 * |gt_faces|), 无状态拒绝采样, 每 epoch 重抽"
fixed_hard_negatives: 47               # 父模型 face FP，refresh=False
negative_union: "torch.unique(random ∪ hard, sorted)"

# ===== 目标函数 =====
loss: Hard4
groups: [tp, tn, fp, fn]               # detached sign(logits) × label
group_weighting: "每组等权 1/4（组均值再平均）"
group_assignment_detached: true        # 梯度不穿过分组边界
edge_term: "常量 float，梯度为 0，仅为 loss 数值保真"
per_mesh_scale: 1/5
pair_chunk: 32768
face_chunk: 32768

# ===== 分类器常数（固定，不训练）=====
EDGE_SCALE: 0.9306077080970389
FACE_SCALE: 0.39804385828730726
FACE_FACTOR: 0.25
embedding_split: "32 -> 前16 'space' / 后16 'time'"
face_classifier_bias: none             # ← 缺陷点，见 FAILURE_ANALYSIS.md

# ===== 缓 存 =====
cache_boundary: "decoder_output_norm 输出 (decoder_hidden)"
cache_trainable_output: false
cache_audit: "最大网格上 loss 与 face 梯度位级一致，否则拒绝训练"

# ===== 预算与协议 =====
max_new_updates: 500
checkpoint_every: 50
eval_every: 50
eval_scope: "全部 50 个 mesh，原生完整前向，独立子进程"
target_face_micro_f1: 0.997
stop_on_target: true
stop_on_target_extra: "cold 复验（全新进程）位级一致"
frozen_edge_strict_ceiling: 40
participation_assert: "min >= new//10-1, max <= new//10+1"

# ===== 可复现性 =====
deterministic_algorithms: true
tf32: false
sdp_backend: MATH
matmul_precision: highest
CUBLAS_WORKSPACE_CONFIG: ":4096:8"
threads: 1
```

---

## 附：数据校验和（硬编码于原实现）

```
DATA_SHA = 4742e72bde899b88633cb70603a256d98b83b7bd85cd301b7a090938081a2ac6
POOL_SHA = 86e7895f7ce44f7740d96a359570a3617c8967414c69aaf38dd70c02a177c61b
```

规模断言：

```
meshes=50, vertices=2872, edges=8364, faces=5576, pairs=235741
```

UID 断言：

```python
assert source['uids'] == [f'teacher_cad50_{i:02d}' for i in range(50)]
```
