# Soft4 控制复用与末端 FFN 启动前审阅

审阅者：GPT-6 Astra，xhigh；2026-09-23。范围：已保存控制的 CPU 证据复核、H 代码只读审阅。此审计没有模型 forward、GPU 初始化或 optimizer update。

结论：旧 Soft4 S100 的现存证据通过本轮 fresh CPU 复核，可用于本轮相同 B2500 父状态的 100 步结构对照。新 H 的实际启动仍须通过代码中 GPU 数值门禁；CPU 审计不替代该门禁。H 源码本次未发现阻断问题。

## 已执行的控制复核

结果：`../repro_outputs/CONTROL_REUSE_AUDIT.json`，日志 `../repro_outputs/control_reuse_audit.log`，两端共享远程根 `/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923/`。本地相同相对路径亦保存结果。

- B2500 父 SHA256：`4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`；父与旧 S0000 的完整 model、Adam、四类 RNG、participation、completed_updates 精确相等；328 个旧可训练参数按组、名称、Adam ID、形状核对。
- 100 条更新日志均为按固定 UID 顺序参与全部 50 条 mesh，累计 2501–2600 步；原 LR、betas、eps、weight decay、clip、Soft4/full、μ 路径、冻结 logvar、数据和池不变。
- 0/25/50/75/100 共 5 个完整 checkpoint SHA 与旧归档相同，Adam 步数/组、进度、冻结 logvar 均核对。
- 250 份 NPZ SHA 已逐一复核并重数 Edge/Face TP/FP/FN。候选三角形均由预测 Edge 构成；去重后候选数量等于完整图枚举数量，因此无 GT 注入或漏枚举。GT 不入候选也计 Face FN。0 步全 50 份保存数组与父数组相同。
- 50 条数据、50 个固定 pool、控制脚本、18 个实际依赖源码及各自 source_archive 均与旧归档哈希一致。Python、torch、CUDA、cuDNN 版本相同；此结论不表示物理 GPU 相同。
- 旧 `ready.json` 已嵌入结果并保存 SHA `1fb87530042a8bc7bfaddfbc19432b6ba0440ffbd01116112a97fea9ade82175`。UID20 全网梯度哈希 `d445214cae052baa2d8979b29314451971110649dd876f6409f2d12e32f47955`，loss `0.84434574842453`，裁剪前梯度 norm `6.520879316263456`。新卡同权重核验应匹配此记录。

最初完整审计通过后，为增加 launch 所需 ready 归档字段又执行一次完整 CPU 审计，两次均通过；无需第三次重复。

## H 源码审阅

以下行号均相对 `../H_terminal_ffn/`。

| 核查点 | 代码与结论 |
|---|---|
| 唯一新增结构 | `terminal_ffn.py:10–41`：最后 attention residual 后、原 terminal LN 前，新增 `x + W2 GELU(W1 LN(x))`；1024→4096→1024，dropout=0；W2 与 b2 零初始化，共 8,395,776 新参数。以命名 module + prehook 接入，旧源码不修改。 |
| 原 Adam 继承 | `train.py:11–33, 51–80`：先严格加载原 model/optimizer；旧四组顺序、组属性、按名称参数与每个 Adam state 精确核对；再添加 FFN 新组，LR=3e-5、state 为空。 |
| RNG | `train.py:68–80, 155–160`：FFN 构造消耗 RNG 后恢复父 Python/NumPy/torch/CUDA 状态；评估 helper 同时核对 RNG 未改变。 |
| 初始输出与旧梯度 | `helpers.py:117–128` 对全 50 个父预测 NPZ 的所有数组做值、dtype、字节比较；`train.py:60–66, 136–160` 比较同卡原网与扩展网 UID20 的所有旧参数**裁剪前**梯度。不是全 50 mesh 总梯度的逐位检查。跨卡与旧 S 的一致性由根启动门禁比较 ready 记录。 |
| 新 FFN 梯度 | `train.py:147–154, 182–188`：第 0/1 步输出层 weight 有非零梯度，W2=0 时前层与新 LN 梯度为零；第二步要求前层与新 LN weight 非零。此处为运行时断言，审阅本身不宣称已经实测通过。 |
| 全局裁剪 | `train.py:122, 181–191`：`active` 包括全部旧参数及 FFN；所有参数共同 `clip_grad_norm_(active,1)`。新增末层首步梯度会改变全局 clip 系数，因此不要求旧参数首步裁剪后梯度/位移等于 S。这是本次干预的一部分。 |
| 损失与预算 | `runtime.py` 与 baseline_source 字节相同；`train.py:172–215` 仅 fully differentiable Soft4，每步 50 条各 `loss/50` 后单次 Adam，共 100 次；旧组 step=2500+t，新组 step=t；检查点仍为 0/25/50/75/100。 |
| 实际 Face | `evaluate.py:5,15–53`：全部 pair、logit>0 预测 Edge、完整 triangle 枚举；GT 的覆盖检查仅用于 FN 统计，不注入候选。`FACE_SECONDS=inf`，保存预测前必须 complete。评价代码与 baseline_source 字节相同。 |
| 持久化 | 命名 FFN 参数进入 state_dict；`terminal_ffn.py:45–53` 的 `load_extended_state` 同时恢复模块、执行 prehook、strict 权重。后续冷载必须使用该扩展加载入口。 |

本次结构干预包含新增参数、fresh Adam 状态与全局裁剪的相应变化；若改善，只支持这一末端非线性分支在指定父状态与预算下有效，不能单独证明旧网全部问题来自无 FFN。未并改 Fourier、Graph、loss、pool 或评分形式。

审阅源码 SHA256：

```
train.py                     2a094ab1b4d9eaafb971a2c6cce0e393a671864df435573ac94498c4125dbd91
terminal_ffn.py              61f6a7f7d9159d382f15880c3a4f885793d85d7d647b338c5b5ba47c1ff07be7
helpers.py                   c7d0bfa9a4c782db13080478116c685edce3ca2a5bcbb45eea72b307aba99496
runtime.py                   3779f891288f5779ede01eb86726c801d8df73bd7c535e684f257936bd7dd108
evaluate.py                  4bfad01b82d946d478d53c9b0f9b79f647b5c5c8b5bf43b699e0f9685b42f09d
effective_loss_and_scoring.py 1fdb8708e185e0e03504355352c0ce4d21755e3848d3c708f34ab6dea7ac5559
```

## 可重放的只读 CPU 审计

脚本 `audit_reused_control.py` 仅读取原实验；仅结果路径写入本轮新目录。不要直接 import 历史 `preflight_cpu.py` 或 `audit_results.py`，它们有会写入旧目录的顶层代码。

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python -B \
  /guohaoran/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923/audit_control/audit_reused_control.py \
  --project-root /guohaoran/nexus_fast_track \
  --output /guohaoran/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923/repro_outputs/CONTROL_REUSE_AUDIT.json
```

启动后审核、实际训练与结果解释由根任务继续，本审计不启动训练。
