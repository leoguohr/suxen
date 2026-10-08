<!-- rigorpilot:repro:begin kind="banner" section="__banner__" occurrence="1" status="success" risk="none" -->

# 📄 README · RigorPilot 复现批注

🟢 `success` · `training` · `trusted` · [SUMMARY](SUMMARY.md) · [COMMANDS](COMMANDS.md) · [LOG](LOG.md) · [status.json](status.json) · [train status](../train_outputs/status.json)

章节覆盖：🟢 1 · ⚪ 4（共 5 节） · 复现得分 1.0

<sub>🟢 成功 · 🔵 未执行 · ⚪ 仅阅读 · 🟡 部分完成 / 资产缺失 · 🔴 阻塞 · 🟣 待决策 —— 原文未改动；相对媒体链接需要原 README 所在目录的上下文。</sub>

<sub>original_sha256: `0a1702dc9a733833a875f4ff574695f54bda58e5e5a218c3e5385afe2cf2aab3` · round-trip: verified</sub>

---

<!-- rigorpilot:repro:end -->
# CAD50：B2500起点的Soft4与论文式Hard4，100步对照

本目录只处理老师50条原始CAD目标上的512维AE；不修改原固定100条或旧Soft4/stopgrad实验。共同父checkpoint为 `../teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt`，SHA256 `4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`。

<!-- rigorpilot:repro:begin kind="section" section="CAD50：B2500起点的Soft4与论文式Hard4，100步对照" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 已锁定协议

S复用 `../teacher_cad50_soft4_stopgrad_pair_20260922/A_soft4_full` 已完成的100步，独立身份核验见 `repro_outputs/CONTROL_REUSE_AUDIT.json`。H从同一B2500完整model/Adam/RNG恢复，新增100次全50条累积更新，累计2600后停止。S本轮新增更新为0；预算比较为两者均从B2500新增100步。

H只把Edge与Face的重建loss改为TP/TN/FP/FN硬四组BCE；分组使用 `logit.detach()>0`，BCE使用原始可微logits。整条mesh跨chunk累计组内BCE总和与候选数，再除 `max(count,1)`，四组之和固定除4。空组0是本轮约定，不冒称作者官方源码。用户提及的paper_hard4_loss.py附件未在本会话可访问附件中找到；这里直接实现用户完整给定公式，未假称已读取附件。

S原fully-differentiable Soft4包含membership、分子和分母梯度。两种loss不按数值排名。

保持四组LR为3e-6/3e-5/3e-5/3e-5，Adam历史、betas/eps/weight_decay、global clip1、原固定数据/pool、全部Encoder/μ/16块Decoder/共享head、latent512、math00 FP32 SDPA MATH、TF32/autocast关闭、确定性Graph、原recompute、μ路径、sampling关闭、KL0、logvar冻结。每条loss除50，50条累积后一次clip/Adam，无额外外层0.25。

<!-- rigorpilot:repro:begin kind="section" section="已锁定协议" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 命令与资源

```sh
/opt/conda/bin/python preflight_cpu.py
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python H_paper_hard4/test_hard4.py
/opt/conda/bin/python launch_h.py
```

当前服务器仅使用用户确认可用的GPU1，UUID `GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351`。启动器会再次检查占用；不终止其他进程。S复用旧结果，不为占满另一张卡而重训。启动器绑定本次设备身份，不应在其他机器盲目复用。

<!-- rigorpilot:repro:begin kind="section" section="命令与资源" occurrence="1" status="success" risk="low" -->

> [!TIP]
> 🟢 **执行成功（低风险）**
> 命令：`/opt/conda/bin/python launch_h.py`
> 结果比较：未提供显式期望指标，因此尚未评估是否匹配。
> 完成步数：100
> 命令来自 README 链接的文档 `README.md`。
> <sub>证据: [SUMMARY](SUMMARY.md) · [COMMANDS](COMMANDS.md) · [LOG](LOG.md) · [status.json](status.json) · [train status](../train_outputs/status.json) · tier: execution</sub>

<!-- rigorpilot:repro:end -->
## 验收与证据

0/25/50/75/100新增步固定checkpoint，全部50条经真实Encoder→μ→Decoder；从预测Edge完整枚举Face候选，判正logit>0，漏候选GT记FN。每步loss和Edge错误是更新前指标；clip、Adam计数、参数位移是更新后记录，不当作逐步实际Face评价。

起点须匹配Edge FP/FN5159/1656、Face5359/3255、Face F1=0.350181050090525、联合32/50及UID集合；新卡另核对原Soft4完整梯度与旧S记录逐位相同。检查后恢复父RNG和原模型模式，新增optimizer更新0，再开始H预算。

报告分别判断实际Face F1≥0.997与严格50/50。100步只是继承Adam的loss切换方向测试；不能外推随机初始化训练上限，也不能将Hard4收益单独归因于删除Soft4第二项。

<!-- rigorpilot:repro:begin kind="section" section="验收与证据" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 交付与停止

`repro_outputs/`保存命令、原始日志、配置、有效diff、复用审计、loss小测试、完整评价、逐UID与困难16条指标、checkpoint路径/大小/SHA256。有效代码与预测打包下载；完整model/Adam/RNG保留服务器，不重复打包全部历史模型。预算后停止，无额外训练。
<!-- rigorpilot:repro:begin kind="section" section="交付与停止" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
