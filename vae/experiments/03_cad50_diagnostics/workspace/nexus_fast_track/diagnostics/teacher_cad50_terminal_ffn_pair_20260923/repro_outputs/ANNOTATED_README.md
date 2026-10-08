# 启动README注释副本

原件与启动哈希保留。本副本更正CPU测试措辞；最终状态见 REPORT.md 和 status.json。

# 原512 CAD50：单个末端FFN的有限预算对照

本实验由用户于2026-09-23明确要求启动，用老师小AE中的逐点非线性变换作为参照，检验原512 AE的Decoder末端增加一个FFN是否改善实际重建。不是重新训练老师整套小网络，不修改原固定100条任务。

## 唯一干预和共同起点

- Control：原512网络，无新增模块。优先复用已完成、经本轮重新核验的 `teacher_cad50_soft4_stopgrad_pair_20260922/A_soft4_full`，不重复追加预算。
- Treatment：在最后Decoder attention残差后、原 `decoder_output_norm` 前增加一个pre-LN逐点FFN残差：1024→4096→GELU→1024，dropout=0，末Linear权重和bias初始化为0。不是扩建第17个attention block。
- 两支共同父：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt`；SHA256 `4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`。
- 旧参数完整继承model/Adam/RNG/进度和有序参数映射；新增FFN单独fresh Adam state，LR=3e-5。FFN随机初始化后恢复父RNG；预检不推进任何optimizer。

## 锁定协议

保留原fully-differentiable Soft4，membership、分子和分母全部参与反传；固定50 UID及原Face pool。每条完整mesh的Edge+Face loss除50，全部50条累积后global clip=1，再Adam一次。新增FFN也参与同一个clip；因此即使旧参数裁剪前梯度相同，新增梯度可能改变clip系数，不能预设旧参数首步位移相同。

Encoder/μ LR3e-6，Decoder、Edge head、Face head及新FFN LR3e-5；Adam betas=(0.9,0.999)、eps=1e-8、weight_decay=0；不重跑warmup或加调度器。完整Encoder/μ/16层Decoder/两head继续训练，logvar冻结，确定性μ、sampling关闭、KL=0，math00/FP32 SDPA MATH，TF32和autocast关闭，原eval-with-grad模式及recompute上下文。

数据2872顶点、8364 GT边、5576 GT面、235741个全部pair；Face基础pool13946。不得变坐标、拓扑、数据顺序、pool、阈值或scale。

## 固定预算与验收

沿用该B2500父点此前短对照预算：各分支100次全50条更新，累计2600；复用Control时本轮Control新增0、Treatment新增100。0/25/50/75/100执行真实Encoder→μ→Decoder全50验收，阈值logit>0，实际Face完整枚举当前预测Edge图三角候选，缺GT候选计FN。

启动前：父文件身份、旧Adam/RNG/参数映射、全50初始预测、原参数裁剪前梯度、FFN零初始化及梯度路径必须通过。CPU小测试仅对合成toy模型做2次Adam更新，不更新真实实验参数。H写ready后等待根代理核对并写release；检测非有限数或协议错误保存现场停止，不回滚或调参。

保存完整model/Adam/RNG checkpoint、逐步loss/每UID Edge错误/梯度norm/clip/LR/Adam步数/实际位移，以及逐检查点预测、实际Face错误分解、严格UID与困难16条汇总。FaceF1达到0.997与严格50/50分开记录。100步是方向试验，不是充分收敛或全架构上限证明。

## 执行命令

以下为本轮新增实验命令；是否实际执行以 `repro_outputs/COMMANDS.md` 与生命周期日志为准。

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python audit_control/audit_reused_control.py
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python H_terminal_ffn/test_terminal_ffn.py
/opt/conda/bin/python launch_h.py
```

run-train实际子命令为：

```bash
/opt/conda/bin/python -u H_terminal_ffn/train.py --mode terminal_ffn
```

## 资源与交付

两台为用户本轮分配资源。现场核查31548有1张、32483有2张A100-SXM4-80GB，均空闲、PyTorch/CUDA相同。Treatment分配32483的GPU0，UUID `GPU-0ae7719a-74e5-08ae-75de-0a6205381365`；31548并行做CPU控制证据核验。启动前再次检查占用，不抢占或终止其他任务。不为占满卡重复训练合格Control。

所有变更仅在本独立git实验目录，分支 `repro/2026-09-23-terminal-ffn`。使用已安装ai-research-reproduction的README intake、run-train生命周期和证据格式；结构干预来自用户明确授权，不冒称原论文复现修复。

完成即停止，提供老师原ZIP逆向工程报告、原版结构对照、有效改动/diff、配置与命令、完整实验轨迹和评价、父/末尾权重身份。大型权重留服务器，报告路径/大小/SHA256；不打包凭据。
