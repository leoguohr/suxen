# 原100条 OwnAE-v2：当前交接

## 当前结论

**原100条V2正式训练尚未启动，正式optimizer更新为0。** 本轮只核查了服务器进程、实际数据和现有代码；没有加载模型进行GPU前后向，没有新建训练运行目录，没有修改服务器代码或checkpoint。

用户最新要求：迅速交接、只给一个压缩包、不要跑训练。此前短暂给出的4小时预算不构成继续执行指令；下一位助手也不得自动启动。

## 这条任务是什么

这是把已存在的OwnAE-v2实现接到原定100条mesh的任务。不是Vertex Diffusion、不是CAD50续训，也不是旧72/73/74条成功模型的末端微调。

已讨论的方向是原100条、V2完整模型、随机初始化、fresh AdamW。正式执行入口尚未完成迁移，不能直接运行包内旧CAD train.py。

## 已有实现的精确定义

- V2选择：B_v2_teacher_blocks。
- Encoder宽512，12个Graph+完整Transformer复合块；Fourier XYZ为6频带、39维输入。
- 每顶点latent512；Decoder宽1024、16个完整attention+FFN块，FFN宽4096，8 heads、GELU、dropout=0。
- 两个共享Linear(1024,32) Edge/Face head，逐mesh中心化。
- 明确μ路径、KL=0、logvar冻结；math00 FP32 SDPA MATH，TF32/autocast关闭、确定性Graph、activation recompute。
- **当前V2目标是Hard4，不是此前旧主线fully-diff Soft4，也不是CAD Soft4 stopgrad实验。** 由logit>0确定TP/TN/FP/FN，各组BCE均值，空组贡献0，四组固定/4；跨chunk累计分子与组计数后归约。
- Edge scale=0.9306077080970389；Face scale=0.39804385828730726；Face面积式内部factor=0.25，不能误当额外loss外层系数。
- 每epoch对每条生成1.5×GT Face数量的唯一uniform non-GT三元组，按seed/epoch/UID确定；不使用旧100条固定难负例池。
- 已有CAD训练实现：fresh AdamW，lr=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0.01，前100次更新线性warmup，clip=1；microbatch1、累积5条统一更新。
- 上述是归档源码的实际定义。不要套用旧实验Adam、Soft4、极低LR等设置。

## 当前服务器（实时核查，不含凭据）

2026-09-24 23:49（Asia/Shanghai）成功连接172.16.78.10:31548，主机6u4cbrr15dg07-0。

仅一张GPU0：NVIDIA A100-SXM4-80GB，UUID GPU-208b1847-4ab0-f090-b8d4-f21ebd908b00。抽查显存0MiB、利用率0%，无compute进程，无训练/评价进程。GPU状态会变化，下次应重新核查。

未发现原100条V2新运行；最新preflight记录fixed100_optimizer_updates=0。这个0不否定历史100条旧架构已经做过的训练。

旧CAD的部分status.json仍写training，但没有对应进程。原CAD旧目录11个评价请求缺done标记；resume目录32个实际请求均有成功done标记，另有worker完成记录。没有恢复、删除或重跑旧队列。SSH容器无squeue/atq/tmux，平台层待排任务未核实。

## 已完成的数据检查

原始清单位于服务器：
`/guohaoran/nexus_fast_track/diagnostics/shared_edge_head_fixed100_20260917/`

本轮CPU读取全部100条mesh与topology，核对200个文件SHA256，并确认面索引、FP32坐标、GT面及面导出的边与topology一致。

- 100个固定不同UID
- 顶点106325
- GT Edge309194
- GT Face204330
- 全部无向pair84669234
- 最大UID nexus_2k_000898（先前预检记录2547点）
- selection.json SHA256：6a88ce78eee5caa00e27e4d7d8d9247132e6725a44697bbb16180ca38fee14c6
- overfit100_manifest.csv SHA256：b90e14f5d318dc0d32ad806c631af7f66be28aa644ff3caf357d34ce3e208b93

实际数据路径由该CSV给出。数据本体及清单不在本包；未凭相似名称替换为其他数据。

## 未完成的必要迁移

1. data_objective.py仍是CAD50专用加载器；epoch_batches断言50、返回10个五mesh组。需要独立原100条加载入口与每epoch20组。
2. train.py的completed//10、%10等恢复游标仍是CAD定义。原100条的新调度、下一UID及负例恢复一致性尚未验证。
3. 原评价器对候选外GT Face计FN的定义正确，但会保存全部triangle/logit列表并concatenate，尚不是有界内存流式评价。需分块写盘、记录游标、支持中断续评；未完成不能称完整100条验收。
4. 最大2547点完全图上界2750577465个三角形。该数是极端上界，不是当前实际候选数。
5. 需独立主线级排他锁、不可变checkpoint SHA绑定，以及正式第1—3次和最大mesh组的五mesh累积+Adam整步资源记录。
6. 未建立新的可执行原100条训练入口，未上传任何迁移补丁，未运行新GPU预检或训练。

## 已有预检证据的范围

先前服务器preflight完成了最大mesh一次完整F/B：约2.122秒、峰值allocated约8.62GiB；没有optimizer状态或step，不是五mesh整步成本。本次没有重跑它。

这些历史记录在服务器：
`/guohaoran/nexus_fast_track/diagnostics/ownv2_face47_finish_fixed100_preflight_20260924/fixed100_preflight/`

## 文件说明

- code/：旧入口及当前模型/目标/评价代码快照。**仅供接手阅读，不是已迁移好的原100条启动包。**
- evidence/current_state.json：本次状态、实际数据核查与待办。
- provenance.json：本包源码来源及哈希。
- NEXT_CHAT_PROMPT.txt：可复制到新会话。
- SHA256SUMS：包内文件完整性清单。

本包不含密码、模型权重、Adam大文件、原始数据数组。原100条新V2没有训练结果可打包；不要把CAD或旧72/74条结果填入其成绩。
