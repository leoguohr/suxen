<!-- rigorpilot:repro:begin kind="banner" section="__banner__" occurrence="1" status="not_run" risk="none" -->

# 📄 README · RigorPilot 复现批注

🔵 `not_run` · `evaluation` · `trusted` · [SUMMARY](SUMMARY.md) · [COMMANDS](COMMANDS.md) · [LOG](LOG.md) · [status.json](status.json)

章节覆盖：🔵 1 · ⚪ 9（共 10 节） · 复现得分 0.25

<sub>🟢 成功 · 🔵 未执行 · ⚪ 仅阅读 · 🟡 部分完成 / 资产缺失 · 🔴 阻塞 · 🟣 待决策 —— 原文未改动；相对媒体链接需要原 README 所在目录的上下文。</sub>

<sub>original_sha256: `d222552b7987d1f53cdd395cd3e00112b567a9029ecfae0253a4068bf1aa334e` · round-trip: verified</sub>

---

<!-- rigorpilot:repro:end -->
# Goal：借鉴老师的计算模块，训练自己的512维拓扑AE

<!-- rigorpilot:repro:begin kind="section" section="Goal：借鉴老师的计算模块，训练自己的512维拓扑AE" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 目标与起点

最终产物是自己的共享网络：由随机初始化和GT监督得到自己的权重，最终在原固定100条上完整重建；不以加载老师权重、使用老师保存latent、抄参考预测或训练点坐标先验代替成功。

本轮先在原无损CAD50完成一组完整训练对照：A为原512骨干，B为下面定义的OwnAE-v2。两者都随机初始化并使用同一新的训练配方。原100条74/73历史模型、CAD B2500和已完成实验只读保留，不从B2500续训本轮，不覆盖正在运行的实验。

此次包中的terminal FFN H实验已完成：末尾Face F1=0.3525331725，Control=0.3539593250；H严格32、Control33，困难组均0/16。不要重新执行它，也不再把结构预审文档中的“尚未执行”当最新状态。

<!-- rigorpilot:repro:begin kind="section" section="目标与起点" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 本轮实验身份

- A_v1_recipe_control：原Encoder512/latent512/Decoder1024，12组Graph+Transformer、16个attention-only Decoder block；原输入和Graph公式。采用下方统一新训练配方。
- B_v2_teacher_blocks：同样宽度、latent、层数、32维Edge/Face输出和评分，但采用下方三类结构变化。
- 不是再训老师128宽的小AE，不加载老师任何预训练权重、不用教师蒸馏、没有可训练UID/顶点查表。
- 这是完整结构版本对照。A/B的区别不止一个子层，不能把B的收益唯一归给Fourier、Graph或FFN；同一新配方的A帮助判断训练更新方式本身能改善多少。

<!-- rigorpilot:repro:begin kind="section" section="本轮实验身份" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## B的明确结构

1. 对顶点XYZ和GT面重心分别先计算39维Fourier特征：
   concat(xyz, sin(pi * xyz[...,None] * 2**arange(6)).flatten(-2),
               cos(pi * xyz[...,None] * 2**arange(6)).flatten(-2))。
   保留原XYZ；不修改数据数组、不合并或重排顶点。
   保留两份共享到所有mesh的输入投影vertex_input/face_input，均改为Linear(39,512)。这点不是照搬老师单一共享输入投影，已明确保留原工程接口。

2. 12组Encoder的Graph残差改成：
   neighbor_mean=mean(raw_h_neighbors)
   message=W_self(raw_h)+W_neighbor(neighbor_mean)
   h=h+GELU(LayerNorm(message))
   不先对参与聚合的raw_h作LayerNorm。neighbor_projection无bias。
   后接pre-LN MHA残差与pre-LN GELU FFN残差；Encoder FFN维度512→2048→512。
   Encoder并非原来没有FFN；当前改动包括Graph顺序及将Transformer激活显式设为GELU。
   保留原Encoder末端LayerNorm以及mu/log_variance两条投影，不再顺带做LN消融。

3. Decoder全部16块都使用：
   u=h+MHA(LayerNorm(h))
   h=u+W2(GELU(W1(LayerNorm(u))))
   每块宽1024，8个attention head，FFN为1024→4096→1024，dropout0。
   不是仅在第16层末尾加一次FFN；不是在老模型上通过hook新增模块。
   从随机初始化开始，FFN不采用为续训保持恒等而设计的末层全零策略。

4. latent512；latent_input为512→1024；末端LN与两个独立Linear(1024,32)沿用。
   两个32维head不合并、不扩大、不新增UID参数。保留每mesh中心化、16space+16time评分、原正scale及Face面积因子。

参考代码student_v2_reference.py是新编写的骨架。它只完成小尺寸CPU前后向与置换一致性测试，未完成当前服务器集成/完整尺寸训练。可以据此实现正式工程，但必须验证实际执行路径，不得宣称它已经完成收敛。

默认B有246,575,680个可训练参数，A有112,212,544个（logvar均冻结），来自代码构造计数。B的参数和计算更大，这不是纯“非线性效果”消融，也不保证更易训练。事先记录显存和单步耗时，采用激活重计算，不静默降宽或删块。

<!-- rigorpilot:repro:begin kind="section" section="B的明确结构" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 统一训练配方——本轮明确的工程选择，不冒称老师原始训练器

- 两支随机初始化seed=0；同seed不意味着不同结构具有同一初始函数。能共享的初始参数可按名称/形状复制一份共同随机初值；无对应参数正常独立初始化，不能从老师或B2500拷贝。
- fresh AdamW，所有可训练参数LR=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0.01；global gradient clip=1。
- 100个optimizer update线性warmup到上述LR，之后固定；不继承任何旧Adam或RNG。此选择借鉴老师的AdamW量级，不声称已证明最优。
- microbatch=1完整mesh；累积5条后统一clip/step，每条loss除5。每epoch对50 UID用独立数据generator打乱一次，分为10个5条batch，每条每epoch恰好参与一次。两支使用相同batch序列。
- A和B都用固定四组hard TP/TN/FP/FN BCE：各组整mesh平均，再固定除4，空组贡献0；Edge+Face相加，不加额外外层0.25。此轮不扫描loss。
- Edge覆盖全部i<j pair，不能负边采样。
- Face包含全部GT面；每条每次参与时重新采样ceil(1.5*F)个唯一非GT三元组（合法负例不足时全取）。保持局部顶点索引、三元组内排序和去重。这个比例是本轮明确选择，不冒称精确恢复老师sampler。
- 使用只依赖seed/epoch/UID的独立负例generator，确保两支同一次参与的负例相同，且网络随机状态不影响负例序列。保存生成代码、seed与计数；不在本轮加入依赖模型预测的hard-negative refresh。
- 原scoring：edge_scale=0.9306077080970389，face_scale=0.39804385828730726，Face interval factor=0.25，logit>0。
- μ路径、sampling关闭、KL=0、logvar冻结，dropout=0。显式sample_latent=False；不得让model.train()自动启用噪声。
- FP32、SDPA MATH、TF32/autocast关闭，保留已验证确定性Graph归约。MHA若有额外推理fastpath，应在成对运行时一致配置，不能混用不同数学前向。
- 所有网络层训练；不得把Encoder/Decoder输出缓存成不更新的特征。

<!-- rigorpilot:repro:begin kind="section" section="统一训练配方——本轮明确的工程选择，不冒称老师原始训练器" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 工程集成必须满足

原runtime通过exec注入sampling_forward.py；其中仍有旧的attention-only Decoder循环。所以仅替换类定义不够。

本轮采用原生nn.Module.forward的版本入口，例如--model-variant teacher_blocks_512；使新Encoder/Decoder在真实训练和验收中都被调用。不要使用末端pre-hook作为永久架构；不要再依赖一个会静默覆盖新forward的旧诊断脚本。

checkpoint必须保存model_variant、完整结构配置、模型、optimizer、RNG、数据和负例generator状态、已完成update/epoch及代码版本。全新进程按配置构造网络后strict加载，不能依赖当前进程已安装过的hook。

训练前只做必要检查：输入/索引、损失分块归约、各新模块梯度连通、train模式仍用μ、单mesh与多mesh逐条结果一致。不要重新做三种loss扫描、老师权重逆向或整个历史审计。

<!-- rigorpilot:repro:begin kind="section" section="工程集成必须满足" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 预算和验收

两支各最多20,000次5-mesh optimizer更新，相当于每支100,000次mesh参与，2000完整epoch；不是20,000次全50条更新。

另设本轮总资源上限24 GPU小时（两支合计），先达到者停止。不同结构计算速度不同；资源截止导致步数不齐时，以共同完成的检查点比较，同时分别报告GPU小时，不能把不等预算端点说成公平胜负。只使用用户授权设备，不抢占原100条或其他任务。

评价点：0、500、1000、之后每1000步直到20,000。完整冻结同一checkpoint评价全部50条，阈值0，由预测Edge完整枚举所有三角候选再分类。GT面没进候选计FN，不补GT边、不限制候选数量、不使用训练pool替代实际指标。

主指标：全50实际Face micro-F1是否达到0.997；同时报告Edge F1、Edge/Face四种错误、严格成功UID，16条66—274点组的全部指标。不能仅凭总F1掩盖困难组。

分别保存最高Face F1、最高严格成功和最终checkpoint。到0.997后在全新进程冷加载完整验收。严格50/50单独报告。达到阶段目标可停止该分支，不自动接Flow/点网络。

如果A已很好而B无额外收益，优先保留原结构和有效的新训练方法；不能为了证明改结构正确而隐藏Control。
如果B显著优于A，把B作为自己的新版候选；不要归因到某一个子模块。
如果两支都差，报告原始梯度/优化日志及按层的必要表示，不立即再加层或再换loss。

<!-- rigorpilot:repro:begin kind="section" section="预算和验收" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 原固定100条的后续

CAD只是低成本结构验证，不是项目终点。选定版本且CAD冷验达标后，使用相同已确定的结构和可复现配方，在原固定100 UID上新建独立训练分支；默认随机初始化、不加载老师权重，保留所有历史模型。

原100条阶段目标仍是实际Face F1>=0.997、严格100/100另列；全部Edge pair、原顶点/面目标和真实Face候选不可改。先测最大mesh的完整前后向显存与耗时；可chunk/recompute，但不删大mesh、截边或用子图冒充整mesh。

本轮不自动启动原100条新长训。在报告中给出可直接运行的迁移命令与资源估计，等待这一轮结果决定。

<!-- rigorpilot:repro:begin kind="section" section="原固定100条的后续" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 最终交付

给出实际代码/diff、架构和参数量、训练命令、配置、数据/采样哈希、每步日志、全量评价和预测、最优/最终checkpoint路径及哈希、冷加载结果。

报告第一句回答：“没有使用老师权重，原结构配新配方与自己的V2在同预算下分别训练到什么程度？”

区分已执行与建议。无需再上传或重新整理全部历史大checkpoint来开始本轮。

<!-- rigorpilot:repro:begin kind="section" section="最终交付" occurrence="1" status="readonly" risk="none" -->

<sub>⚪ 仅阅读</sub>

<!-- rigorpilot:repro:end -->
## 本轮独立工程执行入口

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python test_core.py
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python prepare_initialization.py
/opt/conda/bin/python launch_local.py --role A
/opt/conda/bin/python launch_local.py --role B
/opt/conda/bin/python launch_local.py --role eval
```

A/B启动器在双卡主机执行，eval启动器在单卡主机执行。GPU UUID固定为本轮已分配设备，启动时再次核对空闲。初始化文件只生成一次；已有运行不可覆盖。真正训练命令为：

```bash
/opt/conda/bin/python -u train.py --variant A_v1_recipe_control
/opt/conda/bin/python -u train.py --variant B_v2_teacher_blocks
```

训练进程先完成完整尺寸前后向预检并写ready，等待release及第0步全50评价；这之前真实optimizer更新为0。最终放行由根任务核对两支状态后写入。

第三卡每个checkpoint使用新进程严格加载模型；不会常驻占卡等待队列。计费记录保守包含预检、训练进程等待与评价，训练在总计23 GPU小时收尾，预留1小时完成评价，总上限仍为24 GPU小时。阶段达标后另作一次新进程重复评价核对数组。
<!-- rigorpilot:repro:begin kind="section" section="本轮独立工程执行入口" occurrence="1" status="info" risk="low" -->

> [!NOTE]
> 🔵 **已选为目标 · 未执行**
> 命令：`/opt/conda/bin/python launch_local.py --role eval`
> 已选为最小可信目标；本次运行未请求执行。
> <sub>证据: [SUMMARY](SUMMARY.md) · [COMMANDS](COMMANDS.md) · [LOG](LOG.md) · [status.json](status.json) · tier: code-development</sub>

<!-- rigorpilot:repro:end -->
