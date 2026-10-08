# Nexus 新会话交接提示词

请接手我的 Nexus 拓扑 AE 研究。先只读核对下面的材料，给出实际进度和证据，不要因为接手新会话而自动重启、续训或修改实验。我的重点是判断：老师的 CAD50 很快 overfit，是否主要因为数据规模更小、拓扑重复更多，以及我的当前方法在老师的原始目标上能否严格成功。

## 1. 状态与本次接手任务

<!-- FINAL_RESULTS_START -->

**CAD50训练及服务器文件审计已完成，已按2000次有效轨迹更新停止。**

同一checkpoint达到50/50：否；首次50/50 step：None；最高联合严格成功：30/50（step1300）；末尾：30/50（step2000）。

末尾Edge FP/FN=8993/1916；实际Face FP/FN=11418/3491。

末尾未联合严格成功UID：teacher_cad50_00, teacher_cad50_02, teacher_cad50_08, teacher_cad50_10, teacher_cad50_13, teacher_cad50_20, teacher_cad50_21, teacher_cad50_22, teacher_cad50_24, teacher_cad50_25, teacher_cad50_26, teacher_cad50_32, teacher_cad50_33, teacher_cad50_34, teacher_cad50_35, teacher_cad50_37, teacher_cad50_38, teacher_cad50_41, teacher_cad50_42, teacher_cad50_48。

21个固定checkpoint均实际完整网络评价全部50条，独立CPU审计核对1050条预测、完整三角形枚举、全部checkpoint/Adam和保护模型哈希；文件审计没有另外重跑网络。

仅对已评价的检查点声称成功；不把检查点间隔中的全部更新称为均保持成功。

最高成功checkpoint：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921/run/checkpoint-step1300.pt`；SHA256 `2dcc9e2eaf207a4efd9a648efbb823f4df10495ef7f347e6fdc0e87efd0e37c3`。

最终可续训checkpoint：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921/run/checkpoint-step2000.pt`；SHA256 `ae7c2835fe7ad9da72edd413f66373b415721b2cfac72df8c6b3fc141b987358`。

最终完整推理模型：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921/run/model-step2000-inference.pt`；SHA256 `637cfb41e9ce169bb3196256eab4f93d9a710a871a419bfcd8903ba10ce8174d`。

从step200恢复并精确重放201—287；最终有效轨迹2000步，每条有效参与2000次。含87步中断恢复重算的已记录物理Adam调用合计2087，未自动增加模型轨迹预算。

服务器评估包：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921/TeacherCAD50_Fresh512_2000_Review.zip`；预测包：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921/TeacherCAD50_Fresh512_2000_Predictions.zip`。本段由最终审计生成，随后执行打包；整包SHA以成功生成的package_manifest.json为准。

本地Downloads是否已经同步、通知是否已发送，仍需接手时核对，不在此自动声称完成下载。

这证明或未证明的是当前配置在老师CAD50上的严格拟合；不能自动推论原100条大mesh已解决。规模/拓扑重复事实见data_complexity.json，训练结果和剩余错误见REPORT.md及per_mesh.csv。

<!-- FINAL_RESULTS_END -->

接手后先回答三个问题：

1. 当前训练实际完成到哪一步，是否已按预算停止？同一 checkpoint 是否达到全部50条 Edge＋实际 Face 严格零错误？
2. 数据规模和拓扑重复的实测差异，能支持“老师的数据更容易”到什么程度？区分事实、解释和未验证的因果推断。
3. 若未全对，剩余错误集中在哪些 UID、Edge 或 Face、候选缺失或判别错误？先读已有轨迹，再提出一个最小下一步，不自动增加预算。

## 2. 项目的长期目标和两条实验线

长期目标是固定100条 mesh，由同一个共享模型、同一个 checkpoint 严格重建。不能事后只挑成功子集，不能拼接不同 checkpoint 或不同 mesh 专属 head 的成绩。

当前新增 CAD50 是**独立实验**：我的架构、评分、Soft4 和 math00 跑老师的原始 mesh 目标；不是复现老师的 Flow、文本生成、点生成到拓扑串联，也不是加载老师权重或旧72/74条模型做零样本推理。原100条主线、模型和记录只读保留。

当前两条线都是 μ 重建诊断，不因为历史做过 VAE 就擅自恢复 sampling/KL。

## 3. 统一严格验收规则

- 判正为 logit>0，不以四舍五入的 F1=1.000 代替零错误。
- 每条严格成功：Edge FP=FN=Face FP=FN=0；集合成功必须来自同一 checkpoint。
- Face 必须从当次预测 Edge 图完整枚举三角形/clique 候选，再用 Face head 分类。未入候选的 GT Face 计 FN，训练 Face pool 的结果不能代替实际重建。
- 允许固定 checkpoint 后逐条完整评价，不要求物理上一次 packed50/100。
- 报告真实预测，不修复 mesh、不用 GT Edge 图替代预测图、不截断实际 Face 候选。
- 区分接口预检、真实完整网络训练、缓存路径验证、实际完整网络验收、独立文件审计。

## 4. 已完成历史，只保留影响当前判断的结论

早期两条 mesh 曾严格成功，但新增大 mesh 明显更难。后来固定100条、fresh512、μ路径、KL=0，从随机初始化建立主线；其后做过 LR、固定 Face 难负例、共享 head 和 Decoder 末端适应。历史上有74条联合严格成功的模型；当前两块联合分支的共同源模型是72条。不同分支有取舍，不应混称为同一模型的最好结果。

BF16/Flash 与图聚合的历史数值排查已做过；当前使用已核验的 math00 和 fully-differentiable Soft4。不要无具体异常又从头重跑后端、VJP、line search；11点有限位移验证也已完成。

自由 Edge 表示能成功、共享 head 或末端适应有效，只能支持相应范围的可行性，不代表完整 AE/VAE 和 Face 都通过。单次有限预算失败也不能证明容量不够。

最近完成的固定100条配对实验：从相同两块72条源状态出发，历史 S=71、F=29 固定；A 原目标，B 仅把历史成功组 Edge 项乘0.5，其他三项不变。两支各500次全100条更新，均停止。末尾：

| 分支 | Edge FP/FN | 实际 Face FP/FN | 联合严格成功 | 原72条保持 |
|---|---:|---:|---:|---:|
| A 原目标 | 91097/1 | 5759/178 | 72/100 | 全保留，无新增 |
| B 成功组Edge×0.5 | 100112/0 | 5845/185 | 72/100 | 全保留，无新增 |

B 的历史失败组 Edge loss 较低，但 Edge FP 比A多9015；没有扩大严格覆盖，不能把这次降权宣布为修复。完整训练和16个检查点的真实网络验收均已完成，不重跑。

该报告本地路径：
`/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/decoder_last2_success_edge_half_pair_20260921/REPORT.md`

评估包：
`/Users/luthier/Downloads/Nexus_SuccessEdgeHalf_AB500_Review_20260921.zip`

预测包：
`/Users/luthier/Downloads/Nexus_SuccessEdgeHalf_AB500_Predictions_20260921.zip`

## 5. CAD50 的数据身份和接入方式

真正收到的原始附件叫 `nexus_overfit_data_results_no_code.zip`。协议曾写 `teacher50_for_current_ae.zip`，但这份独立适配ZIP没有提供；当前是从原始附件中的真实训练缓存 `data/point50/training.pt` 无损导出适配数据，不能声称读过未收到的适配包。

原始ZIP SHA256：`982686feb58c207932d9311786cabb3ffd7a1397627b83c44df3d5c8ab2189e5`。

缓存 SHA256：`22fa88be47b589166c1f66a93c03e274083f23c207938b059d1996bb72995465`。

固定 UID 为 `teacher_cad50_00` 至 `teacher_cad50_49`。保留缓存的 FP32 坐标、原 faces、顶点编号及 original_vertex_indices；训练/验收不再重排。没有 d9 量化、焊接、合并近点、删补面、简化或重三角化。sample00=68顶点、sample13=16顶点，不能合并为8点。

float-mesh 接口只提供模型真实需要的 vertices、vertex_mask、faces、incidence_index 及派生标签；不伪造 octree、条件点云。原100条加载器不变。缓存中其他特征、文本和老师预测均不用作标签或初始化。

已核对的数据比较：

| 指标 | 老师CAD50 | 原固定100条 |
|---|---:|---:|
| mesh数 | 50 | 100 |
| 顶点总数 | 2872 | 106325 |
| 顶点数中位数 | 12 | 953.5 |
| 最大顶点数 | 274（sample20） | 2547 |
| GT Edge总数 | 8364 | 309194 |
| GT Face总数 | 5576 | 204330 |
| 全部无向pair数 | 235741 | 84669234 |

原100条全pair总数约为老师50条的359.16倍，**这不是实测训练时间倍率**。

老师50条中，24条都是8顶点、18边、12面，且它们的顶点—面关联结构精确同构；50条共23类这种同构结构。定义是保留顶点/面节点类型的二部关联图精确同构，仅在只读统计时忽略坐标和面朝向。不能据此说所有几何完全相同，也没有因此合并训练样本。

这些事实支持规模与重复度差异很大，但还不能唯一解释“一晚上”或保证我的训练必然成功。两边验收目标和每条参与次数也必须对齐，不能拿高Face F1当严格50/50。

## 6. CAD50 已锁定的实际训练协议

- seed=0 正常随机初始化，fresh Adam；不加载任何旧/老师学习权重或 optimizer。
- latent=512；Decoder16块、hidden1024；共享 Edge/Face head 各 Linear(1024,32)。训练完整 Encoder、μ、输入/latent映射、Decoder和两个head；logvar专属参数冻结且不进optimizer。
- 明确 z=μ，sampling关闭，KL=0。实现用 eval 模式但保留 autograd，已通过 hook 验证真正输入Decoder的 z 与μ逐位一致，避免 train() 隐式采样。
- math00：确定性Graph前后向、显式FP32 SDPA MATH；TF32/autocast关闭；非重入重计算精度上下文一致。不能缓存可训练上游输出绕过训练。
- 原32维 spacetime 评分、逐mesh中心化，不加RMS或新head；Edge scale=0.9306077080970389，Face scale=0.39804385828730726。Face评分内部0.25与loss外层系数分开。
- fully-diff Soft4：membership、分子、分母都参与梯度，tau=1、epsilon=1e-8、FP32 reduction，内部固定/4；**没有额外loss外层0.25**。
- Edge覆盖全部 i<j pair。Face pool保留所有GT，基础负例规则为全部非GT三环＋最多1×GT数量wedge负例＋0.5×GT数量uniform负例；按现有生成/选择代码确定性去重，不足不复制填数。seed=0，逐UID实际数量/哈希有记录。
- Face pool共13946条候选，其中GT5576；训练前固定，不加载旧模型挖掘，不复用100条750215个候选或71/29分组，不动态补负例。用户已授权按此判断执行。
- 每次更新按固定UID顺序完整累积50条：microbatch=1，每条(Edge+Face)/50，最后统一clip=1、Adam一次。各mesh等权。
- Adam betas=(0.9,0.999)、eps=1e-8、wd=0；四组 encoder_mu / decoder / edge_head / face_head。
- warmup公式：第t次更新 LR=目标LR×min(t/100,1)。目标E/μ=1e-5，D/heads=1e-4。100次后保持，不再扫参。
- 预算2000次**全50条**Adam更新，每条直接参与2000次，总计100000次完整mesh训练参与；不是2000 microbatch，也不直接等同老师的AE step定义。
- step0及每100更新完整验收到2000，共21轮；保留实际Edge/Face预测、逐UID计数、Face FN候选缺失/存在但判负、实际Face FP在pool内外分布。单列00、13、20。
- 首次50/50若出现保存checkpoint，并在剩余既定预算内继续；只能声称验收过的检查点保持。预算结束停止，不自动扩层、改LR/loss/pool或恢复sampling/KL。

启动前预检已通过：50条重复forward逐位一致，前后向有限，所有可训练参数连通梯度，512个μ通道有梯度，logvar不变，预检0次optimizer update、Adam为空；00/13/20的loss和embedding梯度与当前参考实现逐位相同。预检不等于训练成功。

## 7. 位置、证据与交付

CAD50服务器根目录（下文记作R）：
`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921`

对应本地目录：
`/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_cad50_fresh512_20260921`

重点读取：

- `R/status.json`、`R/console.log`、`R/complete.json`或`R/failure.json`：执行状态。
- `R/config.json`、`R/data/manifest.json`、`R/pool_manifest.json`、`R/preflight/result.json`：有效设置、输入与预检。
- `R/data_complexity.json`：上述规模/同构统计定义和逐组UID。
- `R/train.py`、`R/runtime.py`、`R/effective_loss_and_scoring.py`、`R/loader.py`、`R/evaluate.py`：实际入口与生效实现；同时看封存依赖源码，不能只信过期注释。
- `R/run/updates.jsonl`、`R/run/eval-stepXXXX.json`、`R/run/predictions-stepXXXX/`：训练与完整实际评价。
- `R/run/checkpoint-stepXXXX.pt`：完整模型、Adam、全部RNG、参与计数。`best.json`、`first50perfect.json`（仅若存在）标记最高/首次成功点，`complete.json`标记末尾。
- 完成后查看 `R/REPORT.md`、`R/MODEL_ARTIFACTS.json`、`R/package_manifest.json`、趋势及独立审计结果。不存在的文件不能当作已产出。

启动证据包已存在：
`/Users/luthier/Downloads/TeacherCAD50_Fresh512_Startup.zip`

完成后计划交付（**须先确认已生成和校验**）：
`/Users/luthier/Downloads/TeacherCAD50_Fresh512_2000_Review.zip`
`/Users/luthier/Downloads/TeacherCAD50_Fresh512_2000_Predictions.zip`

完整大权重不默认重复打包；路径与SHA写入模型索引。评估ZIP应含配置、有效代码、数据/pool清单和哈希、完整日志、逐mesh验收、可读报告、核验范围。不含任何密码。

原100条关键保护模型：

- 当前两块72条源：`/guohaoran/nexus_fast_track/diagnostics/decoder_last2_joint_fixed100_20260920/run/model-last2-joint1000-tail1500-block14new500-inference.pt`
  SHA256 `82cf33fbc0151d72332218c5e447f7bf84e8934947e4c70148205c6a308db884`。
- 其可续训状态：同目录 `checkpoint-new0500-tail1500.pt`，SHA256 `30d207d3508429a749885764712c169a85479e4c2673f94f816fa54cfa94e753`。
- 历史74条模型：`/guohaoran/nexus_fast_track/diagnostics/face_head_recovery_tail1000_fixed100_20260919/run/model-tail1000-face1000-inference.pt`
  SHA256 `0816b3f7358f7a39fc1bd1fbe70ea338a726a0c89fe9ecafc777bbd9b3868170`。

本轮不覆盖、不热替换这些模型。若以后续训，必须明确所选起点、权重/Adam/RNG及新预算；两支步数不能相加成单模型累计。

## 8. 接手操作边界

服务器最近可用地址为 `ssh root@172.16.78.10 -p 31548`，端口可能变化。当前Mac曾建立ControlPath `/tmp/nexus-cad50-31548.sock`；先只读验证连接，不把可用性当常量。本文不含密码，需要时由我另行提供。

原会话的“CAD50独立训练完成验收”监测曾存在，但本次恢复时工具报告已不存在。是否重建完成通知已向用户询问，当前不要假定监测仍有效。服务器恢复入口本身会在训练完成后执行审计与打包；桌面通知、下载及最终交接需单独核实。接手前先检查最新监测状态，避免重复创建或启动。

我希望你少问已经明确的配置，多实际读取证据；缺文件、依赖或连接就列出具体阻塞，不编造执行结果。任何新的训练预算、架构、loss、LR、pool干预先给出基于现有结果的明确建议；本交接提示词本身不授权新一轮训练。

若你不能访问我的本机或服务器，就以我附上的最终Review ZIP为证据，明确范围；不要把 `/Users/...` 本机路径说成可公开下载的网址。


## 本地交付核验补充

2026-09-21，以下两个包已实际下载到本机Downloads，整包SHA256、ZIP CRC和全部1296项内部文件SHA已核对。本文是最终交接提示词，不再是运行中草稿。该交付补充在ZIP生成之后写入本地提示词，不回写ZIP以避免自引用哈希。

- `/Users/luthier/Downloads/TeacherCAD50_Fresh512_2000_Review.zip`
  SHA256 `476986eda83529e3fd7ac71295149dc790fb5583171f1130dfc05228443b1897`；内部文件246项全部通过。
- `/Users/luthier/Downloads/TeacherCAD50_Fresh512_2000_Predictions.zip`
  SHA256 `1e73bb132fa37b35de70d6c972ef052826f9cad2987d2f044bb6b7f9210f714b`；内部文件1050项全部通过。
