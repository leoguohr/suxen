# R1_D15_COMPAT

已按本轮确认推进独立版本。**原D2完整模型重放通过；D15表示检查、非零迁移等价测试及完整权重迁移通过；D15两物体训练已启动，尚未完成生成验收。** 原D2/D4、VAE/AE V2均未修改，GPU1未使用。原先“只做CPU”的限制已由用户本轮授权更新；用户明确允许继续训练且不设时间限制，首轮更新上限6000仍单独记录。

## 已验证结果

|项目|实际结果|证据|
|---|---|---|
|D2原冻结协议真重放|32/32完整树正确；288/288层parents、实际noise、predicted_cells与原终验逐元素相同；estimate最大差异0。433.13秒，GPU0；20步/层、每树180调用，共5760调用|`outputs/d2_baseline/evaluation.json`、`SUMMARY.json`及真实NPZ|
|完整性门禁|11项CPU检查通过。缺失、重复、乱序、超时、非complete、缺层等使整轮incomplete/非零退出；不缩小32条分母；ratio边界为≤0.1|`evidence/integrity_tests.json`|
|CAD50表示往返复算|depth9：0/50、48份点数不变；depth12：0/50、全部点数不变；depth14：48/50；depth15：50/50。D15最大RMSE=3.0517578125e-5，最大ratio=0.090682747308|`inputs/d15/label_audit.json`；仅表示诊断，没有CAD50生成/条件适配/训练|
|真实源代码的非零迁移等价|原冻结VertexDiT前9层与D15逐元素相同，测试激活非零输出与残差分支；深度范围、mask、空树、原序/逆置换、解码通过|`evidence/d15_cpu_tests.json`|
|完整模型迁移|906共享张量+旧embedding前9行不变；仅新增6行×1536=9216元素；2,332,439,560参数；整模型strict load；Adam状态0|`evidence/migration_audit.json`（每个共享张量有SHA）|
|训练接口小模型|CPU损失/梯度有限，fresh Adam实际1次更新与完整模型/Adam保存恢复通过；不计生产模型成绩|`evidence/train_path_cpu_test.json`|

原D2种子28000000..28000015，seed外层、UID顺序000105→000195；**原D2始终是9层**。新D15则明确15层、每层20 Euler，非空树300次网络调用。没有因表面步数相同而混用协议。

首轮D2部署到共享盘时，小文件解包未结束就启动，Python找不到入口后退出，**没有开始模型计算**。改为NVME解包、核实入口存在后重放成功；该失败不被算为一轮采样。CPU训练接口测试发现并修正一个新评价脚本语法错误后重跑通过。没有为通过测试修改模型目标。

## 唯一模型改动与标签

模型差异见 `evidence/model_changes.diff`：独立 `d15_code/mini_nexus/vertex.py` 分离 `max_depth` 与 `rope_reference_depth`，后者固定9；`training.py` 传递参数。旧入口默认深度仍为9，原Fourier、时间编码、阈值、bit顺序、速度目标保持不变。其余模型文件与D2冻结snapshot相同（`evidence/source_provenance.json`）。新构造入口是 `scripts/d15_model.py`。

新embedding固定CPU种子2026092515、std0.02，六行SHA为 `a17cca8fbcc5166f9bac01e3285d32a177ad23fba3a2f16413fbd452528e503c`。源D2 checkpoint SHA为 `2384b430fa52d5012cd793fc2781c79791814f6a4b2e8ac120a711fb91112e1b`（已实际读取验证）。

D15标签从已核验stage2浮点坐标提升float64后floor/clamp重新编码，保存 `original_to_leaf`、原始GT、raw q15、唯一leaf、解码XYZ；逐点核验q15>>6=q9。不是q9乘64。D2两物体仍为8/52唯一顶点，原始24/140条记录分别有16/88个完全重复坐标，真量化碰撞0；显式记录，不将原始GT匹配点数失败藏起来。CAD50未删GT记录，D15碰撞0。未取得附件提到的verified_metrics.json，已用原浮点数组独立复算，未据文件口述补造结果。

## 训练与状态

执行卡：`TRAINING_EXECUTION_CARD.md`。实际训练仅原D2两份固定点云/法向，15层均衡调度；VecSet/DiT联合训练，lr1e-5、WD0、clip1、累积8、BF16、速度MSE。新Adam和新RNG从update0开始，**不是旧Adam/RNG无缝续训**；不用老师模型/Adam。未重复老师逆向或已有重放。

- GPU0：`GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d`。
- NVME代码及运行：`/tmp/r1_d15_compat_20260925`，训练wrapper PID1553（仅本次启动身份，之后须实时复核）。
- 持久目录：`/guohaoran/tmp/r1_d15_compat_20260925`。
- 初始化包：`/tmp/r1_d15_compat_20260925/migration/checkpoint-update0.pt`；SHA `0e2a34840c65128d285a327ac346a33635ab5d664343d868338e7762c4d84fff`。
- 完整训练checkpoint：持久目录下 `checkpoint-last.pt`，含model/Adam/step/配置/迁移信息/CPU-CUDA-Python-NumPy RNG；恒定LR，scheduler=null。以 `checkpoint_identity.json` 中已完成回读校验的版本为准，`.partial`不能当完成。大权重留服务器，不混入轻量证据包。
- 第1次实际更新loss=0.01882604198，5.71秒，只覆盖调度最初8个旧层microbatch，**不代表新6层已学会，更不是最终训练速度或生成成绩**。
- 每200更新开发4种子×2物体；全部通过后预留16种子×2物体终验一次。缺失/容量中止不筛除；失败保留数组，不反复称同一组seed为未见。
- 启动时保存的历史快照到update 42，日志完整保存至update 42；loss/梯度均有限，实际消费UID/depth均符合30组合调度。完整训练checkpoint已持久保存并回读校验至update 1，SHA `3669015745e05ac2f85b8c39ed99d5078594f0bde3c3aff7af5ea82e4d7ef4ac`，907个Adam状态。该启动快照时尚未开发评估；最新进度见下节。已安排本任务每15分钟检查；只有有意义变化才通知，跟进ID `r1-d15`。

## 最新监督：第200步开发评估（2026-09-25 约03:00）

真实开发生成8条已全部完成，完整性门禁通过；**完整15层整数树正确0/8，开发验收未通过**。本地重新核验全部预测哈希和逐层数组、重算指标，与服务器评价JSON完全一致。没有使用缺失样本缩小分母；未运行保留终验种子。

|开发seed|000105预测N / 目标8|最早失配层|000195预测N / 目标52|最早失配层|
|---|---:|---:|---:|---:|
|91015000|2 / 8|11|24 / 52|9|
|91015001|3 / 8|10|18 / 52|7|
|91015002|2 / 8|11|16 / 52|9|
|91015003|0 / 8|10|28 / 52|5|

所有样本点数与目标不同，按老师评价器约定一一匹配XYZ RMSE未定义，不能补造数值。逐层连续占据MSE和阈值余量仍保留。000195在旧深度范围也有失配，但没有在这组开发种子上做迁移前权重配对，不能仅凭本次结果归因为训练导致遗忘。

截至本次下载连续日志已核验219步，loss与梯度均有限；实时进程存在、GPU0在运行。训练保持既定配置继续，未改模型/loss/LR/数据。update200完整checkpoint已持久保存并回读校验，SHA `0f5ab9f16a278b8c6305b84dc4d56e4f627b4875eff2a9cb9968bc6e84851268`，907个Adam状态。下一固定开发评估为update400。

新证据：`outputs/monitor_20260925_0255/verification.json`、`train.jsonl`、`dev-000200/`全部实际预测与服务器/本地复算指标；原始结果归档也保留。这里只做只读监督与本地复核，未新增GPU实验。

## 后续监督：第400步开发评估（2026-09-25 约03:32）

完整性门禁通过，真实完整15层整数树正确 **2/8**，仍未达到开发通过条件。000105在seed91015001、91015002恢复完整正确坐标集合；另外两例最早在depth14失配。000195四例均未通过，最早失配层分别为10、9、10、5。不是仅按预测点数判定。全部预测在本地重新核验哈希并复算，结果与服务器JSON相同。

日志连续核验至update507，loss与梯度均有限；实时训练进程/GPU0仍在运行，配置未改，未启动终验。update400完整checkpoint持久回读已验证，SHA `37d492224d45328d80a509c668331c67e2b4ca01131865fb54221140b1ce49e4`。证据在 `outputs/monitor_20260925_0332/`。下一固定评估为update600。

## 后续监督：第600步开发评估（2026-09-25 约03:50）

完整性门禁通过，真实完整15层整数树正确 **4/8**，仍未通过开发验收。000105四个种子全部恢复完整正确坐标集合；000195四例均失败，最早失配层为9、9、9、5，预测点数为52、52、50、51。前两例虽然点数恰好52，坐标集合仍不正确，未按点数误判通过。全部实际数组在本地完成哈希检查和指标复算，与服务器JSON完全一致。

连续日志核验至update602，loss与梯度均有限；实际进程/GPU0仍在运行，配置未改，终验未启动。update600完整checkpoint持久回读已验证，SHA `ab08af454a3d51a80061768edb5bf4b219f2ca146fcc43bed77f2926aa4e75d6`。证据在 `outputs/monitor_20260925_0350/`。下一固定开发评估为update800。

## 可运行命令与交付

所有命令与源码保留在 `scripts/`、`outputs/d2_baseline/baseline_command.json` 和服务器 `training_command.json`。CPU重放环境沿用上一轮Python及隔离的scipy依赖；实际运行过 `test_integrity_gate.py`、`encode_d15.py`、`test_d15.py`、`test_train_path.py`、D2完整生成/独立评价、生产`migrate_d15.py`。

```sh
# 服务器新目录；此训练已启动，不要重复启动。
cd /tmp/r1_d15_compat_20260925
CUDA_VISIBLE_DEVICES=GPU-5a318b25-13c3-69ab-abb0-6b0e8751097d \
 /guohaoran/envs/nexus-algo/bin/python scripts/train_d15.py \
 --checkpoint migration/checkpoint-update0.pt \
 --expected-sha 0e2a34840c65128d285a327ac346a33635ab5d664343d868338e7762c4d84fff \
 --manifest inputs/d15_manifest.server.json --labels inputs/d15 \
 --output run --durable /guohaoran/tmp/r1_d15_compat_20260925 \
 --updates 6000 --eval-every 200
```

当前入口只接受update0初始化包；训练状态完整保存，但本轮没有执行中途恢复，不宣称已验证生产续训。下一步保持当前配置完成开发/终验，依据真实预测与完整性结果报告。不要把depth15表示可达性、CPU测试或已通过的D2替代D15生成成绩。

本次运行快照及完整已见训练日志在 `outputs/training_initial_snapshot/`；统计见 `outputs/TRAINING_SNAPSHOT_SUMMARY.json`。服务器脚本和本地逐文件代码哈希已一致核验。

已实际CPU mmap读取生产训练checkpoint复核其15×1536 embedding、907个Adam状态及step一致性、CPU/CUDA/Python/NumPy RNG；详见 `evidence/checkpoint_cpu_audit.json`。这是读取生产checkpoint，不是小模型替代。

## 连接异常监督（2026-09-25 约04:23）

三次SSH尝试均在认证前被远端关闭；独立连接诊断确认TCP可建立，但SSH协议标识交换被关闭，尚未进入密码认证。**本轮无法确认训练进程、GPU、checkpoint或第800步评估当前状态**，不能据此认定训练停止或仍在继续。上一轮约04:07实际检查到update800正在保存checkpoint；最后确认持久回读成功的是update600，最后已复算开发成绩4/8。未重启或修改训练，保留定时跟进。证据：`outputs/monitor_20260925_0423/connection_failure.json`。

## 连接异常持续（2026-09-25 约04:42）

本轮两次独立SSH检查仍在认证前由远端关闭，TCP可建立。连续两轮无法读取服务器，当前训练是否运行、第800步保存是否完成、后续评估均未知；没有新增训练结果。未修改或重启远端任务，跟进保持启用。需要检查平台实例状态及32483端口映射。证据：`outputs/monitor_20260925_0442/connection_failure.json`。

## 连接复查（2026-09-25 约04:58）

独立SSH连接仍在认证前被远端关闭，退出码255。当前训练、权重保存和评估状态均无法核实；未执行远端操作，未新增模型成绩。跟进保持启用。证据：`outputs/monitor_20260925_0458/connection_failure.json`。

05:14再次独立连接，仍在认证前由远端关闭（exit255）；本轮无法核实任何新增训练或评估状态。无远端修改，跟进保持启用。证据：`outputs/monitor_20260925_0514/connection_failure.json`。

05:30独立SSH复查仍在认证前被远端关闭（exit255），当前训练、保存及评估状态未知；未修改远端任务，跟进保持启用。证据：`outputs/monitor_20260925_0530/connection_failure.json`。

05:46独立SSH复查仍在认证前被远端关闭（exit255）。当前训练、checkpoint、评估与GPU状态均无法读取；未重启或修改远端任务，跟进保持启用。证据：`outputs/monitor_20260925_0546/connection_failure.json`。

06:02独立SSH复查仍在认证前被远端关闭（exit255）；无法读取当前训练、checkpoint、评估与GPU状态，未执行远端修改。证据：`outputs/monitor_20260925_0602/connection_failure.json`。

06:18独立SSH复查仍在认证前被远端关闭（exit255）；训练、保存、评估和GPU当前状态均未知，未执行远端修改。证据：`outputs/monitor_20260925_0618/connection_failure.json`。

06:34独立SSH复查仍在认证前被远端关闭（exit255）；训练、checkpoint、评估及GPU当前状态均无法核实，未执行远端修改。证据：`outputs/monitor_20260925_0634/connection_failure.json`。

06:50独立SSH连接仍在认证前被远端关闭（exit255）；本轮无法核实训练进程、GPU、checkpoint或评估结果，未执行远端修改。证据：`outputs/monitor_20260925_0650/connection_failure.json`。

07:06独立SSH连接仍在认证前被远端关闭（exit255）；无法核实当前训练、保存、评估与GPU状态，未执行远端修改。证据：`outputs/monitor_20260925_0706/connection_failure.json`。

07:22独立SSH复查仍在认证前被远端关闭（exit255）；当前训练、保存、评估与GPU状态均无法核实，未执行远端修改。证据：`outputs/monitor_20260925_0722/connection_failure.json`。

07:38独立SSH复查仍在认证前被远端关闭（exit255）；当前训练、保存、评估和GPU状态均无法核实，未执行远端修改。证据：`outputs/monitor_20260925_0738/connection_failure.json`。

## 新端口连通后实查（2026-09-25）

用户提供31548端口后成功连接。hostname已由fcdp7gos0kmqc-0变为1i6soqlkem2tp-0，原NVME目录不存在；当前无D15训练进程、GPU0无计算进程且显存占用0MiB。**当前训练已不在运行，持久目录旧evaluating状态不是当前进程状态。** 无法仅由这些证据判定旧实例退出的具体原因。

持久checkpoint为update800，27,990,605,466字节；本轮完整重新计算SHA256=`57a225c82e375feb7d63e25f31fdd285246b2ac41ef152c27860f82bcfa6ade8`，与原身份记录一致。CPU mmap读取确认907模型张量、15×1536 embedding、907 Adam状态且step全部800、lr1e-5/WD0，以及CPU/CUDA/Python/NumPy RNG、下一micro调度位置10。日志1..800连续且loss/梯度有限。源码与输入归档SHA也匹配。没有运行GPU或修改服务器文件。

第800步权重保存已完成；第800步开发结果未持久保存。最后可复核结果仍为第600步完整树4/8；终验未完成。日志、配置、身份与本次CPU检查已存`outputs/monitor_20260925_reconnected/`，未下载权重。原训练入口只接受update0，恢复需先补独立续训入口、验证Adam/RNG与调度恢复，并补做800步开发评估；不可直接重用旧启动命令。

尝试将r1-d15跟进端口更新为31548，工具返回该自动化已不存在；没有擅自重新创建，不再宣称定时跟进仍有效。

## 用户授权重新启动：update800续训（2026-09-25）

用户随后要求重新启动，并明确暂不评估、保存checkpoint后统一评估。已新增独立 `scripts/train_d15_resume.py`，原训练入口及模型源码哈希不变。模型严格加载800步权重，恢复907个Adam状态（step均800）和CPU/CUDA/Python/NumPy RNG，下一micro调度为10；数据与原源码逐文件哈希通过。CPU连续性检查在本地和服务器均验证恢复后的下一次更新、Adam和随机数与不中断路径一致；未运行模型生成评估。

新实例自带ONNX与protobuf存在兼容问题；首次加载进程在执行任何训练更新前停止并留档。设置仅本进程生效的 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 后检查通过，再启动成功；未修改共享Python包。

实际进程：wrapper1031、trainer1032；GPU0对应主机PID122113，显存41826MiB。已实际完成update803，最近一步4.33秒，日志连续到803、loss与梯度有限。**当前训练已恢复，不只是提交启动命令。** 权重保存仍以身份文件为准，启动时最新完整checkpoint为800，下一保存步1000。

- 运行：`/tmp/r1_d15_compat_20260925/run-resume800`。
- 持久：`/guohaoran/tmp/r1_d15_compat_20260925/resume800_20260925`。
- 每200步保存完整checkpoint并回读校验，保留逐步命名权重供后续统一评估；原800步权重保持不变。
- 累计上限6000；启动、训练期间与结束均不自动评估，生成成绩仍只有旧600步4/8。
- 新版执行卡覆盖旧自动评估协议；已重新创建15分钟跟进 `r1-d15`，仅监督训练、保存和异常，相同连接故障不重复通知。
- 新源码及输入已持久归档SHA：`df9a86b9fff742c560d094910aa7d37a76fe95078f4f8d1f44fee660318dce6d`。

恢复证据：`outputs/resume800_startup/`；连续性检查：`evidence/resume_cpu_test.json`；入口差异：`evidence/resume_entry.diff`。本次未下载大权重，未使用GPU1或V2工程。

## 2026-09-26 实时复查：日志2757，权重2600，当前未运行

新密码连接31548成功，hostname为`80dcml4evpit1-0`，GPU0 UUID变为`GPU-d59b34c9-1810-15c1-4cc9-db2bd953b574`。原NVME运行目录不存在；没有D15训练进程、没有GPU计算进程，GPU0显存占用0MiB。旧`status.json`仍写training/2757，是历史快照，不能当作当前仍在运行。没有正常退出记录或明确异常堆栈，无法确认旧实例退出的具体原因。

已下载并复核1至2757步完整日志：步号连续、loss及梯度均有限、实际UID/depth调度符合原协议，最后100步中位耗时4.29秒。1000、1200、1400、1600、1800、2000、2200、2400、2600共9份独立checkpoint仍在持久目录。最新2600步权重本轮完整重算SHA256=`bc13e55829ab4252e18ae9a040823f660e82627209f034d572be217fa1cfc3bd`，与保存身份一致，大小27,990,606,234字节；与checkpoint-last为同一文件inode。其他8份本轮核实文件存在和身份记录，未逐份重算大文件哈希。

因此可恢复起点为2600，2601至2757的157步只有日志、没有保存的权重；尚未到6000。遵守用户暂不评估指令，没有新开发或终验成绩。本次只读核查并下载轻量证据，未重启或修改远端。恢复时须使用新的GPU0 UUID，并处理入口原先只接收800步以及旧日志包含2757步的约束，不能直接重复旧命令。

证据：`outputs/monitor_20260926/live_verification.json`、`log_verification.json`、完整train.jsonl、配置、9份checkpoint身份及启动/恢复记录。权重未下载。下一步应保留原日志，从2600步完整状态在独立输出目录续训，继续不评估。

## 2026-09-26 用户授权继续：2600步恢复已实际启动

已新增独立 `scripts/train_d15_resume2600.py` 与 `scripts/run_resume2600.py`。旧模型源码、旧800步恢复入口与日志不变；新日志仅继承1..2600，原2601..2757仍留作中断证据，`history_boundary.json`记录边界和原日志哈希。服务器CPU恢复连续性检查通过，启动时再次验证完整2600步权重SHA、旧代码、条件与标签哈希。

生产模型已严格加载，907个Adam状态均恢复至2600步，CPU/CUDA/Python/NumPy RNG全部恢复，下一micro索引10。实际已完成2607步；2601..2607的UID、depth、t与噪声SHA逐项和中断前日志相同，loss最大绝对差3.44e-5以内，不宣称浮点训练逐位相同。无开发或终验调用。

运行目录 `/tmp/r1_d15_compat_20260925/run-resume2600`，持久目录 `/guohaoran/tmp/r1_d15_compat_20260925/resume2600_20260926`。本次GPU0 UUID为`GPU-d59b34c9-1810-15c1-4cc9-db2bd953b574`，GPU占用PID51112，约41790MiB；启动wrapper415。每200步保存独立完整权重，下一保存步2800，累计上限6000；继续暂不评估。原2600步权重保持不变。

旧跟进在工具中不存在，本次重新建立 `r1-d15` 每15分钟监督，已使用新目录与GPU身份；训练结束或异常退出只回收证据，不自动评估。新源码/输入持久归档SHA=`eef52b4d52b5cd6a4a4bef7f07d2842d8953b9561a1d28be300b3f98d2ec5632`。证据在 `outputs/resume2600_startup/` 和 `evidence/resume2600_entry.diff`。

## 2026-09-26 后续安排：训练正常结束后自动评测一次

用户再次明确训练期间只保存checkpoint、训练结束后再评测，且无需任务监督。已删除`r1-d15`定时任务。实时检查时训练在step2687、GPU0运行中。训练入口未改，仍每200步保存完整checkpoint并到6000步停止。

已部署独立后台进程`evaluate_after_training.py`（PID724）。它等待`training_exit.json`，只在退出码0、训练状态为`update_budget_complete_evaluation_deferred`且step=6000时，完整验证6000步checkpoint的身份、大小、SHA，然后运行一次实际模型评测。用预留seed92015000..92015015，两个物体共32条完整15层树；20 Euler步/层。预测首先落盘且带哈希，评价入口随后验证人口完整性和预测哈希，再读取GT。训练异常退出时该进程记录跳过，不执行评测。

持久结果路径：`/guohaoran/tmp/r1_d15_compat_20260925/resume2600_20260926/post_training_final_006000/`；流程状态：同目录`post_training_evaluation_status.json`。评测脚本和任务身份已复制到持久目录，也保存于本地`outputs/post_training_job/`。启动时结果文件尚不存在，不能宣称已完成评测；无定时监督和后续自动通知。

## 2026-09-26 再次更换实例：从5200步恢复

用户要求继续。新实例hostname=`7fucc846jkb9s-0`；旧NVME目录消失，没有训练/GPU进程。旧日志连续到5386，但最后完整保存是5200。已完整重算`checkpoint-005200.pt` SHA256=`b00a96af6567bce392c41abe82f5fd4810fc5b97304c572a380a73caae4aa7e9`，与身份文件相同。旧5386步日志和前段所有权重保持不变。

独立新入口`scripts/train_d15_resume5200.py`从5200恢复模型、907个Adam状态（step均5200）、CPU/CUDA/Python/NumPy RNG与调度，使用相同GPU0。服务器CPU恢复连续性测试通过；新运行目录`/tmp/r1_d15_compat_20260925/run-resume5200`，新持久目录`/guohaoran/tmp/r1_d15_compat_20260925/resume5200_20260926`。训练已实际达到5202；重跑5201、5202的UID、depth、t与噪声SHA均和旧日志逐条相同，loss最大差5.05e-5，不宣称浮点参数逐位相同。

新wrapper只在训练正常达到累计6000并记录退出后调用一次`scripts/evaluate_after_training5200.py`。评测还会检查6000步身份和完整SHA，然后对预留16对种子共32条完整树生成与评分；若训练异常退出则不启动评测。没有定时监督任务。评测尚未发生，新结果不能提前宣称通过。源码与输入归档SHA=`231c519c1ca9ef6b49555cbcb8d09848518303254931d3fb6b6c7a1a882ef3da`，证据在`outputs/resume5200_startup/`。

## 最终实查：6000步训练完成，冻结终验32/32（2026-09-26）

服务器仍为`7fucc846jkb9s-0`。训练状态`update_budget_complete_evaluation_deferred`，step6000，wrapper记录训练退出码0；6000条日志步号连续，loss与梯度均有限。完整权重`/guohaoran/tmp/r1_d15_compat_20260925/resume5200_20260926/checkpoint-006000.pt`，大小27,990,606,810字节，本轮重新计算完整SHA256=`6e332b3f8510ffdc8b053075bdf9209404ffa4af1470d33bf488b4482d9598cb`，与身份记录一致。大权重保留服务器，未下载。

训练正常退出后，一次性评测进程对预留`92015000..92015015`共16对种子运行两个条件的完整15层生成，20 Euler步/层。评测状态`complete`、进程退出码0，32条预测完整；`nexus_2k_000105`与`nexus_2k_000195`各16/16条完整整数树和原生顶点集合正确，总计**32/32**。本地CPU重新执行人口、文件哈希与逐层/坐标指标复算，结果JSON与服务器逐项相同，没有以缺失样本缩小分母。

这证明当前**两个固定点云条件的过拟合实验**达到完整树原生D15验收；未验证CAD50生成或新物体泛化。两个源GT分别有24/140条原始重复顶点，而预测是8/52个唯一顶点；因此原始浮点GT点数门禁不通过，不能把原生树32/32说成原始重复顶点的老师评价全通过。显式去重GT的连续XYZ诊断在全部样本中通过，但它是另一个诊断口径。

完整轻量证据和实际预测数组在`outputs/final_6000_20260926/`，包含训练日志、配置、checkpoint身份、所有32条逐层NPZ、预测哈希、服务器评价和本地独立复算。没有定时监督任务，也没有继续训练。
