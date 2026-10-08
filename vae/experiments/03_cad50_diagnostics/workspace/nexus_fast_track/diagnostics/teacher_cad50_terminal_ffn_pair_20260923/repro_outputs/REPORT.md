# 原512 AE末端FFN对照：完成报告

从同一CAD B2500完整训练状态出发、各比较100次全50条更新，末端新增FFN没有胜过原版Soft4对照：实际Face F1为0.352533（对照0.353959），Edge FP减少58但FN增加66、合计错误增加8；困难16条Face F1为0.298711（对照0.302023），两支困难样本严格成功均为0/16；全50联合严格成功为32（对照33）。本轮不支持“只在Decoder末尾加一个FFN就能解决过拟合困难”，也不能据此否定所有FFN结构或断定网络容量上限。

## 做了什么

老师小AE的Decoder具有attention+FFN，原512 Decoder为16层attention残差块。本轮在原网络最后attention残差后、原decoder_output_norm前增加一个pre-LN FFN残差：1024→4096→GELU→1024。原Encoder已有FFN，不能概括成“原网络没有FFN”。新增8,395,776参数，末Linear零初始化，初始输出与原模型完全相同。

两支固定同一B2500父checkpoint、旧model/Adam/RNG及数据顺序。H的新FFN使用独立fresh Adam组，其余旧状态完整继承；完整旧Encoder/μ/Decoder/head继续训练。保留fully-differentiable Soft4、latent512、μ路径、KL=0、sampling关闭、固定Face pool、原学习率/clip/FP32 MATH路径。没有复用可训练层缓存代替网络训练。

原版Control已有合格100步结果，本轮核验后复用，新增0步；H实际新增100步，止于累计2600。新增FFN参与统一global clip=1，所以初始旧参数裁剪前梯度一致不意味着裁剪后位移必须一致。没有训练原固定100条，没有继续500步或调参。

## 同预算终点

| 指标 | 共同起点B2500 | 原版Control +100 | 末端FFN H +100 |
|---|---:|---:|---:|
| 实际Face micro-F1 | 0.3501810501 | 0.3539593250 | 0.3525331725 |
| Face TP / FP / FN | 2321 / 5359 / 3255 | 2454 / 5836 / 3122 | 2338 / 5350 / 3238 |
| Edge micro-F1 | 0.6631407246 | 0.6634849518 | 0.6610485056 |
| Edge TP / FP / FN | 6708 / 5159 / 1656 | 6812 / 5358 / 1552 | 6746 / 5300 / 1618 |
| 联合严格成功 | 32/50 | 33/50 | 32/50 |
| 困难16条Face F1 | 0.2957051177 | 0.3020228948 | 0.2987108958 |
| 困难16条严格成功 | 0/16 | 0/16 | 0/16 |

H的Face FP较少，但FN更多、TP更少，最终F1反而略低，不能只用FP下降或总错误数下降宣布获胜。H对起点的Face F1提高0.002352，低于Control的0.003778。两支均未达到用户采用的0.997 Face F1门槛；严格50/50也未达成。

## 轨迹和覆盖

| 新增更新 | Control Face F1 | H Face F1 | Control严格 | H严格 |
|---:|---:|---:|---:|---:|
| 0 | 0.3501810501 | 0.3501810501 | 32 | 32 |
| 25 | 0.3248545528 | 0.3397753436 | 32 | 33 |
| 50 | 0.3480188814 | 0.3480911598 | 31 | 32 |
| 75 | 0.3261747821 | 0.3434296724 | 30 | 32 |
| 100 | 0.3539593250 | 0.3525331725 | 33 | 32 |

H在25步新增成功teacher_cad50_34，50步后不再严格成功；末尾仍是父点原32条。Control末尾保留父32条并新增teacher_cad50_34。H最高严格checkpoint为25步33/50；最高Face F1与最终checkpoint均为100步。它们各有独立指针与哈希，不能拼接UID。

H在五次完整验收中保留父32条，但逐步更新前的Edge成功数仍在30—33之间。H有15个更新前观测曾丢失父Edge成功UID，Control有18个；涉及teacher_cad50_02、teacher_cad50_24。这些是逐步Edge观测，不是逐步实际Face验收，不能说“全程保住父32条”。完整逐步记录见 `step_edge.csv` 与两个分支的 `updates.jsonl`。

## 错误仍在哪里

H最终3238个Face FN中，2950来自GT Face没有进入当前预测Edge图的三角候选，288来自候选内判负；5350个Face FP中，226在训练基础pool内、5124在pool外。对照分别为缺候选2876、候选内判负246、池内FP268、池外FP5568。H减少了池外误报，却增加了缺候选和有候选漏判。

困难16条在H末尾仍有Edge FP/FN=5286/1615、Face FP/FN=5312/3229。这个末端结构没有让困难样本明显变得可分。记录描述了错误位置，尚不足以把成因唯一归结为Encoder、Decoder深度、坐标编码或loss。

原固定100历史审计则是另一种局面：Edge FN=0、GT Face缺候选=0，主要错误集中于较大mesh的Edge FP和pool外Face FP；最近last2/last3均73，历史最佳74。不能把CAD50的“缺边导致缺面”直接套到原100。原100本轮没有训练更新或新的成功纪录。

## 执行与独立核验

1. 已安装ai-research-reproduction被实际加载；版本/路径/哈希见 `SKILL_LOAD.json`。README intake和run-train生命周期均有记录。没有重新安装工具或升级训练环境。
2. 本轮CPU重新审计复用Control：完整父状态、328个旧可训练参数映射、数据/pool与代码哈希、100次更新、五次全50评价、250份预测及完整候选枚举通过。原始记录和本次事后核验分开保存。
3. H预检确认全50初始预测逐字节一致；旧Adam/RNG精确继承；真实网络UID20旧参数梯度与未扩展网络和Control设备记录一致。新增FFN首步out_proj有梯度、内层为0；第二步内层获得梯度。真实实验放行前0更新。
4. H100次日志、五个完整checkpoint、250预测、有限数、学习率、Adam计数、冻结logvar、错误分解与候选完整性经独立CPU审计通过。旧组最终Adam2600，新组100。
5. 在另一台单卡A100的新进程中，从最终完整model/Adam/RNG冷加载并重新安装FFN hook，全50真实Encoder→μ→Decoder预测与末尾数组逐字节一致；冷验0更新。最终推理权重与可续训checkpoint模型张量相同。
6. run-train正常结束，returncode=0，未超时。H逐步计算用时合计573.21秒；Control旧记录为580.09秒。两者均不含所有预检、保存和等待，不能把单次差值当可靠速度benchmark。服务器其他旧实验未改动。

初次CPU toy测试因protobuf/onnx导入兼容失败，采用原运行环境已有的protobuf纯Python设置后通过。最终CPU审计第一次将state_dict中非Tensor元数据误作Tensor检查，修正审计脚本后通过；错误尝试记录保留。这两个是工具/审计过程错误，均没有修改训练协议或推进额外真实训练步数。真实GPU训练无失败或恢复。

## 身份与交付

启动训练代码commit为 `866d0a88fb4a032d2971bccfd1b20fbd513c26dc`。启动文件SHA见 `CODE_HASHES.json` 和H的 `config.json`；最终交付Git包含后处理和报告，启动版本保留可追溯。主README原件未改，CPU toy更新措辞更正见 `ANNOTATED_README.md`。

父checkpoint SHA256：`4c67709dcd3bacb6a09181b034512469b1e7f38463f1aa7b41affc8bcf435a66`。

H最终完整可续训文件：`/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923/H_terminal_ffn/checkpoint-new0100-step2600.pt`，1,448,830,926 bytes，SHA256 `ca0456d3921ae5d807b1d0e28505551a9740f640b6bb487b627b65aaf2891a73`。完整13项清单含父、两支五个checkpoint和推理权重，见 `CHECKPOINT_MANIFEST.json`。

新模块通过hook执行，加载时必须使用 `terminal_ffn.py::load_extended_state` 或等价显式安装流程；仅把新增张量塞进旧类并不能执行FFN。冷验脚本提供可核验入口。

交付包含有效代码/diff/Git bundle、配置命令、原始训练与监督器日志、500份主评价预测加50份冷验预测、逐mesh与逐步CSV、模型哈希，以及本次原始ZIP逆向报告和代码。大权重、原始数据与原ZIP留原位置；不重复打包历史模型，不含凭据。

老师原ZIP逆向结论与昨日候选、原512的比较见 `REVERSE_ENGINEERING_REPORT.md`。老师原AE权重Face F1约0.9991025；此前小AE从零20k实验约0.990538。恢复一个能解释已存权重的前向网络，不等于获得老师原始训练方法，因此两者成绩差异不能直接判定逆向失败。

本次skill实际帮助固定父状态和比较协议、复用可核验Control、记录训练生命周期并组织独立复算与冷验。新证据是：FFN初始恒等和有效梯度路径成立，但同预算终点未改善实际Face F1、困难CAD和严格覆盖。没有发现被此次核验明确证明的原训练实现bug。本轮已停止，不追加训练；原100根因仍未被这一项干预解决。
