# 20条完整mesh长训练任务（待启动）

本次只准备训练任务，不启动训练、不执行optimizer.step。首次训练执行前完成下列准备检查；不得把本文件、配置文件或历史GPU验证视为新的20条训练入口已经通过验证。

## 推荐配置

从四条共同训练累计4400步的continue1000/checkpoint-update1000.pt开始，新增10000次实际更新。恢复其完整权重、五组Adam和训练RNG；不要使用804-only末尾，也不回到随机初始化。保持E/μ LR=3e-8、Decoder/Edge/Face heads LR=3e-7、logvar LR=1e-4，β=1e-4、wd=0、global clip=1、clamp[-20,10]、math00、fully-differentiable Edge/Face Soft4。全部五组可训练，采样每步更新。

这是目前已有训练证据支持的保守起点，不是20条上的最优参数结论。10000步是预算上限，不保证严格重建；不得自动追加训练。新16条尚未在当前模型上学会；四条末尾也不是四条严格全对的checkpoint。

## 固定数据与每次更新定义

使用selection.json列出的原20条，保留原UID顺序；不是重新抽20条。总顶点22093，最大4999，全部无向顶点对28659713。逐条核对实际顶点、面和标签哈希，不仅核对名称。已有四条candidate pool复用原文件，其余16条按相同归档生成规则准备；记录来源和SHA256，不换成随机小样本负例。

每个optimizer update包含20条完整mesh，目标为sum_i(Edge_i+Face_i+1e-4*KL_i)/20。每条Soft4与KL内部归约保持现有定义。按完整mesh分成固定microbatch前向/反向，所有贡献均先除20；仅在20条梯度累积完成后做一次global clip和一次Adam step。禁止每个microbatch单独clip/step，禁止对子图或候选进行随机抽样以节省显存。

microbatch边界需要通过20条实际数据的显存预检查后写入配置。它是相对四条单次packed流程的执行变化，不能声称已验证与20条同时packed逐位一致；必须保留mesh隔离和相同目标，先检查小规模的目标与梯度累积一致性。若完整4999点单条仍超显存或实际Face候选爆炸，报告具体阻塞，不截断候选并伪称完成。

每次更新每条mesh一份新ε。使用恢复的独立训练Generator，固定UID顺序推进；同一步forward与activation recompute复用对应ε。监控噪声独立于训练RNG，不推进训练流。数据从4增加到20后，不要求每步ε与旧四条分支匹配。

## 启动前检查

1. 核对父checkpoint SHA、全部模型参数、五组Adam状态及RNG。源代码保持冻结快照，记录backend/loss/model文件哈希。
2. 生成并核验所有训练candidate pools；保持原四条pool不变。
3. 用真实20条做无optimizer.step的显存/耗时预检查，确定microbatch方案；检查所有20条均有非零采样扰动和有效梯度。预检查后恢复模型及所有RNG，避免消耗正式起点。
4. step0验收逐条μ和固定ε，并建立50组监控噪声基线。新样本不正确属正常起点，不是旧样本遗忘。
5. 给出实际显存和时间估计，之后才具备启动条件。不得沿用四条单步时间估计20条训练时长。

## 保存与验收

- 每步：逐mesh Edge/Face Soft4、KL均值/方差分解、总objective、各组梯度、全局clip系数、实际参数更新、σ/clamp统计、训练ε哈希及RNG状态哈希、耗时。
- 每200步：完整权重、五组Adam、RNG；全部20条μ和固定诊断ε的实际Edge/Face TP/FP/FN/F1、GT Face候选缺失数及最小margin。
- step0/1000/2000/4000/6000/8000/10000：同一50组监控噪声。
- 末尾：另外50组训练和监控未使用的新噪声。

实际Face必须来自当次预测Edge图重新枚举，threshold=0；不能用training pool替代实际重建。按mesh报告进展，同时报告原两条、原四条以及20条整体严格成功次数。所有mesh成功必须来自同一checkpoint和同一组20份噪声条件，不能跨评估拼接。若分microbatch评估，明确记录其布局，不写成一次20条packed forward。

step1000/2000复盘学习速度；若有限loss正常但真边/真面恢复很慢，先报告，不自行调LR或恢复数值诊断。出现NaN/Inf或损坏时停止保存故障信息；正常未全对不自动判定容量失败。末尾如实报告剩余FP/FN及严格成功比例，不用训练loss下降替代过拟合验收。

## 交付

已补齐服务器train.py、start.sh、prepare.py、20条候选池及不更新参数的GPU预检查。启动方法以README_START.md为准；实测报告见preflight/result.json。正式训练未启动、未创建定时任务。后续应提供完整日志压缩包、loss/结构错误趋势及checkpoint路径和SHA256。
