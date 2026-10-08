# Phase3-A 最终结果

新增 2000 更新已完成，累计 step 14000。最终完整树评估已完成，CPU 数组复核通过。
本轮加权续训没有达到改善目标；未自动续训。

| 指标 | Phase2 step12000 | Phase3-A step14000 |
|---|---:|---:|
| seen 完整树 | 16/40（40%） | 7/40（17.5%） |
| unseen 完整树 | 0/8 | 0/8 |
| seen 首错 depth2–5 | 19/40 | 20/40 |
| unseen 首错 depth2 | 8/8 | 8/8 |
| 共同父格条件切换：两侧都正确 | 21/45 | 15/45 |

同一 UID、同一 seed 配对：原来成功的 16 条中，9 条退步；原来失败的 24 条没有新增成功。
seen 第2层精确恢复 36/40 → 32/40，第5层 21/40 → 20/40。
细层平均指标并非全面下降：第15层 occupancy F1 从 0.72096 升至 0.79387，但完整树恢复下降。
因此不能以平均 loss、F1 或点数替代完整树验收。没有等预算未加权续训对照，不能单独断言是权重引起退步。

训练日志共2000条、16000份microbatch，150个“物体×深度”组合分别参与106或107次。
CPU复核确认粗层 raw velocity MSE ×1.5、细层 ×0.75；固定归一化4/3。
沿用 D15、lr=1e-5、Adam、8次累积、10 seen/2 unseen、固定点云法向。
评估沿用 seeds 94026000–94026003、20 Euler steps/layer、阈值0.5、BF16、自生成父格。
这些是已用过的固定回归种子，不是新的终验种子。

## 复核与错误处理

新服务器检查时已无本任务训练进程。训练和GPU生成在此前实例完成；本轮收尾只用CPU。
原流水线在CPU汇总时因Python导入路径缺失而停止；完整48条生成结果已保存。
补上PYTHONPATH后运行原比较脚本，未修改训练、模型、采样器或预测。
原报错日志和原失败状态保留在audit，CPU恢复命令见COMMANDS.md。
复核Phase2和Phase3共96条树、1440层数组的哈希、父子连接、阈值/bit解码、预测点数/mask、XYZ和顺序；复算逐层与完整树结果。
最终大权重在新实例重新完整SHA256读取并CPU mmap检查：907个model条目、907份Adam状态、全部Adam step14000，scheduler=None，RNG完整、下一micro位置100。
训练冻结代码哈希和部署文件哈希核验通过。

## 怎么读包

- comparison/：前后每样本每层、逐层汇总、逐物体、配对完整树、首错与XYZ诊断。
- post_training_final_014000/：48条真实预测、720层噪声/连续占据/parents/子格数组、prediction.npz及原始评价。
- baseline_phase2/：对应Phase2预测、标签、条件、代码和配置，可在同一包内复算。
- run/：完整训练日志、配置、checkpoint身份；runtime/：实际运行代码、固定点云法向、浮点GT与D15标签。
- train_git_diff.patch、eval_git_diff.patch：相对前阶段的代码差异；源码未修改。
- final_checkpoint_verification.json：本轮CPU完整权重校验；FILES_SHA256.json：包内所有载荷文件哈希。

occupancy accuracy 分母为 GT parents 与预测 parents 并集的8个子格；F1为2TP/(2TP+FP+FN)。
XYZ诊断在原包围盒最长边2的空间计算；保留原始浮点GT（包括重复点）和显式去重诊断。
等点数coordinate RMSE遵循已有评价器匹配；点数不等时该项为空，不删除预测或GT凑点数。
逐样本summary同时保留双向平方Chamfer：两个方向最近邻平方距离均值之和。
共同父格45对结果来自已保存的实际模型评价JSON；原始probe连续张量未落盘，不能声称本次独立复算了这些张量。
不含checkpoint/Adam大张量，也不含未取得的完整原始mesh面文件；本包含实际使用的固定点云、法向、GT和预测。
这不是CAD50生成成绩或老师模型对比。

下一步：保留Phase2 step12000作为基线，本轮记为未改善的实验；不继续放大粗层权重或自动追加训练。

最终checkpoint SHA256：`35beb183ef09668f4efc5ba3bb5c1949fbca24ca157e1fc31abef8177a26eaa5`

服务器权重路径：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/phase3_coarse_weight_20260928/run/checkpoint-014000.pt`
