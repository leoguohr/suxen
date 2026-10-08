# 四条 mesh：原 LR 与 10 倍 E/D/heads LR，400 步配对实验

两支均从 `math00_four_mesh_20260913/checkpoint-update1000.pt` 出发，完整恢复相同权重、五组 Adam 和训练 RNG。两支400步逐步使用完全相同的1600份 ε。唯一训练变量是 Encoder/μ、Decoder body、Edge head、Face head 的 LR；logvar LR、β、clip、数据、loss、backend 和其余配置不变。

| 分支 | E/μ LR | D/heads LR | logvar LR |
|---|---:|---:|---:|
| Control | 1e-8 | 1e-7 | 1e-4 |
| HighLR | 1e-7 | 1e-6 | 1e-4 |

## 结果

末尾 μ 路径：

| Mesh | 分支 | Edge F1 | Edge TP/FP/FN | Face F1 | Face TP/FP/FN |
|---|---|---:|---:|---:|---:|
| 原386点 | Control | 100% | 1152/0/0 | 100% | 768/0/0 |
| 原386点 | HighLR | 100% | 1152/0/0 | 100% | 768/0/0 |
| 原2575点 | Control | 100% | 7719/0/0 | 100% | 5146/0/0 |
| 原2575点 | HighLR | 100% | 7719/0/0 | 100% | 5146/0/0 |
| 新534点 | Control | 60.49% | 1312/1482/232 | 20.86% | 336/1861/688 |
| 新534点 | HighLR | **61.77%** | 1342/1459/202 | **22.47%** | 375/1939/649 |
| 新804点 | Control | 52.36% | 1534/1919/872 | 18.97% | 373/1956/1231 |
| 新804点 | HighLR | **53.79%** | 1568/1856/838 | **20.23%** | 388/1844/1216 |

HighLR 相对 Control 多恢复：534点 mesh 30条真边和39个真面；804点 mesh 34条真边和15个真面。534点 mesh 的 Face FP 同时多了78个，因此其 Face 提升来自召回增加，并非所有错误都改善。804点 mesh 的TP、FP、FN均更好。

两支都没有使新增两条严格全对，四条同时成功在末尾监控50组和额外新50组噪声中均为0/50。两支末尾原两条在这两套噪声中均50/50严格成功。HighLR 在中途 step10、step50 的 μ 检查曾让原2575点 mesh 漏1个面；step20及step100后恢复。Control所有检查点均保持旧两条严格成功。

按补充协议，从同一父checkpoint和RNG精确重放两支前100步，并增加step100的50组噪声验收。重放的每一步loss、KL、梯度、clip、参数更新及ε哈希均与原训练记录一致。Control在step100为旧两条50/50；HighLR为49/50。失败的是监控pair16（seeds `9830032/9830033/9830132/9830133`），原2575点mesh产生1个Face FP。此时HighLR的μ和原固定ε仍全对，说明单次确定性/固定噪声检查会漏掉这次噪声稳定性退化。

同一50组监控噪声上的 reconstruction Soft4：共同起点0.529687，Control末尾0.513635，HighLR末尾0.507080。相对起点分别下降约3.03%和4.27%；HighLR末尾只比Control低约1.28%。HighLR确实更快，但远未达到名义10倍。

## 学到了什么

HighLR 的优势是真实的，但很小。新增534点 mesh 相比父checkpoint增加96条真边、99个真面；804点增加78条真边、61个真面，因此不只是删除预测来提高F1。与此同时，最终仍有大量漏边和漏面，不能说新拓扑已被记住。

漏面的主要上游限制仍是预测边图。末尾534点 mesh 的Face FN中，Control有386/688、HighLR有349/649个因为边图中凑不出GT三角形；804点分别为1026/1231、991/1216。HighLR主要多恢复了一部分边结构，Face head本身也仍有约300和225个候选存在但判负的问题。

## 为什么10倍LR没有带来10倍速度

Control只有105/400步触发global clip，clip系数中位数为1；HighLR为400/400步触发，系数中位数仅0.0611。HighLR的裁剪前总梯度范数中位数16.35、最大605.14；Control中位数0.879、最大3.15。HighLR loss标准差约0.00946，是Control 0.00459的约2.1倍，并多次出现尖峰。

因此实际参数步长远小于名义10倍。每步相对参数更新中位数的 HighLR/Control 比值约为：Encoder 3.79倍、Decoder 3.52倍、Edge head 1.43倍、Face head 3.53倍。更关键的是，虽然 logvar 名义LR完全相同，它的实际更新反而只有Control的0.295倍，因为global clip会统一缩放所有组。

最终 KL 也体现了这一点：Control为8.80437，HighLR为8.88720。Kμ几乎一致，差异主要在Kσ；HighLR对logvar的有效更新被更强裁剪压小。因此本实验不是“仅让E/D快10倍而posterior完全不变”的实际轨迹，尽管配置层面logvar LR没有改变。

## 判定

在当前 global clip=1 的联合训练中，把 E/D/heads LR 同时提高10倍只能带来约1至2个百分点的新mesh F1优势，并引入明显震荡、持续强裁剪和短暂旧样本错误。它不是理想的直接主线配置。Control更稳但学习太慢；这次对照说明后续若要提高学习速度，应处理“学习率与global clipping共同决定的实际更新尺度”，不能继续只按名义LR倍数加速。

本轮只完成用户指定的两支400步实验，没有继续改clip、扫中间LR或延长训练。

## 验证和产物

只读核验确认：两支父权重、Adam moments、训练RNG完全一致；每步ε哈希完全配对；四mesh等权reconstruction；五组每步均有非零参数更新；sampling扰动每步非零；末尾Adam step为E/D/heads 2200、logvar 2000。step100重放的核心训练字段与原日志逐项相等，补充验收没有改变原400步分支。

服务器checkpoint：

- Control: `/guohaoran/nexus_fast_track/diagnostics/math00_four_mesh_lr10x_20260913/control/checkpoint-update0400.pt`，SHA256 `83c70b93188c985525ec1a4343cedf92271ce06381aa5189e0ff1191081dd0a9`
- HighLR: `/guohaoran/nexus_fast_track/diagnostics/math00_four_mesh_lr10x_20260913/high10x/checkpoint-update0400.pt`，SHA256 `c814e2c878e48446b494fede8fde99b3d7d1edefee86c17c15711ea8c4ed7ced`

大型checkpoint和实际重建NPZ保留服务器；本地保存完整JSON/JSONL日志、核验、分析代码和曲线。
