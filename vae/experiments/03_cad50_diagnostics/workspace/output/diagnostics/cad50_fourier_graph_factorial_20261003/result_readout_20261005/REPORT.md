# CAD50 Fourier × Graph/LN结果

四格各2000次五mesh更新，每条CAD参与200次，固定seed0。训练已完成并按预算停止。

| 条件 | Edge FP/FN | 实际Face FP/FN | Edge micro-F1 | Face micro-F1 | 严格成功 | 困难16条 |
|---|---:|---:|---:|---:|---:|---:|
| Fourier_LN_post | 1913/4117 | 1395/4765 | 58.482512% | 20.842971% | 18/50 | 0/16 |
| Fourier_LN_pre | 1050/5792 | 316/5160 | 42.916736% | 13.189601% | 16/50 | 0/16 |
| XYZ_LN_post | 1233/5652 | 620/4944 | 44.065318% | 18.512009% | 9/50 | 0/16 |
| XYZ_LN_pre | 2160/5028 | 2460/4744 | 48.138528% | 18.764096% | 15/50 | 0/16 |

## 判读

Fourier＋后置LN的末尾Edge/Face F1与严格成功最多。在Fourier条件下，后置LN的Edge F1提高15.57个百分点；在XYZ条件下，后置LN的Edge F1降低4.07个百分点。方向随坐标编码改变，存在明显的观察性交互。

严格成功数的变化同样有交互：Fourier下后置LN比前置多2条；XYZ下后置比前置少6条。后置LN下Fourier多9条，前置LN下Fourier只多1条。不能将效果拆成一个对所有条件都成立的根因。

较少FP可能伴随更多FN。例如XYZ＋后置LN的Face FP只有620，但FN为4944；XYZ＋前置LN的Face FP为2460，FN为4744。F1、错误分解和严格成功需一起看。

四格困难16条均0/16。最佳组合仍存在4117条漏边、4765个漏面，尚未完成CAD50严格overfit。单种子、有限预算CAD50对照不能确定原100条历史瓶颈的唯一根因。

XYZ＋后置LN在step1500通过19条，末尾下降至9条；XYZ＋前置LN在step1500通过20条，末尾为15条。不得将最好检查点或不同检查点成功UID拼成末尾成功。

## 验证范围与缺失材料

本轮成功从服务器读取DONE、四支完成记录、配置、四格末尾完整评价汇总、所有评价轨迹和旧控制step0复现gate。表中F1由已读取的TP/FP/FN在本地重新计算，算术一致。

SSH在下载之前被服务器关闭，随后重连均在握手阶段关闭。本包不包含原始updates.jsonl、逐检查点逐UID JSON、预测NPZ或大模型权重。本轮未执行独立原始数组复算、checkpoint哈希重算或GPU推理。prepared audit_and_package.py只通过本地语法检查，尚未部署执行。

本包包含结果快照、轨迹CSV、比较图、已有配置/源码/启动与加速核验材料。大checkpoint的服务器路径、大小和报告记录SHA见result_snapshot_20261005.json；这次未重新核验SHA。provenance.json等早期准备材料中的pending/scope是历史快照，完成状态以本次结果快照为准。

## 单位与定义

micro-F1由全50条TP/FP/FN相加后计算。严格成功指同一checkpoint中一条mesh的Edge与从预测Edge图实际枚举的Face均FP=FN=0。LN pre表示聚合/投影前LN；LN post表示聚合与投影后LN；两种路径均固定GELU、V2其余架构与训练配方。XYZ-only保持39列投影及相同初始参数，36列Fourier输入置零。负例按epoch/UID确定性重采样，四支规则一致。
