# 100条末端联合训练：与已有Edge-only配对对照

已完成500次新增联合更新并按预算停止。从旧71条成功完整模型分叉；只训练Decoder最后一个block、最终LayerNorm、Edge head和Face head。Encoder、mu、Decoder前15块和logvar全部冻结。
原100条每次完整累积后更新一次；目标为mean100(Edge Soft4 + Face Soft4)，两项内部固定除4、fully-differentiable，不额外加权。原两轮Face pool不变、mu路径、KL=0、clip=1、wd=0。LR分别为末端1e-5、Edge1e-4、Face1e-4。
末端与Edge Adam恢复到与Control相同的step500；Face新增组恢复共同71条起点对应Face-recovery step500 Adam。全部三组末尾Adam step1000，不重新warmup。Face解冻及其既有状态是本次策略的一部分，不是单独的梯度冲突证明。
使用同一缓存末块输入，逐mesh顺序与Control一致；起点100条真实网络与缓存联合loss/梯度逐位核对，训练前重复完整梯度逐位一致。末尾回到真实输入完成100条复核。实际Face在每个检查点重新从预测Edge图枚举，无超时截断。

| 新增更新 | Control Edge/联合成功 | Joint Edge/联合成功 | Control Edge FP/FN | Joint Edge FP/FN | Control Face FP/FN | Joint Face FP/FN |
|---:|---|---|---|---|---|---|
| 0 | 71/71 | 71/71 | 156784/1 | 156784/1 | 49411/984 | 49411/984 |
| 50 | 72/69 | 71/71 | 158754/1 | 141188/1 | 59367/738 | 25183/1352 |
| 100 | 72/68 | 71/71 | 159441/1 | 133635/1 | 67753/607 | 15124/1086 |
| 200 | 72/64 | 71/71 | 154475/1 | 127394/1 | 78735/587 | 10239/582 |
| 300 | 73/63 | 71/71 | 148671/1 | 123583/1 | 87538/650 | 8754/433 |
| 400 | 74/58 | 71/71 | 138134/1 | 118264/1 | 92132/744 | 7340/459 |
| 500 | 75/55 | 71/71 | 130832/1 | 117054/1 | 98872/744 | 7537/336 |

Joint末尾71/100联合严格成功、71/100 Edge严格成功。原71条联合成功保留71、丢失0、新增0。后期300/400/500共同成功集合为71条。

## 预先固定的大mesh与失败样本组

| 组 | 条数 | Control末尾Edge FP/FN | Joint末尾Edge FP/FN | Control末尾联合成功 | Joint末尾联合成功 |
|---|---:|---|---|---:|---:|
| all100 | 100 | 130832/1 | 117054/1 | 55 | 71 |
| initial_failed29 | 29 | 130832/1 | 117054/1 | 0 | 0 |
| N_gt1500 | 33 | 129170/1 | 113846/1 | 2 | 7 |
| initial_success71 | 71 | 0/0 | 0/0 | 55 | 71 |

## 判读与材料范围

比较相同步数的Edge/实际Face FP/FN与UID保持。Control只有Edge目标，不能以新联合total loss对Control Edge loss直接排名。旧74条模型还包含另加500次Face恢复，其预算多一段，单独保留且不作为本轮等预算Control。
有限预算的成功或失败不单独证明梯度冲突或容量不可能。本轮没有自动追加、刷新pool、改变架构或扩大解冻范围。

- run/updates.jsonl：500条更新、三组LR、四模块实际位移、分组梯度和clip；edge_trace.jsonl包含501×100条Edge/Face loss与Edge计数。
- run/actual-new*.json、actual_evaluations.jsonl：700条完整实际Edge/Face、候选覆盖、training pool计数和margin。
- paired_comparison.csv、paired_per_mesh.csv、paired_groups.csv：逐检查点、逐UID、预先固定组的同预算对照。retention.json保存成功UID保留/丢失/新增。
- restore_verification.json、cache_gradient_verification.json、benchmark.json、run/verification.json与independent_audit.json：源/Adam/RNG、真实与缓存梯度、冻结范围、末尾重载验证及84,669,234个Edge logits独立计数复算。
- runtime.py、prior_core.py、face_core.py、effective_code与runtime_dependencies：实际执行公式与动态替换依赖；protocol_changes.diff对应旧Edge-only执行入口的改动。prior_core历史train/export入口不在本轮调用。
- Review含日志、逐mesh评价、实际代码、初末尾末端/双head权重与Adam/RNG；FullEvidence另含缓存末块输入、末尾hidden与两个embedding、全部Edge logits、中间checkpoint与固定Face pool。
- 最终实际Face逐候选logits未单独导出；已保留完整计数、评分表示、固定pool和实际枚举实现。训练pool loss不等于实际Face全候选loss。
- 巨大全网络权重不放ZIP；EXCLUDED_FILES.json提供服务器路径与SHA。旧71/74成功模型与Control均完整保留。

源71条模型：`/guohaoran/nexus_fast_track/diagnostics/face_head_recovery_fixed100_20260918/run/model-tail500-face500-inference.pt`，SHA `8de6074700fafa02b3adcfafd352a4555ce0cbc88421667261f29495faa69640`。
新联合模型：`/guohaoran/nexus_fast_track/diagnostics/decoder_tail_joint_fixed100_20260919/run/model-joint500-tail1000-face1000-inference.pt`，SHA `5a36ab88ee147ef160945cfcec11d1d5cb98026e20c536ce602f6303f1448f7e`。
