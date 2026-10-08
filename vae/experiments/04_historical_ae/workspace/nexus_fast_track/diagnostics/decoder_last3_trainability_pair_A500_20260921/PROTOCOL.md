# A500：Decoder最后两块与最后三块可训练性对照

本实验由2026-09-21用户对下一步的修订定义。目标是有限预算内比较训练范围，不宣称冻结过多是根因，也不增加网络层数。原16层网络保持16层。

- 两支都从A500完整权重 `b0b31c3fcef870f303c6d808eb3468b6c70e5eb3313c039fdcbfc34632bede3e` 和Adam/RNG checkpoint `8268d06c33820da22291ff433ee60fc50510c5e6ea19c6f26a073fb5b97bd952` 独立恢复。
- Control：block14、15+原末端LN、共享Edge/Face head。Treatment：只多开放已有block13；原4个Adam组完整继承，只有新增组以空moment/step0开始，LR1e-5。
- 每支500次新增全100条累积更新；原目标四分量均除100，无B降权；其余数据、pool、μ/KL/logvar、评分、阈值和math00不变。
- 同一block13输入缓存，两支都经过block13前向；0步对两支全部100条验证完整网络/缓存hidden、两种embedding、logit、loss和开放参数梯度逐位一致，且A500实际100条预测数组与保存结果逐项一致，才能训练。预检不执行optimizer.step。
- 每个更新后状态0..500都有 `step_records.jsonl`：Edge成功数、成功UID、FP/FN、相对A500及上一更新丢失/新增UID、四个未加权loss、逐UID记录；状态k与产生它的第k步实际参数位移和clip绑定。`updates.jsonl`记录每个实际optimizer.step。缓存路径每步计数，完整网络及Face于0/100/200/300/400/500验收。
- 原组Adam起点为tail/Edge/Face2000、block14为1000；Treatment block13从0起。终点应为2500/1500/500。每支每个UID正好新增500次参与。
- 每步保存自身 `latest.pt`，预定检查点另存；禁止根据checkpoint文件名猜更新数。遇中断保留failure state并停止，没有隐式重跑或恢复。
- 完整Face使用当前预测Edge图的全部triangle/clique；logit>0为正；缺候选GT计FN。同一共享checkpoint的联合成功超过72为本分支进展，超过74为项目新纪录，100/100才是主目标完成。
- 预注册首要比较：最终step500 Treatment联合严格成功数同时超过Control500和72，并保留A500全部72成功UID。中途checkpoint、loss下降及不同checkpoint成功集合并不替代这一判据。
- 非有限数、恢复或哈希/冻结状态/完整枚举失败则停止并保留差异；资源不足停，不换后端、不抢占CAD GPU、不补预算。

只使用新分配机器上的空闲A100 80GB（UUID见config）。运行来源代码、动态导入helper、实际argv、显式环境、GPU身份、日志和生命周期保存在repro_outputs。新实验目录独立Git管理；不改动历史工程入口、pool或checkpoint。
