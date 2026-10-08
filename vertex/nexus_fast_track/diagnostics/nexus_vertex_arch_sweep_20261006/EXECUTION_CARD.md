# NEXUS Vertex 七组结构筛查执行卡

- 日期：2026-10-06，Asia/Shanghai。
- 用户授权：S0–S3四套，加三套有源码依据的完整适配组合S4–S6；每组新增10,000次optimizer更新；两张A100 80GB尽量提高有效显存利用；启动核验后收尾，不设持续监督。
- 本轮是从已有权重出发的结构适配筛查。组合收益不能直接归因于其中单个改动，也不代表从随机初始化训练的架构排名。

## 结构与共同起点

- 七套结构及固定源码依据见 `ARCHITECTURE_CARD.md`。
- 所有组独立从A24000出发，累计步数记为24,000＋本组更新数，终点34,000；不得接续其他分支。
- 来源：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/experiment/A/checkpoint-024000.pt`。
- SHA256：`2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`。本轮已重新读取整文件核对；文件27,990,364,852字节。
- 源模型907个参数张量，Adam状态907项；step24,000；scheduler为null；Python、NumPy、CPU Torch、CUDA RNG均存在。
- 同名同形参数及Adam状态原样继承。新增参数与明确变形参数单列初始化/状态重置；移除参数单列。各组 `startup.json` 内的 `restore_audit` 包含参数覆盖、状态计数与实际张量值一致性核验。
- S1新增末层调制采用零shift/scale，在迁移起点保留原前向。其他改变归一化、query、分头或坐标编码的组不宣称前向等价。

## 数据与更新语义

- 原50个NEXUS2K UID，固定8192点＋法向，原归一化与D9标签；使用 `nexus_d9_ab_20260930/prepared_d9/manifest.json` 并校验每个条件/标签文件哈希。
- 全部depth1–9；每次更新全局8个micro事件。原UID-major对象×层级顺序，所有组从新sweep cursor=0开始，使用完全相同的新事件表及独立于旧训练的噪声/time命名空间；来源cursor=16000只记录于provenance。
- 每组80,000个训练事件。450个对象×depth组合每项得到177或178次曝光，具体计数落盘；不声称每项次数完全相等。分支间逐事件完全一致。
- velocity MSE，先对每个mesh的有效父格×8维求均值，再对8个事件等权求均值。padding不参与loss，不由长序列改变对象权重。
- Adam，lr=1e-5，weight_decay=0，betas=(0.9,0.999)，eps=1e-8；全局梯度裁剪1。BF16前向，FP32参数与Adam。保持源优化器其他语义。
- microgroup只决定同时计算多少事件，取1/2/4/8；全局分母与事件顺序始终是8。批处理可能有浮点舍入差异，需先做小模型逐事件loss/梯度等价检查。
- 所有组预处理可缓存原始Fourier/FPS；可学习VecSet前向及梯度必须实时执行。S3与S5不得沿用不匹配的旧query/Fourier缓存。

## 两卡执行与显存

- GPU0：`GPU-80f199d2-afab-fad3-824d-6d2482a4c882`。
- GPU1：`GPU-0634fd68-4a4d-facb-1b7f-0f9789679b47`。
- 预检时两卡均无计算进程。队列每卡一组，空闲后取下一组；启动前重新检查GPU进程与任务锁。
- 优先关闭梯度重算，使用真实microgroup增加吞吐；根据独立启动预检固定安全group大小。记录allocated/reserved峰值；不通过无效张量填充显存。
- 目标约70GiB/卡，实际占用受父格长度与分支结构影响。若达到该数字反而降低吞吐或导致OOM，以正确更新和稳定吞吐为先。

## 保存 评价与后台完成

- 首次实际更新后保存并读回一个完整checkpoint，随后每200更新保存；仅在完整optimizer update后保存。
- 完整状态包含模型/Adam、scheduler、RNG、branch_update、累计step、下一事件位置、样本depth曝光、数据/源码/config哈希。
- 原子临时文件写入、flush/fsync、替换、SHA256、读回检查成功后才更新checkpoint身份。每组最多保留最近两份完整恢复状态；最终状态作为正式终点保留。
- 仅删除本轮同组已经被更新恢复点替代的文件；原A24000、A–G、E2、VAE/AE文件保留。
- 不做训练中生成评价。10,000更新完成后，执行原50对象×2配对种子、D9从根自由展开、20 Euler步/层、阈值0.5；不加GT点数、GT parents或修补规则。
- 终点评价记录完整树exact、逐层集合/F1/FP/FN、首错depth、预测整数顶点、可适用的XYZ指标、实际噪声/轨迹与checkpoint身份。旧配对种子不称作未见终验集。
- 七组全部完成后自动汇总并打轻量证据ZIP，不包含大权重与任何认证信息。
- 非有限、OOM、保存失败、身份不一致或异常退出：保留证据，失败组不自动重试或改配方，停止启动后续排队组；已经运行的另一组可完成。
- 根任务仅完成部署、真实首步与保存链路核验后退出；不创建心跳、定时通知或持续监督。

## 已有证据边界

- A/F完整树均0/100，F局部条件响应改善尚未转成整体收益；S6与F不同，明确属于组合筛查。
- 本轮连接服务器确认E2的C/N/T三支已完成且队列证据门禁通过。T终点depth9、GT父格20步Euler的集合exact为39/100、micro F1约0.990016；这是单层诊断，不能记作D9完整生成。
- 原模型仍具学习能力，因此结构变更不预先称为必要修复。同期S0用于区分额外训练历史与结构适配的收益。

## 实际执行证据

- 远程独立目录：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_vertex_arch_sweep_20261006`。
- 本地同名目录保存代码、执行卡、预检和启动证据；运行命令、实际group与进程身份在部署完成后补入 `STARTED.md`。

## 完整模型预检结果

- 实际运行S0与S5，最长2528个parents，每次update为8个独立噪声事件，microgroup=2，关闭梯度重算。
- S0：首步3.385秒，张量峰值65.576GiB，allocator预留峰值67.242GiB。
- S5：首步3.644秒，张量峰值66.167GiB，allocator预留峰值67.814GiB。
- 两组均完成约28GB完整checkpoint保存、SHA256及全部模型/Adam/RNG/config/cursor/exposure逐值读回。
- 预检状态明确为`smoke_only`，不计入正式预算。正式七组统一microgroup=2，从各自A24000迁移起点开始。
- 环境兼容：当前容器的ONNX/protobuf导入冲突以进程环境变量`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`解决；没有安装或更换依赖。
