# NEXUS D9 A/E/F/G 一次性执行卡

本轮用户授权新增 E/F/G 各 2000 次 optimizer 更新，合计 6000；保留并复用旧 A。布置后不设置监督自动化，不盲目重试，不追加训练。

| 分支 | LR | WD | 全局条件 AdaLN | 调度 |
|---|---|---|---|---|
| A（已有） | 1e-5 | 0 | 无 | UID-major |
| E | 1e-4 | 0 | 无 | UID-major |
| F | 1e-5 | 0 | 有 | UID-major |
| G | 1e-4 | 0 | 有 | UID-major |

## 共同起点与新增通路

起点 `/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/migration/checkpoint-022000-d9.pt`。
SHA256 `5addf330bbe1a772a4a810181d5043f2fad5c7906a8c31d7c9dd7f9679471b83`。
旧 A 最终 SHA256 `2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`。

E/F/G 都从 step22000、d9_update=0 出发。源模型与原907项 Adam m/v/step22000 完整继承，恢复 RNG。只在 F/G 新增共享 Linear(2048,1536) 的 W/b，3147264 个标量、2个参数张量。W/b 全零，Adam 新状态从0开始，其余参数状态不重置。

有效 condition token 的均值 → 共享零投影 → 加入原 time embedding → 原36块 AdaLN。原 cross-attention、velocity head、VecSet、loss、采样器保留。不加坐标先验、offset、X1、归一化层或 VAE 改动。这是本地受控扩展，不标为论文确定结构。

## 执行和恢复语义

每个分支单独使用一张 A100，顺序执行8个 microbatch，各 loss/8 累积，然后全模型 clip1、一次 Adam（非 AdamW，foreach=False）。新增投影梯度参与同一次全局clip，不单独裁剪或更新。BF16前向、FP32参数与Adam，无activation recomputation；仅缓存不可训练的FPS/Fourier，不缓存detach的VecSet学习特征。

GPU0：E训练 → G训练 → E最终评估 → G最终评估。
GPU1：F训练 → F最终评估。

原A也是每分支单卡执行。本轮服务器物理GPU UUID变更，但A100型号、PyTorch精确版本、冻结源码和数据已核对；不声称跨设备/完整训练轨迹逐位一致。旧A历史训练没有本轮新增中间probe曲线，仅复用已保存最终冻结诊断，不能补造中间数据。

每分支16000个事件；UID×depth参与35/36次。时间与噪声完全沿用 `SHA256(20260930|UID|depth|occurrence)` 私有CUDA布局。训练结束逐条核对与旧A的实际UID、depth、occurrence、t、noise哈希。所有正式训练均重新加载共同起点，不继承预检更新。

## 保存与诊断

第1次和每200次完整保存 model、Adam、scheduler=None、step、d9_update、RNG、计数/下一事件及配置/数据/源码哈希；临时盘写入后拷入SSD并读回SHA核验，各分支保留最近两份。源 checkpoint 与 A 不删除。

每步loss日志；第1次/每200次记录模块参数RMS、裁剪后梯度RMS、实际参数更新RMS和条件尺度。每200次记录同一固定探针库（2144次前向），包括共享父格、同噪声条件切换；探针恢复训练模式/RNG，不是完整树生成验收。

## 预检与终验

预检：CPU小模型零通路/有效mask/新梯度/旧Adam继承，以及完整G模型depth1–9零投影相等、最大parents的8micro真实反传/一次丢弃更新、显存及状态检查。预检权重不写作训练checkpoint。

训练到2000后才做最终完整树评估；每分支50个已见UID × 2配对种子(97029000/97029001) × Euler/DPM，各200条。D9、每层20次网络调用、阈值0.5、容量4096，模型自行展开parents，不读GT点数/GT mask修补预测。

先落原生预测/哈希，再独立读取GT评价。输出完整树、逐层集合/occupancy、首错层、顶点F1、XYZ指标适用性及A/E/F/G配对差值。容量中止保留在分母，点数不符的匹配RMSE标不适用。不由loss或attention尺度单独宣布机制成立；仅筛查固定训练轨迹，不外推普遍规律。

后台完成后生成 `experiment/NEXUS_D9_AEFG_results.zip` 与 `delivery_identity.json`，含A及三组证据，排除大权重和凭据。失败记录实际异常并停止相应队列，不自动重启或改超参。

服务器根：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_efg_20261002`。
临时盘：`/tmp/nexus_d9_efg_20261002`。
启动记录 `audit/launch.json`，实时阶段 `experiment/audit/phase.json` 与每分支 train/evaluate JSON；本卡不代表已启动或训练已通过。

## 已核验启动

2026-10-01 22:45:21 CST 实际核验：控制器 PID664；E/G 队列 GPU0，E PID674 已到19次更新；F GPU1、PID675 已到16次更新；G排队。两组首份完整 step22001 checkpoint 已在SSD成功保存并读回SHA核验。训练未完成，本次到此结束启动核验，不创建持续监督。

服务器CPU 6项通过；完整G丢弃预检9层零投影最大差0，新投影梯度RMS5.581379e-5非零，最大2528parents累计8micro成功，峰值allocated52.517GiB；旧907×3 Adam状态2721张量逐项一致。E/F正式第1步全部事件、noise/time、8份loss与历史A完全一致，不据此声称整条训练轨迹逐位相同。

实测证据：`audit/server_startup/startup_verified.json`、`first_update_historical_A_comparison.json`、`preflight_gpu.json` 和 `backend_reuse_gate.json`。最终成绩待后台完成训练及完整生成后产生。

## 2026-10-02 新实例恢复

08:00:26 CST 核验实例 `a0r39pbocgt20-0`。旧实例任务已不存在；E/F 已各完成2000更新，F最终评估200/200条已完成（Euler、DPM完整树各0/100）。G保存至1000更新、累计step23000；本轮不追加预算，只继续1001–2000。

G恢复checkpoint SHA256 `cf24f57bb8a57751de1a21d4f0c5e588e8375c6ccf98bb52c6162613d10c4ebb`。CPU检查确认模型和909项Adam全部有限、旧907项step23000、新投影2项step1000；与A配对的前8000事件一致。F checkpoint及200条预测、2000个数组文件哈希核验通过。恢复编排3项测试本地和服务器均通过；冻结训练/模型/评估代码未修改。

当前分配改为：GPU0恢复G并在完成后评估G；GPU1只补E最终评估；F复用。控制器PID567，G PID570，E PID571。G启动文件确认完整Adam及RNG已恢复，正在补中断时未完成的update1000固定探针，然后进入update1001；此时不能称剩余训练已完成。E已完成3/200条生成。

实际恢复命令、CPU检查和启动快照见 `audit/recovery_20261002/launch.json`、`cpu_gate.json`、`startup_snapshot.json`。编排入口 `recover_efg_20261002.py` SHA256 `07aaf008e2236fce539a3714cc3b90d2e8a3934232a4c9b1ae231b69b991f056`。旧状态/日志备份在服务器 `experiment/audit/recovery_20261002`。本轮启动核验结束，不设置监督；后台完成E/G评估后复用A/F并生成原定证据ZIP，异常保留证据且不自动重试。

元数据边界：冻结trainer的resume记录中`checkpoint_sha256`字段计算的是身份JSON的哈希；实际G权重哈希以上述CPU检查与checkpoint身份为准。`source_checkpoint_sha256`仍记录共同D9迁移起点，实际恢复模型来自G step23000。未声称跨实例训练轨迹逐位相同。
