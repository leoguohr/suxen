# A24000：GT parents 逐层冻结生成诊断

用户授权：不新增训练，不修改D9、模型规模、velocity目标或条件方式，不加坐标先验；保留F对照。本次仅运行A。启动核验后不设置持续监督。

## 固定协议

- A checkpoint：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/experiment/A/checkpoint-024000.pt`。
- 预期SHA256：`2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba`；累计step24000。
- 原50对象与manifest顺序、8192点固定点云/法向，两个原配对种子97029000、97029001。
- 各样本depth1–9独立使用GT parents，共900次单层诊断。
- 复用冻结sample_level：BF16网络、FP32 ODE状态，20步左端点Euler，t=0噪声到t=1数据，终点阈值>=0.5。
- 共同父格通过整数坐标映射，逐元素复用已有A/Euler实际noise；不能按新shape重新生成同seed后称相同噪声。GT-only parents使用独立确定性噪声，具体key/布局保存在运行配置及每份数组中。
- 原GT parents来自既有D9 leaves，仅提供正确父格；不把目标occupancy送入去噪网络或ODE，不在中途纠正预测。
- 保存parents、初始noise、对齐索引、21个状态、20个速度/时间、最终连续占据与整数子格、哈希；统计逐对象逐层exact/F1/FP/FN与自由展开对照。

## 解释边界

这是GT-parent诊断，不是从根自由生成成绩。GT-only/free-only parents分别统计；缺失自由层不伪造配对。自注意力依赖完整父格集合，因此共同坐标与噪声一致也不等于逐父格独立因果消融。固定旧种子用于配对，不称未见终验。跨实例连续值不预先声称逐位一致，根层同parents同noise用于实际一致性检查。

服务器输出独立于AB/EFG，禁止改写原结果；每份落盘数组哈希后再计算诊断指标，完成后汇总、打包。发生非有限、OOM、哈希或协议失败则记录并停止，不自动重试或改参数。

本卡是执行协议，不代表已经启动或完成；实际状态以本目录audit启动记录与run/status.json为准。

## 本轮启动已核验

2026-10-02 部署至实例 `6hbl02vqntjnq-0` 的空闲A100，GPU UUID `GPU-49342101-4d41-f5bd-7cf1-3de6c20fbe33`。远程根目录 `/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_A_gtparents_20261002`，后台PID1195。

最终诊断脚本SHA256 `a4e7169d735046cfb9aede330c15c82360e467532d1c90e122ab973dc51c3580`；本地和服务器各5项CPU检查通过，Astra只读审计无阻断。真实加载通过A权重、训练源码和数据SHA门禁，未创建optimizer、未进行训练。

首例 `nexus_2k_001150 / seed97029000` 根层与旧A的parents及实际noise一致，最终连续占据最大绝对差0，阈值集合一致。启动检查时前2个对象种子组合、18层已经完成；这不是全部900层最终结果。证据见 `audit/startup_verified.json`。已结束启动核验，不设置持续监督。

后台自动生成 `run/summary.json`、逐样本 `run/seed-*/UID/diagnostic.json`、每层21状态/20速度NPZ，以及 `run.zip` 和 `run/delivery_identity.json`。ZIP包含原自由Euler噪声/数组、GT labels、固定conditions和代码；不含大权重。若异常则记录 `failed_no_automatic_retry` 并停止，不扩大预算或修改模型。

## 完成与复核（2026-10-02）

已从原服务器确认status=complete、100个对象种子组合/900层完成，GPU空闲，无诊断进程。无新增训练。原始ZIP已下载并通过SHA与全部2417个清单成员哈希核验：`/Users/luthier/Downloads/NEXUS_A24000_GTParents_D9_20261002.zip`，SHA256 `a4be8e3ef3b27398be564e9e66752e10b87018b02b070f1ffd36d37033c92264`，364826053字节。

独立CPU复算通过全部900层整数集合/指标、21状态20速度的Euler递推和899个可配对自由层的实际噪声映射；100根层连续输出均与旧A最大差0。唯一缺失自由层保持unknown/unpaired，GT-only的CUDA私有噪声未在CPU重生，只核对已保存噪声、种子/布局与后续轨迹。

depth1–9全对数量（每层100次）：GT parents `[84,54,42,16,10,7,10,7,9]`；自由展开 `[84,47,22,8,3,0,0,0,0]`。第9层GT-parent汇总F1=0.30356052，自由展开=0.07289030；正确parents下仍FP74886、FN73340。说明父格错误有影响，同时层内生成仍未通过，不可把GT-parent成绩当成自由树成绩。

复算报告位于相邻 `nexus_d9_A_gtparents_cpu_audit_20261002/independent_cpu_report.json`；已有轨迹后处理位于本目录 `audit/trajectory_review.json`。原始完整运行日志位于 `audit_server_final/run.log`。本轮不启动新训练、采样器更改或模型修改。
