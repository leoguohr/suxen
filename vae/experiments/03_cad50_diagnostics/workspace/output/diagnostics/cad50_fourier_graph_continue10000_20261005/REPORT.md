# CAD50四格续训：已启动

用户指定角色已实际调用：gpt-6-astra / ultra负责上层分析；gpt-6.1-sol / xhigh负责实现。主代理完成代码复核、服务器测试、部署和启动核验。

## 当前事实

四格各自从完整step2000恢复，新增8000次五mesh更新，累计10000停止。两对在3000、4000、5000、7500、9000、10000共同检查点轮转。仅使用GPU0，另一张卡未使用；固定100条和diffusion任务未改动。

启动快照：Fourier-post累计2050，Fourier-pre累计2052，XYZ-post与XYZ-pre等待轮转。GPU0利用率99%，显存13994 MiB。此为启动时快照，非当前实时状态。后续不持续监督训练。

## 已完成核验

- 四个父checkpoint SHA与指定值一致，均为Adam step2000、每UID参与200次，下一游标epoch200/batch0。
- 四父Python/NumPy/Torch/CUDA RNG哈希相同，数据/有效源码哈希匹配。
- 四格真实Encoder到Decoder全50条实际Edge/Face评价逐UID复现旧终点，严格成功18/16/9/15；这些预检为零optimizer更新。
- 新入口服务器CPU测试10/10通过，包括Adam状态、分组/负采样、游标、预算、防重复与调度故障测试。调度故障/轮转是CPU模拟，尚未冒充实际长阶段验收。
- 已运行两支完整model、Adam及RNG恢复逐位一致；保留激活与重计算的下一完整五mesh loss及412份梯度逐位一致。
- 已确认真实Adam更新、参数非零位移，前3步记录已保存。新完整结构评价尚未到3000步。

## 要回答的问题

历史V2在2000步的实际Face F1约19.1%，10000步约86.4%，19356步约99.53%。当前最佳组合2000步约20.84%，处于历史早期水平。延长四格可检验后期差距是否缩小、保持或反转。

当前XYZ-pre仍包含V2完整FFN和GELU等设置，不是历史完整A结构。历史B初始化也不同。当前16条大CAD中约94%—98.5%的Face FN来自缺边导致候选缺失。Fourier与LN影响消息尺度/分布是合理假说，现有证据尚未指定唯一根因。本轮单种子CAD50结论不能直接推广为原固定100条的根因。

## 服务器位置

/ssdwork/guohaoran/nexus_fast_track/diagnostics/cad50_fourier_graph_continue10000_20261005

入口：run_pairs.py；控制器日志：controller.log；每格日志与checkpoint：runs/<arm>/；明确启动命令见LAUNCH.json，显式恢复规则见README.md。

核心模型/目标/评价器逐字节复用，不改LR、不重warmup、不刷新采样规则。完整模型文件留服务器；本包是源码、配置、起点证据和启动记录，不是10000步结果包。前轮四个父文件的路径、大小、SHA见prelaunch/host_and_parent_audit.json。

大模型约2.96GB/份，各检查点完整保存model/Adam/RNG/cursor；best/final用硬链接引用完整不可变文件。到预算或发生错误按协议停止，不自动重置和延长。
