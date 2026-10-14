# 七组 Vertex 实验：终点补评估与后五组接续

## 本轮发现

- 2026-10-07 新实例 `5ag907vlkr8lb-0`，两张 A100 80GB 空闲，旧队列无活跃进程。
- S0、S1 均已完成10000次新增更新，累计step34000。完整模型/Adam/RNG checkpoint已保存并通过原保存入口的全量读回。
- 两支随后在终点评估导入阶段报 `ModuleNotFoundError: No module named 'diffusers'`，队列依故障停止规则没有启动S2–S6。
- 原版本依赖仍位于 `/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_d9_ab_20260930/deps`。本轮将此路径追加在系统包之后；不安装或升级全局依赖。
- CPU完整导入检查通过：PyTorch `2.3.0a0+6ddf5cf85e.nv24.04`、NumPy `1.24.4`保持系统版本；Diffusers `0.40.0`及scheduler源码SHA与原固定版本一致。本轮仍只使用原Euler评估。

## 完整文件核验

| 来源 | SHA256 |
|---|---|
| S0累计34000 | `3919d924ad7c48a3848ef5b4d1247f8ec9296e7b068343bfc7ffbbf1012e4935` |
| S1累计34000 | `2126ce1be93405a45087af60bfed9e17db4e0377597cdd925b6c45e0146ce193` |
| 共同起点A24000 | `2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba` |

三份文件已重新完整读取并计算SHA256。证据见 `audit/preflight.json`、`audit/environment_import.json`。

## 执行边界

- S0/S1严格使用自己的累计34000最终模型，仅补原50对象×2种子完整树评估；不构造优化器、不新增训练更新。
- 保持原条件、D9、20步Euler/层、阈值0.5、BF16网络/FP32积分，以及原样本顺序、实际噪声规则和容量限制。
- 新目录保留训练历史和失败证据，checkpoint身份引用原真实路径；不复制两份28GB大权重，不覆盖旧结果。
- S2–S6各自从共同A24000完整状态独立开始，各10000更新；模型、loss、数据、优化配置和global8/microgroup2不变。
- GPU0：`GPU-6cf268fa-8e3b-ecfc-c88c-22bc22cc98c3`；GPU1：`GPU-6977060b-ce26-90fc-e2ba-8af1370f8c0e`。一张卡完成当前分支后接下一组；发现失败就停止新派发。
- 七组全部完成后沿用原汇总和轻量证据ZIP流程，不包含大权重。
- 当前源运行目录：`/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs`。
- 本轮新输出目录：`/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007b/runs`。
- 只进行启动核验，不创建持续监督。

## 启动状态

- 新代码已上传并冻结，服务器CPU测试10/10通过，实际非空XYZ匹配评分探针通过。
- 初次提交时SSH断开；重连同一实例后核实无启动回执、日志和相关进程，确认未启动。
- 已启动脱离SSH会话的后台队列，PID1126；GPU0执行S0终点评估，GPU1执行S1终点评估。
- 两支均已实际生成并落盘至少一条D9完整轨迹；首份预测和9层数组SHA复核通过。该核验只确认实际启动与文件完整性，尚无最终准确率结论。
- S2–S6已排入同一队列，空闲GPU自动接下一支，每支仍各10000更新。
- 启动证据见audit/startup_verified.json；已停止人工持续监督。
- 本轮未改训练配置，未设置持续监督。
