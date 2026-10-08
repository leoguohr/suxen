# Topology Flow 本地收录清单

清点时间：2026-10-08T08:11:46.642947+00:00。仅本地只读；没有SSH、GitHub操作、训练或源文件修改。

共 1432 个文件，453.23 MiB；逐文件大小、SHA256、类别和收录建议见 `local_inventory.json`。

## 建议收录范围

| 路径 | 文件数 | 大小 | 用途 |
|---|---:|---:|---|
| `nexus_fast_track/diagnostics/topology_flow_structure_20261002` | 1332 | 449.49 MiB | c0_local |
| `nexus_fast_track/diagnostics/topology_flow_candidates_20261007` | 60 | 0.49 MiB | candidates_local |
| `output/pdf/nexus_topology_structure_20261006` | 21 | 1.71 MiB | research_20261006 |
| `output/pdf/topology_candidates_20261007` | 19 | 1.55 MiB | research_20261007 |

- 两实验目录的有效 `code/`、`configs/`、README、测试/来源哈希、预算/启动/恢复记录；保留历史状态标签。
- `code/_vertex_reference.py` 是Topology依赖，必须保留；这不表示收录独立Vertex训练实验。冻结VAE接口副本同理。
- `actual_run/vae_baseline/`：真实完整 μ、posterior_seed0、posterior_seed1 的逐UID指标和GT/预测OBJ；15/15/14 joint-strict只属于VAE回解。
- `actual_run/inputs_verified/`：固定50条小型原始数组及来源清单。`actual_run/cache/manifest.json`和归一化哈希必须保留。
- `actual_run/cache/*.npz`：完整冻结latent/条件缓存作为可选数据附件，约198MB；不应仅因是NPZ就误删来源manifest。
- 两份研究目录保留PDF/HTML、报告构建源、结构候选数据、固定commit证据和QA。2026-10-07为修订候选目录，2026-10-06保留为历史依据；teacher_evidence是分析笔记，不含老师材料源包。

## 必须避免的结果误读

- `reports/baseline_initial/report_manifest.json` 的训练日志数为0；step500/1000两组 `complete=false, evaluated_meshes=0, identity=null`。其同名PNG/CSV是历史“缺结果”占位，不是Flow实际生成结果。建议不作为主展示上传；需要保留时必须标为历史占位。
- `reports/generation_diagnostic/`同样是0条可用Flow样本的历史诊断，不能替代后续实际诊断。
- `reports/EXECUTION_STATE.json`、`status_20261006/`及部分continuation字段是历史快照；不能覆盖后续1000步完成证据。
- `nexus_fast_track/diagnostics/topology_flow_structure_20261002/actual_run/preflight_20261007/launch_intake.json` 已记录10月7日确认的C0 update1000完整checkpoint SHA，以及500/1000的完整50条生成摘要（joint-strict均0）；本地缺的是逐UID原始评价/生成OBJ目录。此记录中的候选预算、待选架构与GPU是当时状态，不能用作当前启动指令。
- C1/C2本地只有启动/前3步profile，最终进度与预算以主代理本次独立取得的服务器快照为准。
- 本地两实验目录没有完整模型权重；权重留服务器并以路径/bytes/SHA写清单。

## 大文件与重复归档

| 归档 | 大小 | 成员 | 与展开文件逐字节相同 | 差异/缺失 | 建议 |
|---|---:|---:|---:|---:|---|
| `nexus_fast_track/diagnostics/topology_flow_structure_20261002/actual_run/baseline_evidence.tar.gz` | 202.22 MiB | 1182 | 1182 | 0/0 | 排除重复包，保留哈希与展开内容 |
| `output/pdf/nexus_topology_structure_20261006/NEXUS_Topology_Flow_调研与证据包_20261006.zip` | 0.44 MiB | 12 | 12 | 0/0 | 排除重复包，保留哈希与展开内容 |
| `output/pdf/topology_candidates_20261007/Topology_Flow_结构候选与证据_20261007.zip` | 0.54 MiB | 18 | 18 | 0/0 | 排除重复包，保留哈希与展开内容 |

唯一超过10MiB的单文件是212,039,764 bytes的 `baseline_evidence.tar.gz`。其1182成员已逐项对比展开文件；详见JSON。非空内容相同的跨路径组共137组，主要为源码快照/重复GT/报告证据；不要盲删，保留各阶段路径及来源关系。

## 排除项

- `__pycache__/`、`.pyc`、运行锁、PID和临时片段。
- 历史重复tar/zip包；它们不作为Git普通大对象重复加入。
- 独立Vertex实验与handoff、老师原始材料包、外部完整参考仓库。

## 凭据检查

对UTF-8文本做了私钥头、GitHub token、含密码URL、sshpass和凭据赋值模式筛查；发现 0 个待复核条目。仅记录路径/行号/类型，没有复制值。

此扫描不保证识别未知/混淆凭据；PDF/图片/NPZ等二进制未做内容凭据扫描，最终上传前仍需仓库级安全检查。已有远端地址、端口、用户名、GPU UUID和绝对路径只是来源/传输元数据，可由主代理按仓库策略另行脱敏。

## 收录建议汇总

| 建议 | 文件数 | 大小 |
|---|---:|---:|
| exclude | 18 | 0.31 MiB |
| exclude_duplicate_package | 3 | 203.19 MiB |
| exclude_stale_placeholder | 16 | 5.67 MiB |
| include | 1298 | 53.73 MiB |
| include_historical_labeled | 42 | 0.87 MiB |
| include_with_provenance | 5 | 0.34 MiB |
| optional_data_attachment | 50 | 189.12 MiB |
