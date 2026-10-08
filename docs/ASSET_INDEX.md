# 下载与完整性索引

13个ZIP，共 **7.92 GiB**。每份均可独立解压。完整数组保存在私密Release，代码/报告可在仓库直接浏览。

| 下载文件 | 大小 MiB | 内容 |
|---|---:|---|
| [00_collection_manifest_and_commands.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/00_collection_manifest_and_commands.zip) | 1.47 | 采集清单、来源、命令、排除项与SHA |
| [01_vertex_arch_sweep_recovery_and_original.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/01_vertex_arch_sweep_recovery_and_original.zip) | 102.97 | S0/S1终点预测、S2/S3快照、原结构sweep代码 |
| [02_e0_e1_results.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/02_e0_e1_results.zip) | 1241.14 | E0真实重放及E1条件干预 |
| [03_e2_C_results.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/03_e2_C_results.zip) | 1091.91 | E2 C全部评价与日志 |
| [04_e2_N_results.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/04_e2_N_results.zip) | 1119.27 | E2 N全部评价与日志 |
| [05_e2_T_results.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/05_e2_T_results.zip) | 1118.10 | E2 T全部评价与日志 |
| [06_e0_e2_protocol_actual_noise_and_records.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/06_e0_e2_protocol_actual_noise_and_records.zip) | 473.65 | 共同UID事件表、实际噪声/time、父格/target |
| [07_d9_interp_trajectory_results.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/07_d9_interp_trajectory_results.zip) | 907.56 | 插值与真实轨迹配对数组 |
| [08_d15_resume7000_results.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/08_d15_resume7000_results.zip) | 77.33 | D15 50物体累计22000结果 |
| [09_prepared_d9_conditions_labels_and_runtime.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/09_prepared_d9_conditions_labels_and_runtime.zip) | 9.65 | 50物体条件、GT标签、冻结有效代码 |
| [Nexus_Teacher_Reconstruction_20261008.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/Nexus_Teacher_Reconstruction_20261008.zip) | 77.34 | 老师候选复建全部代码和核验材料 |
| [Nexus_Vertex_Historical_Evidence_20261008.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/Nexus_Vertex_Historical_Evidence_20261008.zip) | 1660.45 | 历史Vertex全阶段证据 |
| [Teacher_Original_nexus_overfit_data_results_no_code.zip](https://github.com/leoguohr/nexus/releases/download/evidence-2026-10-08/Teacher_Original_nexus_overfit_data_results_no_code.zip) | 230.73 | 老师原始交付包，原样保留 |

## 校验

- [SHA256SUMS.txt](../manifests/SHA256SUMS.txt)：13个ZIP的SHA256。
- [ASSET_MANIFEST.json](../manifests/ASSET_MANIFEST.json)：字节大小、来源和归档身份。
- 每包内附逐文件清单；Vertex重复数组的原路径由ORIGIN_PATH_MAP映射到唯一文件。
- 上传完成后，GitHub服务器返回的asset SHA256与以上清单逐项比对；上传回执见最终验证记录。

## 权重与原始资料

自己的大checkpoint保留路径、状态及已记录SHA；本次未下载或重算所有大权重SHA。老师原始ZIP自带权重保留。老师原始包无训练源码，候选复建和原件分开。

## 状态边界

本次为已有证据快照，不等待S2–S6完成，也未重启训练。最新实例没有本任务进程，历史running字段保留原样。
