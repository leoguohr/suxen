# 下载、校验与恢复

从本任务的私密 Release 下载所需资产，需要具有仓库读取权限：

https://github.com/leoguohr/nexus/releases/tag/vae-evidence-2026-10-08

`ASSET_MANIFEST.json` 列出每个资产的字节数和 SHA256；`SHA256SUMS.txt` 用于批量校验。每个归档的 `FILE_MANIFEST.json` 或服务器总清单记录内部文件来源和 SHA256。

## 最终完整 checkpoint

下载 `vae_step36220.pt.part001` 至 `.part004` 及 `CHECKPOINT_PARTS.json`。这些是按字节切分的文件，不是四个独立模型。

```bash
cat vae_step36220.pt.part001 vae_step36220.pt.part002 \
    vae_step36220.pt.part003 vae_step36220.pt.part004 > vae-step36220.pt
shasum -a 256 vae-step36220.pt
```

重建文件应为 **2962774094 bytes**，SHA256：

```text
e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15
```

Linux 可使用 `sha256sum`。校验每一分卷后，再检查重建文件的完整哈希。代码与数据均在新目录中展开；不要覆盖正在运行的任务。

## 数据与代码

- `fixed100_original_data.zip`：原100个UID对应的200个 mesh/topology 文件，原 selection/manifest、数据路径映射及哈希。
- `vae_01_frozen_vae_20261008.zip`：VAE各阶段本地证据、定稿源码包、讲义。
- `vae_02_ownae_v2_20261008.zip`：V2迁移、训练与实际评价/可视化记录。
- `vae_03_cad50_diagnostics_20261008.zip`：本方法在CAD50上的诊断与结构对照。
- `vae_04_historical_ae_20261008.zip`：旧AE/Soft4/共享head与末端训练的来源记录。
- `ownv2_fixed100_vae_*.zip`：2026-10-08直接从服务器归档的三个VAE阶段结果，含实际预测分片与完整日志。
- `cad50_fourier_graph_continue10000_20261005.zip`：四分支10000步及恢复重放的最终结果。

原源码中的绝对路径、保存策略、GPU选择和执行授权都属于历史运行环境。运行前应自行创建独立目录并明确新预算；本归档没有自动启动程序。

若只阅读代码，从 `current/code/native_models.py`、`data_objective.py`、`vae_protocol.py`、`train_continue.py` 与 `evaluate_checkpoint.py` 开始。历史源码与当前有效源码的关系见清单，切勿混用不同实验中的同名模块。
