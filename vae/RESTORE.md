# 下载、校验与恢复

从本任务的私密 Release 下载所需资产，需要具有仓库读取权限：

https://github.com/leoguohr/nexus/releases/tag/vae-evidence-2026-10-08

`FROZEN_ASSET_MANIFEST.json` 列出每个资产的字节数和 SHA256；`FROZEN_SHA256SUMS.txt` 用于批量校验。每个归档的 `FILE_MANIFEST.json` 或服务器总清单记录内部文件来源和 SHA256。

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

## 数据、代码与结果

- `vae_frozen_step36220_code_config.zip`：当前有效代码、配置、报告和内部文件清单；也可直接读取本分支的`vae/current/`。
- `fixed100_original_data.zip`：原100条mesh/topology数组、selection与原始manifest。
- `ownv2_fixed100_vae_hard_continue1000_20260930.zip`：最后一段1000次更新的日志、完整实际评价与预测分片。历史中间checkpoint和冗余hidden缓存未收入此包。
- `FROZEN_FILE_MANIFEST.json`：本次保留结果包与数据包的逐文件来源、大小与SHA256。
- `CHECKPOINT_PARTS.json`：最终完整model/AdamW/RNG的来源与分卷恢复信息。

最终模型文件已分为4卷，每卷最多850 MiB。先验证分卷，再验证合并后的完整哈希。不要覆盖其他任务目录，也不要直接复用历史启动脚本的GPU或预算。
