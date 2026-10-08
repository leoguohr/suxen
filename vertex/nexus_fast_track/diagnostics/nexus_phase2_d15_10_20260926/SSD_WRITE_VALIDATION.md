# 完整 checkpoint SSD 写入验证

后续状态：本文件记录最初的单份试写。之后已完成自动保存迁移，新进程自动写出的 update 2601 checkpoint 也通过验收。当前入口与证据见 `SSD_MIGRATION.md`，不要把下文试写时的“尚未切换”当成当前状态。

2026-09-27 实际执行结果：通过。训练进程未停止、未重启，未使用额外 GPU，未修改 VAE/AE V2。

- Phase2 update：2400；累计 step：8400。
- 大小：27,990,506,330 字节，约 26.07 GiB / 28.0 GB。
- 源文件：`/tmp/nexus_phase2_d15_10_20260926/nvme_mirror/checkpoint-008400.pt`。
- 已保留的 SSD 文件：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_phase2_d15_10_20260926/checkpoint-008400.pt`。
- 源清单、复制时计算的源 SHA256、SSD 完整回读 SHA256 三者一致：`19b761a9aafe19eb7cf91fa060306b31efa127376f5518b9dbb600d5aa12f9e7`。
- 复制及回读总耗时：63.596 秒。

实际操作先将完整原文件流式复制至目标 `.pt.tmp`，计算源 SHA256 并调用 `fsync`；随后重新打开 SSD 文件，从头到尾计算 SHA256。确认字节数与哈希一致后，将临时文件原子替换为正式文件并同步目录。没有重新序列化或删减 checkpoint，所以保留原文件完整内容；本轮没有额外运行模型加载或恢复训练测试。

服务器同一 SSD 目录保存了 `ssd_write_validation.json`（原始验证结果）、`ssd_backup_identity.json`、manifest/config/startup audit、训练日志快照、恢复记录快照及实际训练和训练后评估入口的源码与哈希。未下载大权重至本机。

训练在最后检查时仍是原进程 PID 2036，update 2437/6000。本轮只完成一份完整大文件的实际验证与备份，**尚未切换后续自动保存路径**。原 `/guohaoran` 路由中的完整恢复点仍为 update 1600；独立 SSD 备份已推进到 update 2400。不要把原入口硬编码的状态字段当作这份 SSD 备份不存在，也不要把一次备份写成后续 checkpoint 已自动持久化。

本次证明此 SSD 挂载当时可以写入并完整回读本模型的一份 checkpoint。原 `storage` 挂载为何清理后仍报 EDQUOT，以及 SSD 的学校平台保留期限，尚未由后台配额记录确认。
