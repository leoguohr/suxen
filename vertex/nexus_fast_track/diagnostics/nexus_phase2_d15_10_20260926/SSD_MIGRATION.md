# Phase2 自动保存迁移到 SSD

当前已运行 GPU1 执行提速版，PID 1208/1209；见 `SPEED_OPTIMIZATION_20260927.md`。SSD 持久保存策略未变。

最新续训已迁到新实例物理 GPU1，训练 PID 453，最终评估等待 PID 454。2601 恢复及 2602 新 SSD checkpoint 已实际核验；见 `GPU1_RESUME_20260927.md`。下文 PID 5294/5295 与 GPU0 是上一实例的历史记录。

本轮目的：让后续完整 checkpoint 和日志自动写入 SSD，而不止保留一份手工备份。模型、10 个训练对象、2 个未见对象、Adam、学习率、loss、8 次累积、6000 更新预算和最终评估协议保持不变。

## 活跃路径

服务器 SSD 根目录：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_phase2_d15_10_20260926/`

- 完整 checkpoint 与主日志：`run/`。
- 运行状态、启动核验、训练 stdout 和评估状态：`audit/`。
- 不自动删除的迁移起点与迁移记录：`migration/`。
- 完整源码、manifest、12 个对象的条件与 D15 labels：`runtime/`。
- 训练完成后的预测与指标：`post_training_final_012000/`。
- 旧 `/guohaoran/.../run_10/storage_migration_pointer.json` 指向新位置；旧 status 是冻结的历史快照。

临时盘 `/tmp/nexus_phase2_d15_10_20260926/run_ssd_20260927/` 仅是本地写入暂存和日志副本。每次 checkpoint 必须完成 SSD 复制、fsync 和回读 SHA256 校验才更新身份清单和持久保存步数。

## 已完成的迁移检查

原进程 PID 2036 在完整 update 2600 / cumulative step 8600 保存后响应 SIGTERM 正常退出。日志末条更新与 checkpoint 更新一致，**没有丢弃或重算已完成的更新**。

迁移源及 SSD 回读 SHA256 均为：

`b8df8bc9257233dfe7a49c9391dc75b109705e06d0d71f0a14d5b4a7290af812`

该完整状态为 27,990,506,330 字节，保存于 `run/checkpoint-008600.pt`，并在 `migration/checkpoint-008600.pt` 保留硬链接。后续轮换不会删除迁移目录中的恢复点。先前试写的 `checkpoint-008400.pt` 也保留在 SSD 根目录。

186 个历史源码文件和 manifest/12 个对象的条件、labels 均按原记录核验哈希后迁移。新训练入口静态核对显示：模型与数据准备、optimizer/RNG 恢复以及前向/反向/更新代码区域与原入口一致，两个新入口语法检查通过。

首次恢复启动 PID 5084 在构造 Adam 时遇到 protobuf/ONNX 兼容错误，尚未执行新更新；错误证据保存在 `migration/startup_attempt_1/`。已恢复原启动脚本使用的 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`，CPU Adam 导入与构造检查通过后重新启动。没有安装、升级或降级软件包。

当前训练 PID 为 5294，最终评估等待进程为 5295。GPU 环境沿用 `GPU-6cf268fa-8e3b-ecfc-c88c-22bc22cc98c3`（当前容器逻辑 GPU0）。启动审计已实际通过：strict model load、907 份 Adam 状态均为 step 8600、Python/NumPy/CPU/CUDA RNG 回读与源状态一致、下一组合索引为 100。

## 自动保存与评估规则

迁移后第一个完整 optimizer 更新额外保存一次，用于验证实际自动保存；随后每 200 次更新、训练结束或收到正常停止信号时保存。保留最近两份不同的完整 checkpoint，`checkpoint-last.pt` 是便捷硬链接，**不计入两份数量**；迁移起点另行保留。写盘或数值错误会退出并记录错误，不盲目续训。

最终评估等待新训练 PID 成功完成 6000 更新、step 12000 的 SSD checkpoint 身份及回读校验通过，才执行原 4 种子 ×（10 seen + 2 unseen）完整树生成。每层 20 步 Euler，结果、日志和压缩包也写 SSD。当前尚未执行最终评估。

新训练入口 SHA256：`96eb005b3258d294f608294768e438a8637f1dbf63b12b80488de51809c24954`。

新评估等待入口 SHA256：`399afaa6367085500198b95bc206616bc71a1feed6c30a61c4492777c4fea7bd`。

迁移记录中的完整执行命令、初次失败和修复环境均已保留。**切换已完成，并已通过第一份实际自动保存的完整 checkpoint 验收。**

## 首份自动保存的实际验收

update 2601 / cumulative step 8601 的文件为 `run/checkpoint-008601.pt`，大小 27,990,466,036 字节。独立完整回读 SHA256：`fb8b41508aac80263733c9bd00f9ff3d576716f67d0fa904cb359a51ab9feb78`，与新训练入口保存的身份清单一致。

额外使用 CPU mmap 加载实际文件，确认 907 个 model 条目、907 份 optimizer 状态且全部 step 为 8601；Python/NumPy/CPU/CUDA RNG、配置、源 checkpoint 身份和下一批位置均存在。scheduler 为原协议的 None；下一组合索引为 108。首个新更新实际消费的八份 UID/depth 与原调度完全一致。SSD 中两份数值命名 checkpoint 的 inode 不同，迁移起点仍在独立目录保留。

最后直接检查：训练 update 2629/6000，状态 training，`checkpoint_storage=gpfs_ssdwork`，`last_persistent_checkpoint_update=2601`。训练后评估仍在等待成功完成，不是已完成评估。原始 JSON 证据已保存到本地 `ssd_migration_evidence/`，没有下载大权重。
