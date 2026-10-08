# A/B 最终保存中断后的恢复

本轮连接到新实例 `943rmm7q1s5mt-0`，原实例为 `bfbhhdmeiut1i-0`。原训练/编排进程不在，原 `/tmp/nexus_d9_ab_20260930` 不存在。旧 `phase=training` 是遗留状态，不是存活证据。

两组原训练日志都有连续的 2000 次更新，但最终保存/提交被中断，评估未执行：

- A：最终 `.partial` 仅 19327315968 字节，不能使用。最后已验证恢复点是 update1800 / step23800，SHA `ac5b709bc664553825c42eb30aa6894e0ac563323e423bb08cd6689726850bdb`，本轮重新读取一致。
- B：最终 `.pt` 为 27990364660 字节。本轮只用 CPU 完成整文件 SHA、全部 ZIP CRC、模型/Adam 形状与有限性、907 份 Adam step24000、代码/配置/数据清单哈希、16000 microbatch 调度/计数/日志、RNG 结构核验。最终 SHA `183f703c7718156fa9c3204a209b55a641193ad414522bf68d1141d1ff27039d`。这是当前 SSD 的事后核验，旧 NVMe 和旧最终 SHA 记录不可用。

恢复方案：保存原状态/日志及 A 残缺文件；B 保留 update2000，只补有证据支持的完成元数据；A 使用冻结训练入口 `--resume`，从完整的 update1800 模型、Adam、RNG、计数和游标重算 1801–2000。两组有效训练长度仍各 2000 更新；不修改模型、loss、学习率、数据、采样或预算。

两组最终完整权重再次核验后，运行既定 Euler/DPM × A/B 共 400 棵树评估，再生成 `experiment/NEXUS_D9_AB_results.zip`。这次恢复不代表生成验收已通过。

证据：`audit/recovery_cpu_verified.json`、`verify_interrupted_cpu.py`；实际恢复命令和启动状态以 `audit/recovery_launch.json`、`audit/recovery_startup_verified.json` 为准。恢复入口 `recover_interrupted.py` 独立于冻结训练核心。
