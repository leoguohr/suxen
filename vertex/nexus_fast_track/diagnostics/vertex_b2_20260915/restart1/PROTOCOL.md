# B2 restart1：新实例，恢复原R1而非失去的B2进度

用户提供新端口36910要求继续。新主机5utktasmq5anv-0，旧实例的/tmp目录和B2检查点不在新机，最后旧进度只确认到191。持久R1 step1000仍在且SHA验证一致，所以本次从同一R1模型/Adam/RNG重新执行原1000-update B2，不称作从191无缝续训。

训练数学、数据、种子、优化器配置、预算、验证种子与原PROTOCOL一致。不会重做已消费的冻结64噪声评估，只读取原报告作为B2进入依据。

唯一代码变化是检查点持久备份：每100update先保存本地完整checkpoint，再逐块复制到/guohaoran/tmp/vertex_b2_restart1_20260915/checkpoint-last.pt.partial，fsync、回读SHA相等后发布checkpoint-last.pt及backup_verified.json。备份未完成时保留上次完成文件，status=backing_up是正常I/O步骤。不能在第100update备份完成前宣称新B2权重已经持久化。

原R1持久文件/guohaoran/tmp/vertex_r1_preserved_20260915/checkpoint-step1000.pt不覆盖。SHA a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279。

所有新证据下载到本地restart1/run，不覆盖旧run。新服务器实际目录仍/tmp/vertex_b2_20260915；新运行身份见本地restart1/launch.json。
