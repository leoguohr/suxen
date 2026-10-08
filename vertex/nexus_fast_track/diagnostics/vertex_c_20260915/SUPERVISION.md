# 2026-09-16 恢复复核完成

最终step3800权重冻结复核已完成：开发完整4/4、原定终验完整8/8、所有层局部正确。详见../vertex_c_recovery_20260916/RESULT.md；不声称原终验首次执行。训练预算完成，禁止重启C或启动D。自动监督更新工具返回任务不存在。

# 2026-09-16 状态更新（优先于下方旧实例信息）

新实例1ni1p7s53t4df-0，SSH端口36910，ControlPath /tmp/nexus-c-36910-sep16.sock。旧/tmp消失；持久C检查点已到c_update1800/cumulative3800，SHA f66c3b8e28440b39df1dd557a0631efa6228c27cc89fa0e04f171f757509eff4，已本轮重新读取原权重并复制回读校验。训练预算完成，禁止重启C或启动D。正在新目录/tmp/vertex_c_recovery_20260916恢复冻结评估；本地对应diagnostics/vertex_c_recovery_20260916。原终验是否执行过未知，16000000..16000007此次只能称恢复复核，不能称未见。原训练日志本地仅到713、完整评估到700，后续日志缺失，不可伪造或覆盖。

# 当前授权：C单mesh全部九层，固定新增1800更新

前置冻结B2已64/64通过；结果在../vertex_b2_frozen64_20260915/run/report.json，权重累计step2000/B2update1000，SHA 9dc3cc9ad8ba64a6aaf947df37f1451dc8e4532c2a8eb1f779e989042edc2f3c。14000000..14000063已经消费，不重跑不重新称未见。

服务器root@172.16.78.10:31548，主机12japcu8f32g-0。ControlPath /tmp/nexus-vertex-next.sock。凭据使用当前任务用户提供的信息，不写进文件或输出。若ControlPath失效，先用已授权SSH认证恢复连接，不要猜新密码。
训练PID972，start_ticks323161073；先核验hostname/proc starttime/entrySHA。入口 /tmp/vertex_followup_20260915/code/scripts/train_vertex_c.py，SHA895e32a20a6f871e550a5bb9c86d774cb745a69021444278b99292b59e3be820。
代码与实际数据 /tmp/vertex_followup_20260915/code 和data；run=/tmp/vertex_c_20260915/run，console=/tmp/vertex_c_20260915/console.log。来源checkpoint=/tmp/vertex_b2_step2000.pt。持久C目标 /guohaoran/tmp/vertex_c_20260915/checkpoint-last.pt（首次100更新备份完成前不能声称已保存C）。原B2持久文件保留。

训练严格遵守PROTOCOL.md：固定000105点云/法向，完整2.332B，Adam/RNG恢复，LR1e-5 WD0 clip1 accum8。depth=1+((update-1)*8+micro)%9；每micro新随机time/noise，无warmup、结构/loss/阈值变动。每100评估并完整checkpoint备份+回读SHA。state=backing_up正常，禁止重复复制或启动第二训练。不要因为开发曲线而改预算或超参。完成1800后8份fresh全树与局部层验收；不自动启动D或追加训练。

开发seed15000000..15000003：每seed九层各GT parents纯噪声采样，另一路仅模型生成parents从根完整采样。每层实际noise seed=base+depth*1000。终验seed16000000..16000007仅一次。同样20Euler/threshold0.5，4个开发和8个终验是预先声明工程选择。层级GT整数格unique(vertices//2**(9-depth))；根1parent8occupied，深2..9各8parents8occupied。

对full_generated_parents报告最终坐标集合与first_mismatch_depth；local_gt_parents报告每层正确数量。不得把GTparent局部测试当全树通过。所有连续误差、阈值余量只诊断，不追加硬门槛。超过4096输入parents时容量中止记失败并保存，不截断继续；空树不强制补点。

每次同步JSON/JSONL/NPZ/console/launch到本地当前目录，禁止下载*.pt、*.tmp、*.partial和大模型权重。status未complete时不得打包成最终完成。确认resume_verification中source_step2000、Adam2721tensor相等、RNG未被初始评估消耗。核对真实train.jsonlUID、depth/time/noise序列，每9update各depth8micro。

普通进度无变化保持安静，只在首次有意义评估、异常、恢复失败、连接变化、完成时通知；用户主动询问则立即用最新事实答复。遇NaN/OOM/异常退出先保存证据，不静默重启，不改训练协议。最终完成后核对持久backup metadata为C1800/cumulative3800并SHA一致，复核保存数组、打包所有小型日志数据代码，不含权重；暂停此监督任务。若本地没有automation id，不要声称已暂停。
