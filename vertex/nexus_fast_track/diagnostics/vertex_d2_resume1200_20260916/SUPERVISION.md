D2已完成3600步及16对终验，32/32全树正确。勿重启训练或重复终验。结果目录：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/Vertex_D2_final3600_20260916。以下为历史监督说明。

当前实例已更换，请改读 /Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_d2_resume3000_20260916/SUPERVISION.md。以下是旧实例历史说明。

# D2续训监督：从1200恢复到原总预算3600

当前主机feg52efqv6of7-0，SSH root@172.16.78.10:36910，ControlPath /tmp/nexus-d2-current.sock。凭据使用本任务用户最近提供的信息，不写文件或输出。PID2181/start_ticks329453020，入口/tmp/vertex_d2_resume1200_20260916/code/scripts/train_vertex_d2.py，SHA 6e18d5b1ca3fbe4d14d2adb7547962d92a6779daac97ef68ff3da96e904b9266。先核验身份。
本地当前目录diagnostics/vertex_d2_resume1200_20260916，远端root同名/tmp目录，run子目录；持久根/guohaoran/tmp/vertex_d2_resume1200_20260916。持久console=/guohaoran/tmp/vertex_d2_resume1200_console_20260916.log，launch=/guohaoran/tmp/vertex_d2_resume1200_launch_20260916.json。旧持久/guohaoran/tmp/vertex_d2_20260916保持原样，有训练1..1206和检查点1200。

用户原D2总预算3600不变，当前从D2update1200/cumulative5000恢复，SHA e6328c17e507df41f116c69d706d46796fb523e17fd9653de8c58c93631a4ad6。执行1201..3600共2400次，其中1201..1206是丢失内存状态后的重放。合并有效训练轨迹时取旧1..1200+新1201..3600，旧1201..1206保留标为被重放，不能算新追加训练或删除证据。

初始恢复评估update1200是见过种子的一致性检查，不是新开发改善；原84份数组逐份比较（noise/parents/整数数组完全相同，estimate rtol1e-5 atol1e-6）。只有recovery_verification与resume_verification成功才开始训练。检查source_step5000，resumed_d2_update1200，Adam2721张量一致，RNG未被初始评估消耗。训练1201..1206必须matches_original_replayed_input=true，代码逐micro验证UID/depth/time/noiseSHA。若恢复不匹配，保存证据并诊断，不绕过断言。

完整实验设置见PROTOCOL.md：A000105/B000195，8/52顶点，depth1..9均衡，lr1e-5 WD0 clip1 accum8 BF16，VecSet/DiT都训练。每200先完整checkpoint及回读SHA再开发评估；日志每步fsync、评估ledger和NPZ由训练本身持久保存。开发4对27000000..27000003，终验16对28000000..28000015只在全局3600后一次使用。完整条件切换需[[true,false],[false,true]]；depth2共8parents实际同noise的切换诊断独立报告。误差不追加门槛，生成无GT插入/修补/强制非空。

每次监督同步新run小型JSON/JSONL/NPZ和持久console，禁止下载*.pt/*.tmp/*.partial。原1200评估A4/4、B3/4、完整配对3/4、共同parents4/4，B剩余一条最早depth7；恢复复现这个结果不算新进步。正常无变化保持安静，仅有意义变化、异常、完成或用户需处理时通知。用户主动询问直接报当前实际步数。

禁止自改预算/超参或扩到4/10/20。若再换实例，先核验新持久checkpoint及ledger，恢复剩余预算而非重复3600。已有终验seed_started不能称未见。完成后核验training_complete、checkpoint_verified全局3600/cumulative7400、16对final_evaluation_complete；合并旧/新有效日志、离线复算数组并打包不含权重的总包，再暂停监督。工具返回automation不存在时不能声称已暂停。
