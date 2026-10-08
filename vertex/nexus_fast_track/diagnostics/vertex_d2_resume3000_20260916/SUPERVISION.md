D2已完成3600步及16对终验，32/32全树正确。勿重启训练或重复终验。结果目录：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/Vertex_D2_final3600_20260916。以下为历史监督说明。

# D2 当前续训监督：3000 → 原预算3600
当前主机43ag0soirt181-0，root@172.16.78.10:36910，ControlPath=/tmp/nexus-d2-now.sock；PID272，start_ticks331741245。入口/tmp/vertex_d2_resume3000_20260916/code/scripts/train_vertex_d2.py，SHA6e18d5b1ca3fbe4d14d2adb7547962d92a6779daac97ef68ff3da96e904b9266。首先核验身份。
当前run=/tmp/vertex_d2_resume3000_20260916/run；持久=/guohaoran/tmp/vertex_d2_resume3000_20260916；console=/guohaoran/tmp/vertex_d2_resume3000_console_20260916.log；launch同前缀_launch_20260916.json。本地diagnostics/vertex_d2_resume3000_20260916。
旧实例日志到3030，完整检查点3000/累计6800，来源/guohaoran/tmp/vertex_d2_resume1200_20260916/checkpoint-last.pt，SHA3bd18daa5ec377eb469f2512f835696f8b37968517e289a82859eda369668980，27990376090字节，已复制/tmp/vertex_d2_step3000.pt并读回核验。只执行3001..3600共600更新；3001..3030是重放，原记录保留。禁止重启整轮或改变预算超参。
入口代码未改，沿用测试通过的完整模型/Adam/RNG恢复。初始3000开发评估应和旧84份NPZ匹配，然后recovery_verification.json与resume_verification.json需成功。1200恢复链后这次source_step应为6800，resumed_d2_update3000，Adam2721张量一致。训练3001..3030需matches_original_replayed_input=true，逐micro UID/depth/time/noiseSHA均比较。恢复一致性不通过不可绕过断言。恢复评估为已见种子，不是新的终验。
最新开发3000：A4/4、B4/4、全树8/8、共同父格4/4。但2200/2600也是4/4，2400/2800回到3/4，不能声称稳定最终通过。最终16对28000000..28000015尚未开始，只在3600权重冻结后运行。
原实验A000105/B000195、全9层、联合训练VecSet/DiT，lr1e-5 WD0 clip1 accum8 BF16，20Euler/0.5，无GT父格注入、强制非空、topk，不追加误差门槛。详见PROTOCOL.md。
每次同步小型日志JSON/JSONL/NPZ及console，排除*.pt/*.tmp/*.partial。安静处理无变化进展，重大进步/异常/完成通知。用户主动询问直接报实测。日志每步、checkpoint每200先持久写回SHA，再评估。
最终有效日志合并：原vertex_d2_20260916取1..1200，resume1200取1201..3000，当前resume3000取3001..3600。旧1201..1206与旧3001..3030分别保留标为重放，不重复计入训练量。3600/累计7400检查点+16对终验完成后，复算所有数组并打包日志、配置、来源、代码、数据、趋势、不含权重，暂停监督。不得自动扩D4/10/20。
