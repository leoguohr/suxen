# 固定100条VAE：Face负例来源配对对照
<span style="color:#64748b" data-repro-annotation="true">[执行注记：状态以实时日志与完整评价为准。]</span>

本轮用户已授权执行。A_uniform与B_hard各从相同VAE34720完整checkpoint独立恢复，新增500次五mesh累积更新，末尾各35220；两支不相加。只改变Face负例来源。

父点SHA：7fb1e2a5d128762e147a4b87524d4cc212909625b706841c80e3519576e05562。父μ实际Edge FP/FN=59/28，Face=1509/58，联合28/100。

A沿用原epoch/UID均匀合法三元组采样。B困难集合来自父μ预测边图中的实际非GT Face误报，logit降序、三元组字典序打破并列，每mesh上限GT面数。1509个全部纳入，分布65条；不使用五组监控噪声，不刷新。B先生成与A相同U，再由独立epoch/UID RNG从U\H选满Nneg-|H|，与H合并排序。H为空的35条B与A完全一致。每条负例总数不变。

完整保留模型、buffer、两组AdamW、训练noise RNG、全局RNG、participation和调度游标。原组Adamstep34720，logvarstep500。两组lr1e-4、betas(.9,.999)、eps1e-8、wd.01、clip1，无warmup。sampling每forward新epsilon、重计算复用；beta1e-6、Hard4内部/4、每mesh等权、5mesh累积、math00、全网络保持。

新增0/100/250/500保存完整状态并真实网络μ验收全部100条；起末同五组噪声。每20步另存滚动恢复点。评价从预测Edge图完整枚举Face；训练负例loss不能替代实际重建。每分支500更新结束停止，NaN/Inf或协议异常保存失败现场且不自动改配置。

当前服务器一张A100授权卡，A后B顺序执行，各自独立进程与输出目录。run_pair.py带排他锁、STOP和失败停止；未完成A不会自动当成成功切到B。结束后compare_pair.py核对2500份训练ε、顺序、总负例数，生成actual指标、父FP修复/消失/新增表以及evaluation.zip。无需助手长期监督。

## 路径与命令
<span style="color:#64748b" data-repro-annotation="true">[执行注记：状态以实时日志与完整评价为准。]</span>

服务器根：/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_negative_pair_20260930

启动/继续整个有限流程：

```bash
/opt/conda/bin/python -B -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_negative_pair_20260930/run_pair.py
```

该入口自动对尚未完成分支选start/resume，不重置已保存状态。已有活跃进程时排他锁阻止重复启动。整段停止请求在根目录创建STOP；分支入口在更新边界保存（未完成微批组丢弃并恢复该组前RNG），不增加预算。不要同时手动启动第二份进程。

最新状态：pair_status.json；逐分支status.json、updates.jsonl、recovery-latest.json；日志A_uniform.stdout.log/B_hard.stdout.log。完整checkpoint约2.96GB，继续保存在SSD持久挂载，源父文件只读。

## 交付边界
<span style="color:#64748b" data-repro-annotation="true">[执行注记：状态以实时日志与完整评价为准。]</span>

当前CPU准备已通过：实际父SHA、完整Adam34720/500、1509困难候选、2500次A/B采样计划和等数量检查；未将这些CPU测试称作训练完成。启动状态和实际第一步证据另记repro_outputs/status.json。
