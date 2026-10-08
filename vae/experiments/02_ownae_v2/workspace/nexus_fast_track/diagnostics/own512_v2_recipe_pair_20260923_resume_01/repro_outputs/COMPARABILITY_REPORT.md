# CAD50 A/B中断恢复

这是原A原版骨干/B V2的同一实验续跑，不是新模型初始化。原始训练源码、网络、损失、数据、采样器和评价器逐文件SHA与父checkpoint记录完全相同；新增入口仅放在recovery/。原实验目录保留。

新设备是用户提供的单张A10080GB，GPU UUID为GPU-0a371e11-9041-9b22-d2d5-8469622c4681。单卡顺序执行：先补完旧欠评checkpoint，再按较低有效步数选择分支，每次续至下一个完整验收点后释放GPU，完整评价，再切换另一支。没有DDP、没有同时占用其他卡。

父状态A10000 SHA256=1efbb1ac20ee0e240d582d8bace84f9f9e4c1753268ce1e6f310e994861af7b2；B9000 SHA256=f4605caa21c402060b39bc6523829b658b3581678be40d5352b59992a4457546。完整model、Adam一阶/二阶矩和step、Python/NumPy/Torch/CUDA RNG、参与计数和采样位置均恢复。学习率继续1e-4，不重新warmup。

旧运行最后记录A10643/B9464，但后643/464步没有可恢复的完整权重，原日志只读保留。再各预留1次可能完成但未写入日志的更新，以保守计入物理预算。共同有效终点因此为19356：A原10643+预留1+新9356=20000；B原9464+预留1+新10356=19821。两支有效模型都最多走19356步，同步记录实际消耗，不能说成各完成20000个有效步。达到Face F1>=0.997、资源上限或错误会提前停止。

原资源账本将掉线前到本次确认旧锁释放之间的离线时间一并保守计费，没有清零24 GPU小时总上限。新训练到累计23 GPU小时停止，保留1小时评价额度。

CPU准备已核验父文件SHA、全部Adam槽/参数映射、结构、数据及代码身份。每段GPU启动再次逐元素核对完整model和全部Adam槽，并恢复RNG。首个重放步骤还与旧日志同一步的UID、负例SHA、Edge/Face四组候选数精确比较，以及loss/梯度norm和BCE分子进行rtol1e-5、atol1e-6比较，位于optimizer.step之前。失败保存现场并停止该分支，不改学习率重试。

repro_outputs/RECOVERY_PLAN.json是预算/来源事实；RECOVERY_LAUNCH.json是后台进程证据；各分支segments/含恢复和首步核验。recovery/controller.log和job_logs/保留实际命令及结果。运行入口为：

```sh
/opt/conda/bin/python -u /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01/recovery/controller.py
```

入口已经安排后台运行，不应重复执行。每段训练由RigorPilot run-train的resume模式监管。结束后自动汇总与分包，大权重留服务器，清单提供大小及SHA；不会自动追加预算或启动原100条训练。
