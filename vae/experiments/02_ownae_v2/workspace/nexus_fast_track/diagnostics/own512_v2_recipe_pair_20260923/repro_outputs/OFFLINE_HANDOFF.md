# CAD50 V2 无人值守交接

本轮实验仍在训练，尚未完成。A 为原512骨干＋新配方，B 为 Fourier＋新Graph顺序＋16块逐层FFN的V2＋同一新配方；两支均从随机初始化、fresh AdamW开始，每次累积5条完整mesh，最多各20,000次更新。全量实际Face F1门槛为0.997，严格50/50单独报告。

双卡训练通过独立session运行，父进程已经为PID1，不依赖本地SSH或Codex窗口。训练内置NaN/Inf/评价协议错误停止、预算停止和现场保存。训练到23累计GPU小时停止，为评价保留1小时，全部任务上限24累计GPU小时。不隐式恢复、不扩预算。

第三张卡完成了两支0、500、1000、2000、3000、4000步完整评价；此后31548连接在SSH握手前被关闭，共享评价worker锁为空。原因未确定，不能将其写成评价正常运行。两支训练继续按固定预算执行，待评价checkpoint保存在queue中。

postprocess/finish_unattended.py 为本轮增加的CPU收尾协调器，完全不修改训练器。若原评价worker不在，它只在A或B自然结束、对应已分配GPU确认无进程、显存和利用率均为0后，启动原生评价队列。未调用任何新训练或续训命令，不终止别的进程。延迟评价可能导致0.997的达标发现延后，但各分支仍不超过20,000次更新。

训练和评价均收束后，协调器核对逐步配对日志与逐mesh汇总，生成repro_outputs/SUMMARY.md、RESULTS.json、CHECKPOINT_MANIFEST.json和delivery/目录中的独立ZIP分包。每个ZIP验证CRC，清单包含大小与SHA256；完整model/Adam/RNG权重留服务器，仅记录身份，最后模型独立重算SHA。报告为自动事实汇总，深入科学解释尚需回看结果。若收尾脚本失败，记录UNATTENDED_FAILURE.json；不会以训练重启解决。

服务器根目录：
`/guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923`

查看状态：
```sh
/opt/conda/bin/python /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923/postprocess/snapshot.py
cat /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923/repro_outputs/UNATTENDED_STATUS.json
```

最终收尾标志：repro_outputs/UNATTENDED_COMPLETE.json。所有结果仍保存在服务器；本地电脑关闭时不会自动下载。用户返回后可取回delivery/分包并做最终分析。

原固定100条、历史模型、其他实验及老师生成模块均未纳入本次新训练。
