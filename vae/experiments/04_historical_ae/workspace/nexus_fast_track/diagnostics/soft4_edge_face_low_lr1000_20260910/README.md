# B支低LR续训1000步（已完成并核验checkpoint）

从B支step11100+Adam继续到12100；E LR=1e-8、D/Face LR=1e-7。Edge/Face均Soft4，μ路径，其他设置不变。原代码与checkpoint不修改。

已通过31248端口连接服务器6880f6moms5d8-0，核验checkpoint与运行时版本后在GPU0启动。命令：

```sh
/usr/bin/python -u run.py --phase continue --gpu 0
```

每步记录两条mesh的Edge及固定训练候选Face；每50步完整评估实际Edge/Face，枚举边图候选并计入未进入候选的漏面；每200步保存模型与Adam。日志中的I_t仅表示两条Edge完美，完整联合验收看full_reconstruction中的both_edge_and_face_perfect。

训练完成后运行verify_checkpoint.py，只forward重放最终模型，原数值后端的计数差异如实保留。取回JSON/JSONL日志后运行summarize.py生成报告与曲线。preparation.json记录状态和脚本哈希；训练结果以continue/trace.jsonl和complete.json为准。

已完成1000次更新，到step12100。完整结果见REPORT.md与summary.json；最终checkpoint仅forward复查结果见continue/checkpoint_verification.json。
