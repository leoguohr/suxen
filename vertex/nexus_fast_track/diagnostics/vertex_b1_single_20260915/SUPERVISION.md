# 当前任务已推进到B2

唯一当前监督入口：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/vertex_b2_20260915/SUPERVISION.md。不要启动旧任务。以下为历史证据。

# 2026-09-15 已完成并打包

R0在500步未通过，R1在1000步未通过（16/16坐标和每例MSE已过，1/4响应误差0.121402>0.1），两组均未触发64独立holdout。进程已退出，无进一步训练。压缩包路径见local_status.json，用户不要权重。自动化应暂停，不再运行旧实验。

# 当前任务：独立单mesh B1 R0/R1

用户2026-09-15最新要求先把同一8顶点物体多噪声实验跑清楚，覆盖旧A100先行要求；不要重启A100、不要等旧A通过。详细设置见PROTOCOL.md。训练和评估只有nexus_2k_000105，t=.5，depth9，GT parents、点云固定；原完整模型。R0/R1从头初始化、fresh optimizer，不加载A权重。LR1e-5、WD0、warmup100、8噪声累积；不改loss/阈值/结构。

## 服务器

主机必须为1fl3imip3trjj-0，SSH root@172.16.78.10:31548，ControlPath /tmp/nexus-vertex-b1-status2.sock。认证用当前对话用户最新密码；用户粘贴的\@是转义，认证实际为@，不得保存/打印密码。服务器/tmp/vertex_b1_single_20260915，本地本目录。实例更换导致旧A100权重丢失；不再用旧PID或旧任务路径。

当前PID3925、start_ticks319041181，入口SHA256 c6227312de38744fd8dadbbe380fafac5ead164564a12588e05a7f1dbdc0bb01，launch.json保存完整命令。R0先运行，结束后同一进程从头初始化R1。运行前核验host、PID start_ticks、冻结入口SHA。不要重复启动，不修改运行中的代码。

## 验收与监督

每组预算500，仅预定最近3次评估改善且100更新内降低>=10%才自动延长一次到1000。每50update验证16噪声、4FP32响应；候选全部通过才执行一次64独立holdout，每例MSE<=.01、坐标精确、signed gain[-2.2,-1.8]及relative error<=.1。holdout用后不论过否结束该组；R0结束继续独立R1，整个流程不进入B2/C/D。

报告应包含scope实际消费UID、train_config、training_cache_probe、16/64逐probe、empty_baseline和responses_fp32。train.jsonl有每步UID/noiseSHA/LR/投影梯度及更新量。第0步training_cache_probe为空是正常（尚无实际训练输入）；之后缓存与第一份实际噪声SHA需一致。读取result.passed和heldout_evaluated，complete只是进程结束。R1头两步VecSet梯度可能为0，不能直接判断图。

每次监督同步小型console.log、状态、配置、train.jsonl、evaluation与NPZ到本地run/；不要下载权重。检查点目前保留服务器/tmp，不宣称共享盘持久化。正常推进保持安静，只在首次有意义的评估、通过/失败、异常或需用户输入时通知。遇OOM/非有限/segfault保存证据并先诊断，不换小模型或调参。完成两组后整理结果并暂停自动化。

## 新服务器CPU运行环境检查

默认环境两次各33/34通过，同一个旧overfit CPU测试子进程signal11；本轮B1测试两次都通过。限制OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1后全34通过。因此正式启动使用这3个进程级环境变量，训练脚本仍显式torch.set_num_threads(8)，不修改模型或训练协议。该现象提示线程环境相关性，未确认底层根因。保留所有测试日志。
