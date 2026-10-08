# B2：从冻结R1 step1000继续随机时间

## 进入依据

最新用户方案明确允许几何/回归通过后进入B2，不追溯性修改旧严格B1记录。冻结64种子9000000..9000063结果：64/64正确，平均MSE0.0015148376，最差0.0066605806；局部响应3/64超标、最大误差0.1958486。因此geometry_regression_passed=true、local_response_diagnostic_passed=false、strict_passed_on_this_followup=false。这64例已消费，不能再作为未见终验集。

## 恢复与训练

- 唯一来源R1第1000步完整checkpoint，SHA256 a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279。持久备份/guohaoran/tmp/vertex_r1_preserved_20260915/checkpoint-step1000.pt，回读校验通过。训练从同SHA的原/tmp副本读取，避免共享盘mmap读取延迟。原权重不覆盖。
- 完整2,332,430,344参数模型，单一UID nexus_2k_000105、depth9 GT parents、固定点云/法向、0/1标签，原flow和阈值。
- strict load模型，原AdamW所有状态张量与param groups逐项相等；恢复torch CPU/CUDA RNG，并断言初始开发验证不消耗训练RNG。不新建有效训练状态、不从头训练、不增加warmup。
- LR1e-5、WD0、clip1、BF16 autocast/FP32参数及Adam。每次8个microbatch，各独立t~U[0,1)和高斯噪声，loss/8反向后一次更新。每条日志记录8个时间和noise SHA、真实LR/UID/投影梯度与更新。
- 固定1000次新增更新（累计1001..2000），每100次开发评估并保存checkpoint。无自动扩预算或超参修改。

## 开发验证与最终单层采样

- t分别为0,.1,.3,.5,.7,.9,.95；每个时间16份固定开发噪声11000000..11000015。逐probe保存MSE、TP/FP/FN、坐标及原始NPZ，不用跨时间平均值掩盖低t。
- 8份固定开发纯噪声12000000..12000007，GT parents，20步Euler为正确性测试，40步仅积分敏感性对照。
- 固定目标的FP32响应每个时间4份probe，参考gain=-1/(1-t)；delta_x标准差=.1*(1-t)，t=.5时与B1的.05相同。保存原始扰动数组；响应继续作为诊断，不挡住几何测试。
- 本轮工程开发门槛：全部7×16例MSE<=.01且坐标精确，且8个20步采样坐标全对。40步结果不能替代20步。
- 1000更新结束后一次性8份全新纯噪声13000000..13000007做20/40步采样，不用其结果继续调参。B2 passed要求开发门槛和这8个20步采样均通过。这些是当前小问题的工程验收，不是论文规定。
- 不自动启动C/D，不增加mesh。预算结束保存结果；失败后先报告，不另加训练。

## 环境、证据和持久性

OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1，脚本显式torch.set_num_threads(8)；PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python。原CPU线程环境曾signal11，限制后B1测试/实际训练通过；本轮复测记录保留。

服务器/tmp/vertex_b2_20260915，主机1fl3imip3trjj-0；launch.json保存PID、start_ticks、冻结代码身份。旧R1 checkpoint已有持久备份，B2新checkpoint目前仍写/tmp，不能把计划备份说成已备份。每次监督下载小日志、JSON/NPZ到本地，不下载权重。
