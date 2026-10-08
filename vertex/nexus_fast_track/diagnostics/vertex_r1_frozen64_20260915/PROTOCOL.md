# 冻结R1 step1000的新增独立验证

用户2026-09-15新方案授权：保留原严格B1未通过记录，固定R1 step1000权重补做原未使用的64种子，不训练、不选中途checkpoint。若64例几何/回归全部通过，再从原模型/Adam/RNG继续B2；否则先查失败样本，不擅自追加训练预算。

检查点备份到/guohaoran/tmp/vertex_r1_preserved_20260915/checkpoint-step1000.pt；27,990,365,786字节，SHA256 a5f1e7ec30b34f78e9f31f388b3a0f1115d1ad6fbb51a8802a63dfdd26f2e279。备份回读校验通过；完整模型权重只在服务器，不下载。

新脚本evaluate_vertex_b1_frozen.py，沿用原冻结代码的make_model、regression_probe、noise_for、response_metrics。参数load_state_dict(strict=True)，step/variant/UID/条件哈希校验，eval/no_grad/requires_grad(False)，无optimizer，执行前后核对参数version与checkpoint stat。输出目录必须全新，不改旧报告。

几何：原种子9000000..9000063，UID000105、depth9、GT parents、t=.5，原BF16 autocast预测；每例MSE<=.01且完整整数坐标集合相同。

局部响应：64相同新种子在完全FP32下，扰动std=.05，保留signed gain[-2.2,-1.8]及relative error<=.1。分别输出geometry_regression_passed、local_response_diagnostic_passed、strict_passed_on_this_followup，不放宽原阈值。

额外重现旧8000000..8000003四探针，保存x、delta_x、velocity_before/after、noise/target/parents的原始FP32数组，并为8000002记录全部block RMS和逐token能量。64新噪声一经执行即标记已消费，不再用作未见终验集。

小模型CPU执行测试2项通过，验证检查点未改变、旧目录不能覆盖、原始扰动数组能复算signed gain与relative error。完整模型运行与小模型测试分开记录。

若满足几何/回归条件，B2唯一主要变化为每microbatch独立t~U[0,1)，同样随机噪声；保留原Adam状态、lr1e-5、8噪声累积，不新增warmup。预算1000次新增更新，每100开发验证；逐t=0,.1,.3,.5,.7,.9,.95报告，另给GT parents做20步Euler单层纯噪声采样，40步仅对照。其他时间的响应参考-1/(1-t)，不能一律要求-2。B2尚未启动。
