# Step6000：Small / Large Soft4梯度比较

同一个step6000 checkpoint，一次完整两mesh前向，共用同一计算图，分别对每条mesh的Soft4反传。无optimizer、无参数更新、无梯度裁剪，不乘联合loss中的1/2。τ=1、membership detach、FP32组归约；梯度点积与范数用FP64累加。

|部分|cos(g_S,g_L)|‖g_S‖|‖g_L‖|‖g_S‖/‖g_L‖|比例百分数|
|---|---:|---:|---:|---:|---:|
|Encoder|-0.01305230|0.164528457|57.5652798|0.00285811964|0.285812%|
|Decoder body|0.01975740|0.0231440456|12.7915949|0.00180931664|0.180932%|
|edge head|-0.01837630|0.00828270115|1.793157|0.00461906077|0.461906%|
|全模型|-0.01202765|0.16635463|58.9966249|0.00281973131|0.281973%|

Encoder、Decoder body、edge head三者互不重叠；Decoder body不包含edge head。全模型为三者之和，111,491,168个参与边重建的参数。冻结的logvar和face head在本目标下梯度为0，不影响全模型范数。

四处余弦都接近0，而非明显接近−1。全模型small梯度只有large的0.282%，large梯度范数约为small的354.6倍。这个checkpoint上的原始联合梯度主要由large决定；本结果不支持“small以同等或更强梯度，强烈反向拖住large”这一解释。

结论只针对这个后期checkpoint和原始梯度，不能排除早期训练干扰，也不能由此直接确定Adam实际更新方向或最终残留错误的根因。

|本次前向|Soft4|TP / FP / FN|Edge F1|
|---|---:|---|---:|
|Small|0.000108885331|1152 / 0 / 0|100.000000%|
|Large|0.0777621865|7711 / 1 / 8|99.941676%|

原训练step6000记录large为TP7709、FP2、FN10；本次同权重重新前向为TP7711、FP1、FN8。保留原Flash/CUDA后端，存在数值非逐位确定性；两条梯度始终来自本次同一个前向，未混用两个状态。

|重复反向检查|全模型余弦|相对L2差异|
|---|---:|---:|
|small同loss重复|0.999995269|0.309683%|
|large同loss重复|0.999980265|0.629257%|

重复反向的方向非常接近，范数比和“没有强烈反向冲突”的结论不依赖微小数值波动。所有参数、buffer、RNG在诊断前后不变，checkpoint文件哈希未变化。

Checkpoint：`/guohaoran/nexus_fast_track/diagnostics/soft4_two_mesh_resume_6000_20260910/continue/checkpoint-6000.pt`
SHA256：`5233366c71450d5dc08111d69080cc6fcf10d11093519447d9be509d1c107450`

完整原始small/large参数梯度保存在服务器本目录的per_mesh_parameter_gradients.pt；本地complete.json包含全部数值、模块参数量、重复检查与来源记录。
