# Topology AE 当前代码阅读顺序

当前模型只有一条定义：`topology_ae_final_layernorm_no_rms_v1`。保留模块内部
LayerNorm，新增 encoder 和 decoder 末端 LayerNorm；输出 embedding 只做逐
mesh 中心化，不再有 RMS 归一化函数、开关或旧模型兼容分支。

## 主调用链

```text
全 FP32：run_topology_ae_overfit20_packed.sh
仅 Flash I/O BF16：run_topology_ae_overfit20_fp32_flash_bf16.sh
  -> train_topology_ae_overfit_packed.py（共享训练循环）
  -> training_2k.py（每 mesh 的监督与 loss 聚合）
  -> topology.py（网络、interval 与恢复）
     Flash 路径由 flash_varlen_topology.py 替换 attention 执行
```

上述入口均在 `scripts/` 下。AIStation 训练任务使用
`run_aistation_topology_ae_overfit20_dual_fp32.sh` 或
`run_aistation_topology_ae_overfit20_dual_fp32_flash_bf16.sh`；它们增加服务器环境、
双卡、manifest 哈希和输出目录检查，最终进入相同训练循环。

评估单独运行：`scripts/evaluate_nexus2k_topology_ae.py` 通过
`mini_nexus/topology_checkpoint.py` 恢复当前模型，再用
`mini_nexus/topology_evaluation.py` 汇总无向 Edge/Face 指标。

## 网络与评分

```text
encoder blocks -> encoder_output_norm -> μ / logvar heads -> latent
decoder blocks -> decoder_output_norm -> Edge / Face heads
  -> 各 mesh 分别中心化 -> interval -> 独立固定 scale -> BCE
```

- 两个末端 LayerNorm 都作用在预测头之前；不直接归一化最终 spacetime embedding。
- Edge/Face 独立 logit 校准保留：初始化时，在固定 20 个 mesh 和 4 组 VAE 噪声
  上统计 raw interval 的 RMS，计算两个正 scale，训练期间固定。
- 这里的 RMS 只是一次性校准统计量，不是 embedding RMS 归一化。
- Face factor 默认仍为 `0.25`，且校准统计在乘这个因子之前计算，避免校准抵消它。
- Edge 仍监督所有无序顶点对，Face 仍使用全部正面与固定 sidecar 负面。
  四组 balanced BCE、KL 权重 `1e-4` 和逐 mesh 等权聚合不变。

## 训练合同

固定 20 条 manifest 每 step 全部参与；packing 和 DDP 只安排执行与梯度汇总。
Adam、weight decay `0`、gradient clip `1.0` 不变。保留已经执行的 LayerNorm
1,000 步对照的 warmup：默认 200 步，step 1 的 LR 为 `1e-5`，线性增加至
step 200 的 `1e-4`，之后恒定；每步实际 LR 写入日志。不是额外引入新 scheduler。

参数、非 Flash 计算、梯度通信及优化器状态为 FP32；Flash 入口仅外部
FlashAttention-varlen 的 kernel I/O 使用 BF16。全 FP32 入口使用标准 attention。
benchmark/profile 脚本只用于性能测量，使用显式单位 logit scale，不能代替上述
带初始化校准的训练入口。

## 论文边界

24/512 encoder、64 latent、16/1024 decoder、Adam、`lr=1e-4`、全顶点对 edge
监督、采样 face negatives 和 edges-first recovery 来自论文公开设计。
末端 LayerNorm 的位置、warmup、logit 校准、Face factor 数值、32 维 spacetime、
具体 block 划分、KL 权重、负样本配比及混合精度是独立复现选择，不冒充论文细节。
