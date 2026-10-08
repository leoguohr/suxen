# Fixed100 小噪声 VAE 结果

状态：**complete**。生成时间：2026-09-30T09:55:28.302503+00:00。

仅描述原 AE 28/100 工作点上的小噪声、固定 beta=1e-6 的 KL 适应；不能据此宣称成功 AE、全 100 严格成功或保证全 Gaussian 扰动下重建。

完整性以 JSON 与已提交的 updates.jsonl 为准；pending 不作完整结果解释。

## 完整性检查

| 检查 | 状态 | 详情 |
|---|---|---|
| source_baseline | passed | checked |
| checkpoint_0 | passed | checked |
| checkpoint_100 | passed | checked |
| checkpoint_250 | passed | checked |
| checkpoint_500 | passed | checked |
| eval_0_mu | passed | checked |
| eval_0_noise-861001 | passed | checked |
| eval_0_noise-861002 | passed | checked |
| eval_0_noise-861003 | passed | checked |
| eval_0_noise-861004 | passed | checked |
| eval_0_noise-861005 | passed | checked |
| eval_100_mu | passed | checked |
| eval_250_mu | passed | checked |
| eval_500_mu | passed | checked |
| eval_500_noise-861001 | passed | checked |
| eval_500_noise-861002 | passed | checked |
| eval_500_noise-861003 | passed | checked |
| eval_500_noise-861004 | passed | checked |
| eval_500_noise-861005 | passed | checked |
| 500_paired_epsilon_hashes | passed | {"expected": 500, "matched": 500} |
| training_record_consistency | passed | checked |
| 500_contiguous_updates_and_25_participations | passed | {"records": 500, "expected": 500, "final_optimizer_steps": [[34720], [500]]} |
| step0_gradient_audit | passed | checked |

## 原工作点与各评价路径

microF1 由逐 UID TP/FP/FN 汇总重算。保留/丢失均相对原 28 个 strict joint 成功 UID；新增来自原 72 个失败 UID。完整 UID 名单见 comparison.json。

| 新增步/路径 | 状态 | Edge FP | Edge FN | Face FP | Face FN | Edge microF1 | Face microF1 | joint | 保留 | 丢失 | 新增 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| source AE / 34220 | baseline | 51 | 64 | 1508 | 123 | 0.9998140287 | 0.9960223878 | 28/100 | 28 | 0 | 0 |
| 0/mu | complete | 51 | 64 | 1508 | 123 | 0.9998140287 | 0.9960223878 | 28/100 | 28 | 0 | 0 |
| 0/noise-861001 | complete | 51 | 68 | 1513 | 131 | 0.9998075589 | 0.9959906546 | 26/100 | 24 | 4 | 2 |
| 0/noise-861002 | complete | 55 | 69 | 1515 | 134 | 0.9997994741 | 0.9959784509 | 26/100 | 24 | 4 | 2 |
| 0/noise-861003 | complete | 53 | 68 | 1509 | 130 | 0.9998043252 | 0.9960028192 | 23/100 | 22 | 6 | 1 |
| 0/noise-861004 | complete | 57 | 80 | 1517 | 155 | 0.9997784480 | 0.9959221700 | 26/100 | 25 | 3 | 1 |
| 0/noise-861005 | complete | 56 | 65 | 1517 | 125 | 0.9998043271 | 0.9959956298 | 31/100 | 28 | 0 | 3 |
| 100/mu | complete | 19 | 164 | 1456 | 312 | 0.9997039999 | 0.9956857425 | 27/100 | 25 | 3 | 2 |
| 250/mu | complete | 49 | 127 | 1502 | 256 | 0.9997153531 | 0.9957112118 | 23/100 | 19 | 9 | 4 |
| 500/mu | complete | 59 | 28 | 1509 | 58 | 0.9998593187 | 0.9961790832 | 28/100 | 23 | 5 | 5 |
| 500/noise-861001 | complete | 65 | 33 | 1517 | 67 | 0.9998415316 | 0.9961376216 | 27/100 | 22 | 6 | 5 |
| 500/noise-861002 | complete | 65 | 30 | 1519 | 64 | 0.9998463835 | 0.9961401070 | 26/100 | 21 | 7 | 5 |
| 500/noise-861003 | complete | 62 | 29 | 1510 | 61 | 0.9998528511 | 0.9961693111 | 27/100 | 22 | 6 | 5 |
| 500/noise-861004 | complete | 62 | 34 | 1520 | 69 | 0.9998447647 | 0.9961254392 | 26/100 | 21 | 7 | 5 |
| 500/noise-861005 | complete | 69 | 29 | 1533 | 61 | 0.9998415337 | 0.9961134464 | 26/100 | 21 | 7 | 5 |

## 后验统计

**sigma p01/p50/p99 均为先在每个 mesh 的 latent 元素中计算分位数，再对 mesh 等权平均；不是合并所有 mesh 后的分位数。** sigma RMS 与扰动 RMS 也先在 mesh 内计算再平均。KL/Kmu/Ksigma 在 mesh 内按 vertex × channel 取均值，外层 mesh 等权。

| 新增步/路径 | sigma均值 | sigma RMS | p01 | p50 | p99 | Kmu | Ksigma | raw≤−20 | raw≥10 | 扰动RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0/mu | 0.04978706 | 0.04978706 | 0.04978707 | 0.04978707 | 0.04978707 | 2.138966 | 2.501239 | 0 | 0 | 0 |
| 0/noise-861001 | 0.04978706 | 0.04978706 | 0.04978707 | 0.04978707 | 0.04978707 | 2.138966 | 2.501239 | 0 | 0 | 0.04979266 |
| 0/noise-861002 | 0.04978706 | 0.04978706 | 0.04978707 | 0.04978707 | 0.04978707 | 2.138966 | 2.501239 | 0 | 0 | 0.04977911 |
| 0/noise-861003 | 0.04978706 | 0.04978706 | 0.04978707 | 0.04978707 | 0.04978707 | 2.138966 | 2.501239 | 0 | 0 | 0.0497874 |
| 0/noise-861004 | 0.04978706 | 0.04978706 | 0.04978707 | 0.04978707 | 0.04978707 | 2.138966 | 2.501239 | 0 | 0 | 0.04979171 |
| 0/noise-861005 | 0.04978706 | 0.04978706 | 0.04978707 | 0.04978707 | 0.04978707 | 2.138966 | 2.501239 | 0 | 0 | 0.04976859 |
| 100/mu | 0.04845924 | 0.04869459 | 0.03896836 | 0.04808005 | 0.06068838 | 2.138103 | 2.533026 | 0 | 0 | 0 |
| 250/mu | 0.04661508 | 0.04712789 | 0.03411448 | 0.04605732 | 0.06854808 | 2.140627 | 2.577471 | 0 | 0 | 0 |
| 500/mu | 0.0436549 | 0.04449054 | 0.02781834 | 0.04265938 | 0.07101175 | 2.201259 | 2.650646 | 0 | 0 | 0 |
| 500/noise-861001 | 0.0436549 | 0.04449054 | 0.02781834 | 0.04265938 | 0.07101175 | 2.201259 | 2.650646 | 0 | 0 | 0.04449059 |
| 500/noise-861002 | 0.0436549 | 0.04449054 | 0.02781834 | 0.04265938 | 0.07101175 | 2.201259 | 2.650646 | 0 | 0 | 0.04447669 |
| 500/noise-861003 | 0.0436549 | 0.04449054 | 0.02781834 | 0.04265938 | 0.07101175 | 2.201259 | 2.650646 | 0 | 0 | 0.04448252 |
| 500/noise-861004 | 0.0436549 | 0.04449054 | 0.02781834 | 0.04265938 | 0.07101175 | 2.201259 | 2.650646 | 0 | 0 | 0.04449016 |
| 500/noise-861005 | 0.0436549 | 0.04449054 | 0.02781834 | 0.04265938 | 0.07101175 | 2.201259 | 2.650646 | 0 | 0 | 0.04447126 |

## 训练与梯度

已提交 500/500 次更新；最终已记录 optimizer steps=[[34720], [500]]。每 UID 参与次数及完整统计见 comparison.json。

clip 系数 < 1 的更新：494。下表是更新前 loss/gradient/clip 的逐步统计。

| 字段 | 首条 | 末条 | 均值 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|
| reconstruction_edge_mean | 0.159895 | 0.2123912 | 0.1877984 | 0.005436599 | 0.4758109 |
| reconstruction_face_mean | 0.004661311 | 0.004609415 | 0.06806563 | 0.002828622 | 0.4306581 |
| kl_mean | 4.645515 | 4.823924 | 4.742609 | 4.620873 | 4.857729 |
| k_mu_mean | 2.144275 | 2.17622 | 2.166682 | 2.097187 | 2.235676 |
| k_sigma_mean | 2.501239 | 2.647704 | 2.575927 | 2.501239 | 2.650735 |
| total_mean | 0.164561 | 0.2170054 | 0.2558687 | 0.01374203 | 0.7050751 |
| gradient_norm_before_clip | 2.560937 | 4.101193 | 4.658476 | 0.1749153 | 16.81494 |
| clip_coefficient | 0.3904819 | 0.2438314 | 0.262724 | 0.05947093 | 1 |

各模块位移来自每次更新前后 FP32 参数实差。sum_step_l2 是各步范数之和，不是起末参数净位移。

| 模块 | 非零更新数 | 首步L2 | 末步L2 | 平均L2 | sum_step_l2 |
|---|---:|---:|---:|---:|---:|
| encoder_mu | 500 | 0.1535897 | 0.1428479 | 0.1489422 | 74.47111 |
| decoder | 500 | 0.2785841 | 0.3158288 | 0.2350934 | 117.5467 |
| edge_head | 500 | 0.004330941 | 0.003948921 | 0.003960794 | 1.980397 |
| face_head | 500 | 0.002865552 | 0.005155391 | 0.003980653 | 1.990327 |
| logvar | 500 | 0.05106931 | 0.01107145 | 0.01185831 | 5.929153 |

训练后验为已提交训练 mesh-forward 的等权统计；sigma 分位数同样先取 mesh 内分位数再平均。

```json
{
  "sigma_mean": 0.04669589294791222,
  "sigma_rms": 0.047172440072894097,
  "sigma_p01": 0.03461863718628883,
  "sigma_p50": 0.04613606417775154,
  "sigma_p99": 0.06575511420667171,
  "sigma_min": 0.030971988037973644,
  "sigma_max": 0.08152315709888935,
  "logvar_mean": -6.149626083183288,
  "raw_logvar_mean": -6.149626083183288,
  "clamp_low_fraction": 0.0,
  "clamp_high_fraction": 0.0,
  "clamp_fraction": 0.0,
  "perturbation_rms": 0.04717474700510502,
  "perturbation_max": 0.2797556205749512,
  "kl": 4.742608959388733,
  "k_mu": 2.1666821970939636,
  "k_sigma": 2.5759267612457277
}
```

step0 梯度审计 scope=next_optimizer_batch_5：先对下一真实 batch 的五个 mesh 梯度按 1/5 累积，再求范数/内积；不是 full100 梯度，也不是 mesh 范数平均。

```json
{
  "scope": "next_optimizer_batch_5",
  "epoch": 1711,
  "group": 0,
  "uids": [
    "nexus_2k_000382",
    "nexus_2k_001178",
    "nexus_2k_000039",
    "nexus_2k_001427",
    "nexus_2k_000673"
  ],
  "optimizer_updates": 0,
  "beta": 1e-06,
  "reconstruction_gradient_norm": 2.5609368856086947,
  "kl_gradient_norm": 2.153107232032644,
  "beta_kl_gradient_norm": 2.153107232032644e-06,
  "reconstruction_beta_kl_gradient_inner_product": -7.533739641171485e-08,
  "total_gradient_norm": 2.5609368560971095,
  "reconstruction_group_gradient_norms": [
    2.560629962116315,
    0.03964756213638108
  ],
  "kl_group_gradient_norms": [
    2.1161240216297545,
    0.3973535928021645
  ],
  "logvar_reconstruction_gradient_nonzero": true,
  "logvar_kl_gradient_nonzero": true
}
```

## 固定 epsilon 配对

预期 5×100=500 对；当前：{"matched": 500}。逐对 UID、seed、起末 SHA 和不匹配项见 comparison.json。

## Checkpoint 元数据

以下路径、大小、SHA 仅引用 JSON 元数据，未读取或重新 hash 大权重文件。

| 新增步 | 路径 | bytes | SHA-256 |
|---|---|---:|---|
| 0 | /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/run/checkpoints/vae-0000.pt | 2960557774 | ed9b29decf0c7e788d36e9273f572e034ef15a00371ce317ae6c9efdbeedde04 |
| 100 | /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/run/checkpoints/vae-0100.pt | 2962762766 | 2cba8bac92bf376754548bb9720e729f30164a56109f1b7aaa481d05fbf173f5 |
| 250 | /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/run/checkpoints/vae-0250.pt | 2962712334 | 919e9bf207bd5fec76278419d851c959e9984111dfe70b37efbf672cb73bab6a |
| 500 | /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/run/checkpoints/vae-0500.pt | 2962762830 | 7fb1e2a5d128762e147a4b87524d4cc212909625b706841c80e3519576e05562 |

## 完成后服务器只读复核

已重新读取500条连续更新日志，核对每条mesh新增参与25次，以及500对固定epsilon哈希。服务器无本实验训练进程。末尾约2.96GB完整checkpoint已实际重新计算SHA256，与保存清单一致：`7fb1e2a5d128762e147a4b87524d4cc212909625b706841c80e3519576e05562`。此次未执行GPU前向或新增训练。详情见 `evidence/completion_readonly_verification.json`。
