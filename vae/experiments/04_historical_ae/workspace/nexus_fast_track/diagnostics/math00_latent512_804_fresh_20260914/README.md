# 804点完整网络：fresh latent512，μ重建

服务器目录：`/guohaoran/nexus_fast_track/diagnostics/math00_latent512_804_fresh_20260914/`

本轮固定种子20260914，模型通过归档类正常初始化，没有载入旧权重或离线拟合head。latent512，Decoder hidden1024，Edge/Face各32维；math00与确定性Graph；只训练完整 `nexus_2k_001333`。这是新建构流程实验，不能把结果唯一归因于latent扩维。

目标为 `Edge fully-diff Soft4 + Face fully-diff Soft4`。Soft4内部除4保留，无旧诊断的外层1/4，无sampling、KL或蒸馏。logvar专属参数冻结且不进入Adam。其余四组使用fresh Adam、wd0、global clip1。

第1步 E/μ LR=1e-6、D/heads LR=1e-5；第100步达到1e-5/1e-4，此后保持。最多3000次实际更新，完成后停止。

完整checkpoint和实际重建检查：0、100、200、500、1000、1500、2000、2500、2750、3000。Face从当次预测Edge图完整枚举，分块评分，不截断候选。严格成功要求同一次forward Edge TP2406、Face TP1604，四项FP/FN均0。2500/2750/3000均严格成功才记为后期保持。

## 日志

- `status.json`：当前状态与已完成更新数。
- `console.log`：启动、每25步摘要和完整验收。
- `updates.jsonl`：每步更新前的两项loss、Edge及训练Face pool计数、四组梯度/实际位移、裁剪系数、尺度与512通道梯度检查。Face pool计数不是实际Face重建。
- `eval-updateNNNN.json`：更新后的完整Edge与实际Face验收、候选覆盖、最小margin。
- `checkpoint-updateNNNN.pt`：完整模型、Adam、RNG、配置、更新计数。
- `manifest.json`、`preflight.json`、`source_archive/`、`sampling_forward.py`、`loss_changes_exact_runtime.txt`：执行配置、源码及初始核验。
- `completion.json`：训练结束的汇总；异常时为 `failure.json`，并尽可能保存 `failure-state.pt`。

第一次预检查的μ梯度日志钩子挂在返回切片，未能读到梯度；当时**0次optimizer更新**，已归档在 `attempt0_logging_preflight/`。修复为读取实际μ head输出梯度后，用同一种子重新初始化。该归档不是一个训练对照分支。

查看服务器日志：

```bash
cd /guohaoran/nexus_fast_track/diagnostics/math00_latent512_804_fresh_20260914
cat status.json
tail -n 20 console.log
```

训练脚本会拒绝覆盖已有step0 checkpoint；不应重复启动当前任务。旧64维、20-mesh与其他成功记录均未替换。
