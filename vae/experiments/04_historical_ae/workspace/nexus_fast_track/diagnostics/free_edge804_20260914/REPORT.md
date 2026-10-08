# 804点自由Edge embedding拟合

获得当前32维评分表示对804点GT边图的严格可行解；这不是完整AE/VAE成功。

本次只训练804×32=25,728个表示参数，从only804_mu step1000导出的edge_head_raw初始化。
每次forward仅中心化一次；沿用16+16拆分、原spacetime评分和scale=0.9306077080970389。
全部322,806对参与fully-diff Soft4，外层除4；fresh Adam lr=1e-3、betas=(0.9,0.999)、eps=1e-8、wd=0、clip=1。
Encoder/Decoder/其他head没有载入；无sampling、KL、Face目标。执行2000个更新，无延长。

初始表示、全部logits、Edge loss及raw表示梯度与导出快照bitwise一致；重复forward/backward一致。

| Step | TP | FP | FN | F1 | Edge Soft4 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2074 | 1645 | 332 | 67.722449% | 0.314262867 |
| 50 | 2220 | 1294 | 186 | 75.000000% | 0.281994402 |
| 100 | 2400 | 1205 | 6 | 79.853602% | 0.251339376 |
| 200 | 2406 | 698 | 0 | 87.332123% | 0.170166641 |
| 500 | 2406 | 0 | 0 | 100.000000% | 0.026056128 |
| 1000 | 2406 | 0 | 0 | 100.000000% | 0.004270414 |
| 2000 | 2406 | 0 | 0 | 100.000000% | 0.000801624 |

- 首次严格全对step：320
- 2000个更新后状态中的严格全对次数：1681
- 最长连续全对：1681
- 最小FP+FN：0，首次达到的step：320
- 最后200步：{"perfect": 200, "f1_min": 1.0, "f1_median": 1.0, "f1_max": 1.0, "fp_min": 0, "fp_max": 0, "fn_min": 0, "fn_max": 0}
- 最后500步：{"perfect": 500, "f1_min": 1.0, "f1_median": 1.0, "f1_max": 1.0, "fp_min": 0, "fp_max": 0, "fn_min": 0, "fn_max": 0}
- 2000步均存在非零参数更新：True
- 发生裁剪的更新数：0
- 训练及过程保存耗时：13.3秒
- 原快照文件哈希未改变：True

## 结论边界

保持当前32维、评分、scale、中心化和fully-diff Soft4，在自由表示参数化下已严格拟合完整GT边图。
因此，这条804点边图不能再归因为“当前32维评分根本表示不了”或“Soft4必然无法拟合”。
截图中的局部梯度取舍确实存在，但没有阻止本次收敛。它不是不可行性的证明。
本次绕过了上游网络，并采用表示变量专属Adam和LR；没有单独证明Encoder或Decoder哪一个有错，
也没有验证Face、完整AE/VAE或多mesh共同重建。后续可以使用成功表示作为上游学习的明确参照。

从first-perfect及末尾checkpoint重新载入原始表示，按原评分重新forward；另从GT faces独立重建Edge标签。
核验结果见run/saved_checkpoint_verification.json，确认全部322,806对的FP/FN均为0。

文件：run/updates.jsonl含step0及全部2000步的更新后全量指标；各checkpoint带表示、Adam、全部pair标签和logits。
若出现严格成功，first-perfect.pt/npz保存首次成功状态。best.pt仅按FP+FN最小保存，不能自动视作成功。

最初启动在创建Adam时遇到服务器ONNX/protobuf导入兼容错误，尚无optimizer update。
只为本进程指定PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python后重新启动；没有安装或修改服务器依赖。
失败启动记录单独保存在setup_failed_before_update，不属于这2000步训练。
