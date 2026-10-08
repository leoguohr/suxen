# 本次核验交付

先看`repro_outputs/REPORT.md`。文件包含原样候选网络代码、本次只读运行脚本、全部本次预测、下载包历史预测、原老师参考预测、GT导出、逐mesh复核、命令日志和哈希。

仅复算预测数组需要Python与NumPy，不需要GPU、PyTorch或大权重：

```bash
python audit_saved_predictions.py
python audit_fresh.py
```

这两条命令只写本目录的核验JSON与日志，不训练模型。输入相对包目录定位。

完整网络重放还需要老师原权重。为避免重复，ZIP没有打包权重二进制；路径、大小、SHA256见`repro_outputs/CHECKPOINT_AND_INPUT_MANIFEST.json`。服务器本轮独立目录已具有这些权重；原下载包的README列出了可移植CLI入口，`code/`中的网络源码未修改。

本包证明已执行的范围，不提供完整老师训练器。CPU历史结果和A100新结果分别保留，不能混合挑选seed或误作同一数值执行路径。
