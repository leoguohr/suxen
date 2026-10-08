# Cold verification 只读代码审阅

审阅者：GPT-6 Astra / xhigh；2026-09-23。审阅结论：未发现阻断问题，实际执行结果需由 COLD_VERIFY.json 单独证明。

- fresh setup 后通过 load_extended_state 重建 FFN 和 terminal-LN 前 hook，再 strict 加载最终完整 model；参数值不会依赖新构造的随机初始化。
- 旧四组后追加同顺序 FFN 参数组，加载完整 Adam 并恢复 Python/NumPy/torch/CUDA RNG；逐项精确比较，旧组 step2600、新组 step100。
- 全50按原 runtime/objective/evaluate 重新前向。对最终保存结果逐UID比较 loss parts、counts、strict UID 与所有 NPZ 数组 dtype/字节；cold_verify 子目录路径由相对 H ROOT 的 prediction_path 正确解析。
- 冷验前后 model、Adam、RNG 再次比较；没有 optimizer.step。独立 CPU audit 另外证明完整候选、checkpoint SHA、状态计数、推理文件与可续训文件一致。

边界：冷验验证此最终可续训 checkpoint，未重新训练，未承诺其他 loader 会自动恢复新增 hook；后续必须继续使用命名扩展加载入口。

审阅文件 SHA256：`640cb094b27113913e9a30dc7c94b1b36faa9947148c4a53f5ecf91e7ad3509d`。
