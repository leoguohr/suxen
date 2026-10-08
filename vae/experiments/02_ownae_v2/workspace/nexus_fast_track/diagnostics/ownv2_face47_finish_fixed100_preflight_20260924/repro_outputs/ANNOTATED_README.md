<!-- rigorpilot:repro:begin kind="banner" section="__banner__" occurrence="1" status="partial" risk="none" -->

# 📄 README · RigorPilot 复现批注

🟡 `partial` · `bounded Face finishing and zero-update preflight` · `trusted` · [SUMMARY](SUMMARY.md) · [COMMANDS](COMMANDS.md) · [LOG](LOG.md) · [status.json](status.json)

章节覆盖：🟡 1（共 1 节） · 复现得分 0.5

<sub>🟢 成功 · 🔵 未执行 · ⚪ 仅阅读 · 🟡 部分完成 / 资产缺失 · 🔴 阻塞 · 🟣 待决策 —— 原文未改动；相对媒体链接需要原 README 所在目录的上下文。</sub>

<sub>original_sha256: `f9a52cebf06c3524f6b36b0ac955cdef6c275891b2fc7be11e40ef08da61a3e8` · round-trip: verified</sub>

---

<!-- rigorpilot:repro:end -->
# B19356 Face finishing and fixed100 zero-update preflight

Authorized 2026-09-24. Parent SHA256:
`6f965d3837c32101dc7efd52252c12387fb3f10c788e6fd0d33ae359a44903a2`.

Only the original face_embedding weight and bias train. Upstream, final LN,
and Edge head remain frozen. Restore full model/AdamW/RNG, preserving LR1e-4,
weight decay0.01, clip1, Hard4 and five complete meshes per update. Continue
the original stateless random negatives and add all47 non-GT triangles from
the parent's predicted graph, deduplicated per UID. All50 meshes participate.
Maximum500 new updates; actual full-network evaluation at0 and every50.
If Face micro-F1 reaches0.997, stop training and verify all prediction arrays
in a fresh process. Never extend this budget. Retain complete frozen Adam
slots without updating them. Cache only frozen decoder outputs, and verify
cached versus real-path head predictions and gradients before training.

Fixed100 preflight: verify original selection and all mesh/topology hashes.
Use randomly initialized identical V2, all pairs, all GT faces plus the same
random-negative ratio for one largest-mesh full forward and backward.
Optimizer updates=0; no Adam is instantiated. Preserve FP32/MATH, original
coordinates, vertex order, topology and activation recomputation. Do not
enumerate the randomly initialized model's potentially dense triangle graph:
this is training resource preflight, not reconstruction evaluation.
An OOM is recorded explicitly; any retry may only change chunking/recomputation.

Original projects, data and checkpoints are read-only. No fixed100 long
training is launched; that requires a separately specified budget.
<!-- rigorpilot:repro:begin kind="section" section="B19356 Face finishing and fixed100 zero-update preflight" occurrence="1" status="partial" risk="medium" -->

> [!WARNING]
> 🟡 **部分完成（中风险）**
> 命令：`python face_finish.py train; resumed at update50 with --resume checkpoint-new0050-step19406.pt`
> 阻塞项：未记录
> 观测指标：`Face_F1=0.9948103078024337` · `Face_FP=41` · `Face_FN=17` · `joint_strict=38`
> 建议下一步：Budget complete. No more training authorized by this run.
> 命令来自 README 链接的文档 `face_finish.py`。
> <sub>证据: [SUMMARY](SUMMARY.md) · [COMMANDS](COMMANDS.md) · [LOG](LOG.md) · [status.json](status.json) · tier: execution</sub>

<!-- rigorpilot:repro:end -->
