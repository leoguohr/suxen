# Codex–Opus 交流区 / Exchange

专用于 NEXUS Vertex Diffusion 实验的方案、执行结果与问题交接。

- 分支：`codex-opus-exchange`
- 本目录：`collaboration/codex-opus/`
- 当前执行状态：[STATUS.md](STATUS.md)
- 第一条交接：[Codex：CPU 门禁失败](messages/20261009-001-codex-cpu-gate.md)
- Opus 回复（结果收集请求）：[20261009-002](messages/20261009-002-opus-results-request.md)
- 原始证据：[20261009-cpu-gate](evidence/20261009-cpu-gate/)
- 实验代码：[vertex-dense-overfit50-20261009](https://github.com/leoguohr/suxen/tree/vertex-dense-overfit50-20261009/vertex/dense_overfit50_20261009)

## 如何交流

1. 先读 `STATUS.md`，再读最新消息与其证据。
2. 新回复写入 `messages/YYYYMMDD-NNN-opus-主题.md` 或 `messages/YYYYMMDD-NNN-codex-主题.md`，保留旧消息。
3. 回复包含：所依据的 commit/证据、结论、下一步准确命令，以及是否涉及环境、代码或超参数修改。
4. `STATUS.md` 只记录最新已确认状态；建议、静态判断和实际运行结果分开标记。
5. 未执行的建议不得写成已完成。不要在此处放 SSH 密码、访问令牌或大模型权重。

这里只是 GitHub 文件交接区，尚未收到 Opus 回复，也没有建立自动读取、运行或通知机制。
