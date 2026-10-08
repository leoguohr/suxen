# R1_D15_COMPAT 交接包

这是 Nexus 第一阶段顶点生成的两物体、固定点云条件、15 层八叉树过拟合实验的代码和轻量证据包。最终累计训练 6000 步，冻结权重在预留的 16 对种子（92015000–92015015）上生成 32 条完整树；原生整数坐标集合 32/32 正确。详见 `R1_D15_COMPAT.md` 最后一节和 `outputs/final_6000_20260926/verification.json`。

## 从哪里看

- `TRAINING_EXECUTION_CARD.md`、`R1_D15_COMPAT.md`：协议、过程、结果和限制。
- `PHASE_0_5_READONLY_CN.md`：D2 基线、checkpoint 字段、D15 接口及本轮 CPU 表示往返的只读核对。
- `NEW_SERVER_FINAL_RECHECK_CN.md`：新服务器上对 6000 步权重、日志和全部最终预测的再次只读核验。
- `d15_code/`、`scripts/`、`frozen_code/`：本轮模型、训练/评估入口及冻结参考代码；`teacher_candidate/` 是候选参考实现，不是本轮训练模型。
- `inputs/d15/`：两物体 D15 标签与表示检查；`inputs/<UID>/condition_point.npz`：训练与评估使用的实际固定点云/法向，文件 SHA 与原服务器清单相同。
- `inputs/d15_manifest.server.json`：当时服务器上的绝对输入路径；`inputs/d15_manifest.portable.json`：包内相同记录的相对路径版本。解压后若需使用后者，从包根目录运行入口，或按环境改成绝对路径。
- `outputs/final_6000_20260926/`：最终连续训练日志、配置、checkpoint 身份、32 条真实生成的预测与逐层数组、服务器评价和本地独立复算。
- `FILE_SHA256.json`：包内逐文件哈希。

## 权重与验收边界

按此前约定，本包不含 27.99 GB 的模型权重。最终 checkpoint 保留在服务器：

`/guohaoran/tmp/r1_d15_compat_20260925/resume5200_20260926/checkpoint-006000.pt`

SHA256：`6e332b3f8510ffdc8b053075bdf9209404ffa4af1470d33bf488b4482d9598cb`；大小 27,990,606,810 字节。`outputs/final_6000_20260926/checkpoint_identity.json` 记录保存身份及 Adam 状态数量。没有权重时，包内预测、日志和代码可供独立复算，但不能重新运行完整模型生成。

32/32 只证明这两个固定点云条件的原生完整树过拟合；不是 CAD50 或新形状泛化成绩。源 GT 各含重复浮点顶点（24→8、140→52），因此原始点数门禁不通过；显式去重后的连续 XYZ 检查是另一个诊断口径，不能代替原始 GT 标准。
