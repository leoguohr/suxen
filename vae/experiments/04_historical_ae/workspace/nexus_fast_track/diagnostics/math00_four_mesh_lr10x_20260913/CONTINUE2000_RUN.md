# MidLR 连续新增2000步运行交接

2026-09-13 16:21 CST 已启动；不是完成报告。

- SSH root@172.16.78.10:31548；ControlPath `/tmp/nexus-midlr-cont2000-20260913.sock`，会话有效时可免重复认证。凭据不落盘。
- 主机 `fs2f7n0minffk-0`，PID 1643，命令 `python -u run_continue2000.py --branch continue2000`。检查 PID 必须同时验证命令/主机/输出进度。
- 远端根目录 `/guohaoran/nexus_fast_track/diagnostics/math00_four_mesh_lr10x_20260913`；本地根目录为本文件目录。输出 `continue2000/`，日志 `continue2000_launcher.log`。
- 起点 `mid3x/checkpoint-update0400.pt` SHA256 `640a981738f96992ba5dd8a6f8a5b0d0306a32560af8edabc195f010d23ba8b1` 已在服务器重新核验。
- 起点累计1400更新；Adam steps为encoder/decoder/edge/face 2200、logvar 2000；训练draw计数6800。新增2000后累计3400；Adam4200/4000；draw14800。
- LR从起点完整加载：encoder_mu 3*1e-8，decoder_body/edge_head/face_head 3e-7，logvar1e-4。其余训练数学完全复用哈希固定的run.py与backend00.py。未重置Adam或logvar，未增加调度、warmup、辅助loss。
- MidLR保存的manifest缺少groups/source_sha256，所以仅这两类元数据从已验证原共同父checkpoint读取；模型、完整Adam和RNG全部来自MidLR，不回放旧epsilon。
- `verify_continue2000.py --start-only` 已通过：模型、完整optimizer、RNG、args逐项相等；mu/固定epsilon/全部50监控结果与MidLR末尾完全相同。step0旧两条50/50，四条0/50。

## 固定预算与验收

只有一支，新增2000实际更新，不自动延长、重启或陪跑。每步同一次四mesh packed forward、各一份新epsilon、四mesh等权；beta1e-4、global clip1、wd0、math00 fully-diff Soft4、原pool和clamp不变。

mu及原固定诊断epsilon：0与每200步；完整模型/五组Adam/训练RNG：0与每200步。监控50组：0/400/800/1200/1600/1800/2000；末尾新50组：每组四seed为2026091300+4*j+i (j=0..49,i=0..3)，已断言与旧监控/旧末尾/固定诊断种子不交。所有评估不反传且断言不推进训练RNG。

结束后运行远端 `python verify_continue2000.py`。核验逐步RNG链、8000个新epsilon唯一且不重合旧1600份、全部checkpoint更新计数与Adam步数、起点完整一致、所有检查点存在、监控epsilon跨checkpoint一致。原始运行器每步保留loss、KL分解、sigma尾部、clamp、梯度/clip和实际参数更新。

同步JSON/JSONL和脚本到本地，避免下载大checkpoint和全量NPZ；用rsync通过ControlPath可以排除*.pt、*.npz。对远端各checkpoint计算SHA256并记录。创建CONTINUE2000_REPORT.md及结构趋势图，区分事实与解释。

报告逐mesh各200步的Edge TP/FP/FN；Face TP/FP/FN；Face FN分解为缺边未入候选与候选存在判负；候选覆盖率=1-missing/(face tp+face fn)；同时给出FP+FN。记录旧样本每个检查点严格保持及监控同集合趋势，不只按F1或末尾loss。

稳定通关仅当新增1600/1800/2000各checkpoint的mu四条同次packed forward全部Edge/Face FP=FN=0且监控四条50/50，并且末尾额外新50组四条50/50。Face从当次预测Edge重新枚举。监控集合参与过配置选择，不称独立测试。末尾首次成功只能称成功工作点；未全对则报告学习进展，不能声称有限预算证明不可学。

禁止自动扩大预算/扫描LR/调整loss/冻结head/VJP/line search/关机。正常训练无变化保持安静；有意义变化、失败、需要用户处理或完成时通知。完成报告交付后暂停本次跟进自动化。
