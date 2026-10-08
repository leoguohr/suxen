# Nexus overfit 实验、数据与最终结果（不含代码）

## 内容导航
- output/pdf/nexus_experiment_results.pdf：13 页汇总，三方法 loss 与全部 50 个最终预测对比。
- output/pdf/：另有 CAD 图册、早期三方案对比和给定真实顶点的最终拓扑对比。
- data/cad50/：实际采用的 50 个 CAD，含原始文本 prompt、OBJ/STL/STEP、网格统计；manifest.json 按 accepted 样本整理。
- data/point50/training.pt：实际点训练输入缓存，含文本特征、归一化顶点、重排面索引。
- results/overfit/：cosine、euclidean、spacetime 三方法的早期同预算实验，含 CSV、曲线、配置、权重与预测。
- results/minkowski_continue_01/：Minkowski 早期续训记录。
- results/minkowski_target_099/：最终拓扑模型、训练记录、两组种子的预测与评价。
- results/point_diffusion/latest.pt：最终采用的点网络权重；text_conditions.pt 为文本条件缓存。
- results/point_prior_candidate/candidate.pt：最终点先验训练检查点；independent_verification.json 与 verified_points/ 为独立验证。
- results/cascade_overfit/：最终点到拓扑串联的全部 50 个预测（00-49），含 OBJ、PLY、NPY、faces.json、指标与可视化。
- FILE_MANIFEST.json：每个内容文件的字节数和 SHA-256。

## 最终结果与读法
Minkowski 指 Nexus 默认 Spacetime Interval（式 5）。早期三方案均训练 VAE 3000 步、Flow 4000 步；之后只继续 Minkowski，不能把其最终结果当作三方法同预算比较。
最终串联：点数准确率 100%；坐标 RMSE 6.788794929e-08；平均对称平方 Chamfer 1.237046213e-14；面 micro-F1 0.9910931174（TP 5508、FP 31、FN 68）。
点坐标归一化为包围盒最长边 2。点模型新增了学习的文本坐标先验，原去噪分支权重约 1.22e-5，结果主要来自先验对训练集的记忆，不证明泛化。推理未输入真实顶点或面，未修复预测 mesh；顶点匹配只用于评价。
micro-F1 为汇总指标，不代表每个 CAD 都超过 0.99。CAD 00 的最终面 F1 约 0.681。
各目录保留本阶段历史记录；point_diffusion 下旧的曲线和评价不是最终先验验证。最终以 point_prior_candidate/independent_verification.json 及 cascade_overfit/metrics.json 为准。
顶点/面数在原始 CAD 与训练预处理后可能不同。最终评价使用训练缓存中的目标。

## 打包边界
不含 src、scripts、tests、第三方代码、虚拟环境、下载的 text-to-CAD 预训练模型、生成的 CadQuery 脚本或包含这些脚本的原始模型回复。
仅保留 50 个接受样本；排除失败生成尝试、冒烟实验、旧压缩包、临时文件及冗余中间权重备份。
manifest 删除了可能包含生成代码的初始执行错误文本；保留采样来源和模型修订号。
训练权重和张量缓存是实验数据，不含项目源码。这个包用于交付结果与数据，不是可独立运行的代码复现包。
