# 固定100条：Decoder末块＋最终LayerNorm＋共享Edge head

从原第二轮难负例epoch900/update22500及其原始head开始。只训练原decoder_blocks.15、decoder_output_norm和edge_embedding；不是从诊断head的step2000继续。
全部100条输入缓存于最后一个Decoder block的LayerNorm之前。完整网络与缓存路径的hidden、中心化表示、loss、全部可训练参数梯度逐条bitwise复现；预检两轮完整100条forward/backward重复一致。
目标mean100(fully-diff Edge Soft4)，全部84,669,234 pairs，原32维16+16评分和scale；每mesh反向/100，100条累计后统一clip=1并更新一次。fresh Adam，末块+LN LR=1e-5，head LR=1e-4；wd=0、μ路径、KL=0。500步后停止。
实测完整100条评分+反向耗时 [1.4837983376346529, 1.4047507750801742] 秒；峰值allocated 3.330 GiB、reserved 19.615 GiB。训练与末尾核验/导出合计 661.6秒。

|状态|Edge严格成功|FP|FN|mean Edge Soft4|
|---|---:|---:|---:|---:|
|原始基线|50/100|209400|2|0.0630971387|
|Head-only step500|55/100|171467|2|0.0601569159|
|末块+LN+head step500|71/100|156784|1|0.0388917592|

原50条成功保留50，丢失0，新增21。逐UID清单和>1500组结果在summary.json及per_mesh_comparison.csv。

**本次仅Edge诊断。未进行实际Face验收，未将新末块覆盖原源网络。Face head虽冻结，末块适应会改变它的输入；不能据此宣称完整Edge+Face通过。有限预算未全对也不是表示不可能的证明。**

材料索引：config/manifest/freeze_contract锁定来源与训练范围；export_complete与cache/manifest记录真实网络对齐；benchmark记录成本和重复性；run/updates为500条实际更新；run/evaluations为501个同一模型状态；comparison提供已有Control step500逐mesh配对；checkpoint包含tail和Adam；最终logits按i<j的triu_indices顺序，GT来自cache的edges。
启动预检问题（deterministic安装顺序、state_dict非Tensor元数据、缓存模板中的诊断hook序列化，以及保持attention checkpoint的grad/context约定）均在第一个optimizer update前修复，失败日志保留，没有改变模型公式、loss、LR或预算。
Review包包含完整日志、逐mesh结果、实际代码、核验记录、初始/最终末块与head及最终Adam；大型缓存、全量最终logits、中间checkpoint的路径/哈希见EXCLUDED_FILES.json。FullEvaluation包另含上述缓存、logits与中间checkpoint。源大网络和Face pool不入包；源路径与SHA在config.json。
