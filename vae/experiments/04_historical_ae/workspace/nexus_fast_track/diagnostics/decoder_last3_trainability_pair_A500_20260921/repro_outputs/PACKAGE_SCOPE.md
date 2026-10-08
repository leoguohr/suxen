# 本轮实验完整交付范围

四个ZIP合计包含本轮Last2/Last3固定预算对照的全部结果与代码。

- Review：报告、逐步和逐UID指标、完整日志、运行配置、源代码快照、动态依赖代码、源码Git历史、预检和恢复证据、输入及checkpoint清单。
- Predictions：两支0/100/200/300/400/500共1200份完整预测数组，含GT与实际Face候选。
- Weights：本轮所有完整模型、所有中间/最终/latest续训checkpoint和中断快照，以及共同父A500完整权重与Adam/RNG。
- Inputs_Caches：原100条mesh/topology、实际固定Face pool、block13输入缓存、原始预检与跨容器恢复的预测数组。

不包含其他历史实验或CAD50的大权重，也不复制整个conda/CUDA环境；环境版本与原绝对路径已记录。代码与输入保留路径映射，迁移机器时需重新绑定记录中的绝对路径。本包不宣称已在另一台机器完成从零重放。

最终分卷SHA256以FINAL_DELIVERY.json为准；各卷另带PACKAGE_FILE_MANIFEST.json。旧DELIVERY.json属于自动初版打包记录，不能替代最终交付清单。
