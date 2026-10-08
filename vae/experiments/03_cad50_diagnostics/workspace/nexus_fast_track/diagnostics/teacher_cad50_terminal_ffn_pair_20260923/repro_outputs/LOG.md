# 本轮日志索引

1. 已读安装skill及引用、原结构诊断和原ZIP重建材料；实际intake输出保留于 intake/。
2. 两服务器GPU/环境与父checkpoint现场核验，见 resources_31548.json/resources_32483.json。
3. Control CPU重新审计两次通过；第二次补充元数据，没有Control新增训练。
4. FFN CPU toy测试首次依赖导入失败，沿用原protobuf环境设置后通过；toy Adam2次，真实参数0次。
5. H在双卡服务器GPU0预检，ready/startup_gate通过；release于2026-09-23T05:53:12Z放行。
6. H完成100更新及0/25/50/75/100五次实际全网络评价；run-train返回0并正常停止。逐步更新总耗时573.21秒，监督器总生命周期包含等待，不是纯训练耗时。
7. 单卡服务器新进程冷加载最终model/Adam/RNG与FFN hook，全50数组逐字节一致，0更新。
8. 独立最终CPU审计第一次遇到非Tensor元数据类型检查错误，保存错误尝试并只修审计；重跑通过全部100步、五checkpoint、250预测及候选完整性。随后关联冷验证据。
9. 小文件下载哈希、独立比较表、报告和代码整理；交付ZIP完整性结果见下载目录 delivery.json。一个本地报告生成heredoc曾因文本编码解析失败，改用文件补丁写入；没有执行训练或修改模型。

原始日志保留：H_terminal_ffn/updates.jsonl、repro_outputs/_runtime/train/*/下的stdout.log、stderr.log、events.jsonl、resources.jsonl、state.json，以及cold_verify_stdout.log和audit_control/。它们未被报告中的叙述替代。
