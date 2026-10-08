# 科学协议变更记录

用户授权参照老师小AE结构诊断原512网络。本轮仅新增一个末端FFN，未把整个网络改成老师的小AE，也未改原固定100训练。

训练目标、Face pool、全部Edge pair、数据顺序、坐标、阈值、评分、latent512、确定性μ、sampling/KL关闭、logvar冻结、原数值后端全部锁定。原有效 runtime、loader、evaluate 和 Soft4/评分文件与 baseline 字节相同。

新增 `terminal_ffn.py` 实现命名模块及显式hook恢复；`train.py`增加原Adam按名验证、新组、零步与梯度检查、固定100步日志；`helpers.py`仅支持新组初始step0和更强的父预测字节核验。完整差异在交付包 `DIFF_FROM_BASELINE.patch`。

CPU toy测试第一次在依赖导入阶段失败，没有训练更新。用原运行环境本来采用的 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 重跑通过，没有升级PyTorch/CUDA或训练环境。初次失败日志保留。

训练启动代码commit：`866d0a88fb4a032d2971bccfd1b20fbd513c26dc`。后处理代码和最终报告会另行提交；启动代码哈希保留于 `CODE_HASHES.json` 和H `config.json`。

不在本轮自动执行任何额外预算、学习率扫描、全层FFN扩展、负例修改或原100迁移。
