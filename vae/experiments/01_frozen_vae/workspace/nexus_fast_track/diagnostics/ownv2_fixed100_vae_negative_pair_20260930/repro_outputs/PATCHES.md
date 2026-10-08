独立实验目录，非Git checkout，无分支/commit；未修改既有运行源码。新增pair_negatives.py/prepare_pair.py/train_pair.py，以及独立启动/比较入口。继承源码SHA见inherited_code_sha256.json；最高风险是错误恢复Adam/noise RNG或改变负例数量，CPU计划核验和首点真实复现用于约束。
