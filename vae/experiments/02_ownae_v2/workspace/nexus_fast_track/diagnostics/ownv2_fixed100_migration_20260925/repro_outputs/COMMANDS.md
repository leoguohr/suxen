# 准确命令

以下启动命令已实现，但本轮未执行GPU入口，当前配置会拒绝GPU启动。必须先将明确获准的GPU UUID及gpu_authorized写入run_config.json，并再次核对占用。不要擅自启用GPU0。

```bash
cd /guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_migration_20260925
# 已实际执行：仅CPU
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python -u test_cpu.py --data-source /guohaoran/nexus_fast_track/diagnostics/shared_edge_head_fixed100_20260917 --output repro_outputs/CPU_TESTS.json
# 以后获准后：唯一新运行；不能在已有run上fresh
bash launch.sh fresh
# 完整恢复同一run
bash launch.sh resume
# 只续评最后冻结checkpoint
bash launch.sh evaluate
# 仅停止本AE；安全边界保存
 touch run/STOP
```

恢复前需明确撤销本run的STOP请求；程序不会自动忽略该文件。预算账本与源快照均在本run内，不读取旧CAD队列、release或账本。
