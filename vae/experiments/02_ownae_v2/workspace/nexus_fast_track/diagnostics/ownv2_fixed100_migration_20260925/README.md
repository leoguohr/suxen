# OwnAE-v2 原100条迁移

本目录是独立的代码适配工作目录。旧CAD、旧100条72/74模型及diffusion任务均未改动。

## 当前执行状态

已部署迁移代码并完成CPU测试；未执行真实V2 CUDA前后向或optimizer更新。服务器指定GPU UUID不存在，当前GPU0/1使用许可尚待确认，因此run_config.json中的gpu_authorized仍为false。不得绕过该开关。

用户最新指示是持续训练，不再要求填写预算；配置已记录continuous_authorized=true、max_new_updates=null、gpu_hours=null，默认每1000次有效更新完整评价100条。这不授权切换到未知GPU。原4h及CAD旧24h/20000步均未沿用。

## 固定实现

native_models.py逐字保留交接版SHA；入口显式选择B_v2_teacher_blocks。完整Encoder/Decoder、latent512、Hard4、μ、冻结logvar、math00及fresh AdamW配方不变。五mesh累积、每epoch20次更新；每mesh原Hard4分组分子/计数跨chunk合并，再/4，负例仍按epoch/UID原规则生成。

原始100条数据从run_config.json的数据源加载并核对selection、manifest和200个mesh/topology哈希。输入数组不重新排序或归一化，不加载旧Face pool负例。

## 入口与恢复

训练前必须确认配置中的GPU使用许可、GPU UUID和持续/有限预算授权。launch.sh仅导出指定UUID并调用独立入口；不会自行找空卡。

```bash
cd /guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_migration_20260925
bash launch.sh fresh
# 只从本run的完整recovery-latest.json恢复；不使用CAD或推理权重
bash launch.sh resume
# 仅恢复同一checkpoint的未完成实际评价，仍须GPU授权
bash launch.sh evaluate
```

计划的唯一运行目录为本目录下run/；目前尚未创建。主线锁为/guohaoran/nexus_fast_track/.locks/original100_ownae_v2.lock，锁文件登记运行目录，不能在另一目录重复建立新主线。

recovery-a.pt/recovery-b.pt交替原子保存，recovery-latest.json指向最近完成提交的完整更新边界。每20次更新及正常停止保存恢复状态。评价所用checkpoints/update-N.pt不可变并绑定SHA256。恢复校验参数名、AdamW、RNG、LR、下一组UID、负例哈希和每mesh参与次数。如果日志存在晚于最后有效checkpoint的更新，明确拒绝隐式重放；不把重算工作称为新增进度。

## 完整实际评价

每条mesh从真实网络生成预测Edge图，以i<j<k流式枚举全部clique；分片包含候选ID、logit和GT标签。网络前向输出仅作为该不可变checkpoint的只读评价缓存，绝不用于训练。

每个分片先原子落盘，再原子提交游标与计数。重启覆盖未提交孤立分片，不重加已提交计数。GT缺候选计FN；只有100条全部完成才生成complete.json及正式micro-F1。Face micro-F1>=0.997是工程门槛，联合严格成功数单独记录。

## 停止与资源

连续模式仍支持本run的STOP文件和非有限值/OOM立即失败；不改精度、样本、模型或自动重置重跑。若以后设置有限预算，训练和评价共用账本，按已测评价时间留余量；预算不足的评价保持incomplete。

```bash
# 请求本AE在安全边界停止；不涉及其他任务
 touch /guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_migration_20260925/run/STOP
```

故障时另存failure-scene-not-resumable.pt用于诊断，不能当作边界checkpoint恢复。首1—3次正式更新与最大mesh所在组会单独记录五mesh、clip、Adam状态建立和step耗时、峰值显存。

## CPU验证和限制

```bash
CUDA_VISIBLE_DEVICES='' /opt/conda/bin/python -u test_cpu.py --data-source /guohaoran/nexus_fast_track/diagnostics/shared_edge_head_fixed100_20260917 --output repro_outputs/CPU_TESTS.json
```

CPU_TESTS.json区分合成图、小型CPU网络和实际100条文件检查。CPU测试中两条UPDATE来自临时小模型，测试目录自动清理，不能算V2正式训练或GPU性能证据。

真实V2整步性能、完整网络训练、全100条重建成绩尚未执行，不能从CPU测试推出这些已通过。reference/保留旧CAD入口和旧GPU预检，仅供阅读，禁止直接运行来替代迁移入口。
