# CAD50 Fourier × Graph/LN：进度与双卡恢复记录

Fourier两支各完成累计10000步及真实完整50条Edge/实际Face评价；XYZ两支因实例退出，中断时日志到9989/9977，最后完整可恢复状态均为9000。已获用户明确重放授权，现两张A100分别恢复XYZ-post和XYZ-pre，终点仍为每支10000。本包是进度包，尚不是四格10000步最终结果包。

## 最新完整验收

| 分支 | 完整验收步数 | Edge FP/FN | 实际Face FP/FN | Edge micro-F1 | Face micro-F1 | 联合严格成功 |
|---|---:|---:|---:|---:|---:|---:|
| Fourier，聚合投影后LN | 10000 | 785/185 | 1829/314 | 94.4021% | 83.0820% | 32/50 |
| Fourier，聚合前LN | 10000 | 1284/1223 | 2746/1858 | 85.0676% | 61.7608% | 32/50 |
| XYZ，聚合投影后LN | 9000 | 1200/3556 | 1429/3982 | 66.9079% | 37.0741% | 27/50 |
| XYZ，聚合前LN | 9000 | 1472/4505 | 1989/4312 | 56.3563% | 28.6329% | 30/50 |

以上各点固定的66—274顶点组均0/16联合严格成功。不同预算的结果不得用于四格终点交互比较。各评价使用该checkpoint的真实网络及预测Edge图完整Face候选，不以训练pool指标替代。

## 已具备的同预算结果

9000步时，四格实际Face F1为83.7878%、52.8915%、37.0741%、28.6329%。在post条件下，Fourier−XYZ为46.71个百分点；pre条件下为24.26个百分点；两者差22.4551个百分点。此处是单种子CAD50同预算的描述性干预结果，不能确立历史原100条瓶颈的唯一根因。XYZ-pre仍是V2完整FFN/GELU骨干，不能当作历史原始A结构。

Fourier-post到10000步的Face F1比9000步略降，严格成功33→32；FP增而FN减，不能称为单调改善。Fourier两格10000步Face F1差21.32个百分点，strict均32也不代表其余样本错误相同。

## 双卡执行与速度

GPU0 UUID：GPU-80f199d2-afab-fad3-824d-6d2482a4c882，运行XYZ_LN_post。
GPU1 UUID：GPU-0634fd68-4a4d-facb-1b7f-0f9789679b47，运行XYZ_LN_pre。

两卡均A100-SXM4-80GB。已关闭activation recompute并保留真实训练激活；每支保留microbatch1、五mesh累积、Hard4、原AdamW/LR/RNG/数据及负采样规则。未用缓存替代可训练网络；未改变固定100条和diffusion任务。

- XYZ_LN_post：快照status更新到9241；匹配UID的30步均值从0.7571s降至0.4812s，实测1.573倍；实际allocated峰值4.896GiB。
- XYZ_LN_pre：快照status更新到9231；匹配UID的30步均值从0.7644s降至0.5102s，实测1.498倍；实际allocated峰值5.613GiB。

GPU占用约7GiB/卡，瞬时利用率见LAUNCH_VERIFICATION.json；显存占用率不等于速度。CAD最大274点，当前固定执行方式无需75GiB，未通过无效占位虚增显存。以上只是当前已验证加速配置，不是全局最优速度的证明。

训练继续运行，各文件读取时刻不同，因此status与日志计数可能相差数步；这些运行快照不是新的冻结checkpoint。剩余训练约6—7分钟，另需保存、哈希及末尾实际验收，估计7—10分钟；不是完成承诺。

## 恢复核验、授权及预算

服务器CPU检查已核对四份最新持久checkpoint SHA、412项Adam完整状态及超参、所有step、RNG字段、游标、样本参与数与当前数据manifest。CUDA未用于该CPU审计。恢复入口CPU合成测试3项、重放监护CPU合成测试4项均通过；CPU合成测试不冒充GPU训练。

双卡任务已实际完成模型/Adam/RNG恢复、9000点真实全50条评价、零更新状态检查，并进入正常optimizer更新。首段重放的UID、每mesh FP32 loss bits与Adam step均匹配，未见不一致；见replay状态与restoration证据。全989/977重放区间仍须等监督器结束后才能声明完整验证通过。

原XYZ目录全部只读保留。新恢复目录独立，只复制2001—9000日志前缀，并硬链接不可变完整9000 checkpoint及原best；丢失尾段原日志完整保留。RECOVERY.json是准备时的pending快照，authorization-*.json记录随后的人类批准，supervisor-*.json记录本次启动。

新执行每支1000次，分别重放989/977次；模型有效谱系仍止于10000。本续训段实际计算次数至少为8989/8977，相比原每支8000增加989/977。至少1966次重放开销单独记录；没有把两支预算相加为一份模型。

运行中再出现不一致/异常时，控制器请求两支在完整更新边界保存并停止，不自动重置或再开新预算。两支到10000后按原实际评价器验收，保存完整model/Adam/RNG/cursor并自动停止。

## 路径与启动命令

服务器根目录：
`/ssdwork/guohaoran/nexus_fast_track/diagnostics/cad50_fourier_graph_continue10000_20261005`

本次恢复目录：
`/ssdwork/guohaoran/nexus_fast_track/diagnostics/cad50_fourier_graph_continue10000_20261005/recovery_dualgpu_20261005`

原始根目录9份Python源码保持原hash；新增管理程序仅位于management/。运行命令记录在supervisor和active证据中。本次实际命令如下，已有实例运行，勿重复执行：

```bash
/opt/conda/bin/python -B -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/cad50_fourier_graph_continue10000_20261005/management/run_recovery_dual_gpu.py \
  --recovery-root /ssdwork/guohaoran/nexus_fast_track/diagnostics/cad50_fourier_graph_continue10000_20261005/recovery_dualgpu_20261005 \
  --source /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01 \
  --gpu0-uuid GPU-80f199d2-afab-fad3-824d-6d2482a4c882 \
  --gpu1-uuid GPU-0634fd68-4a4d-facb-1b7f-0f9789679b47
```

## 交付范围

包含：已完成22轮逐mesh评价、有效源码和配置、checkpoint元数据、启动/恢复/CPU审计记录、指标CSV、曲线PNG/PDF及SHA清单。源码以服务器实际文件为准。

未包含：约2.96GB/份的大checkpoint、完整逐步updates.jsonl、原始数据NPZ、完整预测NPZ。大模型路径、字节数及实测SHA见live_audit_20261005.json及各分支checkpoint/best/complete元数据；日志与预测数组仍在服务器相应runs/和recovery目录。此进度包不替代最终训练与完整预测交付。不含SSH密码。
