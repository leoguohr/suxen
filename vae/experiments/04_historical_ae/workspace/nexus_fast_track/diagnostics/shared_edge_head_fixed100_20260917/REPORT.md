# 固定100条 hidden：一个共享 Edge head 的有限拟合

全部固定特征取自原第二轮难负例epoch900/update22500，同一原Edge head初始化；没有使用三条专用head。仅训练同一个FP32 Linear(1024,32,bias=True)，共32,800个参数。原Encoder、Decoder和其他heads不参与训练循环。

先导出并逐条复现基线，再测量成本并固定预算：2000次实际更新；fresh Adam LR=0.0001，betas=(0.9,0.999)，eps=1e-8，wd=0，global clip=1，不修改LR、不自动延长。

每次更新完整遍历原定100条，逐条全量pair loss按1/100反向累积，最后统一clip与一次Adam step。训练目标为mean100(Edge Soft4)，Soft4内部四组均值再除4；不额外乘整目标1/4。没有Face、KL、sampling、MSE或mesh专属head。

## 实测成本

100条共106,325个顶点，84,669,234个无向pair；hidden原始张量415.33 MiB。
A100-80GB实测完整一轮评分0.188秒，评分加反向平均0.539秒。预检峰值allocated 3.22 GiB，reserved 19.62 GiB；预检不更新权重。
实际训练循环用时1264.5秒，每条直接参与2000次；这是2000次共享head更新，不是100份独立head。

| 检查点 | 同一head Edge严格成功 | 全量FP | 全量FN | mean Edge Soft4 |
|---|---:|---:|---:|---:|
| 0 | 50/100 | 209400 | 2 | 0.0630971387 |
| 50 | 53/100 | 195376 | 2 | 0.0613714058 |
| 100 | 53/100 | 187115 | 2 | 0.0608903319 |
| 200 | 54/100 | 179028 | 2 | 0.0604544805 |
| 500 | 55/100 | 171467 | 2 | 0.0601569159 |
| 1000 | 55/100 | 168986 | 2 | 0.0601209026 |
| 1500 | 55/100 | 169294 | 2 | 0.0601196202 |
| 1750 | 55/100 | 169262 | 2 | 0.060119586 |
| 2000 | 55/100 | 168839 | 2 | 0.0601196027 |

末尾保留原成功50条，丢失0条，新增5条。身份清单在summary.json。
首次100/100同时严格成功：None；最长连续0次更新；末尾500点中0次100/100。不同step曾成功的UID不能累加冒充同一head成功。

## 判读

有限预算末尾尚未获得全部100条共同Edge读出；这不是特征或32维评分表达不可能的证明。
本次只评价完整Edge图；没有安装新head到全网络作Face验收，没有宣称Edge＋实际Face或完整AE/VAE通过。原模型checkpoint与历史分支完整保留。

## 核验与材料

- cache/：全部100条FP32 Decoder hidden、GT vertices/edges/faces、原始Edge head输出及基线head梯度；head_original.npz保存原始共享head。
- baseline_verification.json、export_complete.json：缓存读出与真实源网络的raw/center/loss/gradient逐位一致，基线计数复现。
- benchmark.json：完整评分/反向成本、重复性和零更新检查。
- run/evaluations.jsonl：step0到末尾每个同一head状态的100条全量计数、loss、margin。run/updates.jsonl：每次更新的梯度、clip、参数实际位移。
- run/checkpoint-step*.pt：同一head及Adam的检查点；run/verification.json：所有固定检查点重载验收。
- run/final_outputs/：全部100条末尾raw/centered Edge表示及所有pair logits；pair顺序为本地顶点编号i<j的lexicographic上三角顺序，GT由cache中的edges定义。
- per_mesh_trace.csv、per_mesh_final.csv、per_mesh_summary.json、summary.json、curves：逐条轨迹、成功保留与分组汇总。
- 入口、runtime、effective_code、source_archive、review_runtime：实际执行逻辑与有效评分公式。

不含原始大网络checkpoint和Face训练pool二进制；其服务器路径/哈希在EXCLUDED_FILES.json及source_manifest.json。所需固定hidden、GT结构、原head、最终head和所有最终pair logits均已入包。

启动兼容修复：首次Adam初始化因protobuf环境退出，发生在任何更新前；恢复已有PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python设置后重启。保留失败记录，未更改目标、初始化、LR或预算。
