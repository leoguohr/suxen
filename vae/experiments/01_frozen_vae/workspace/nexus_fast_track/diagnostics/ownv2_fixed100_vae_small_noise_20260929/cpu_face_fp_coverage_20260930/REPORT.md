# Face 实际误报监督覆盖审计（CPU，2026-09-30）

已实际完成。**当前主要证据是直接负例覆盖不足：末尾 μ 路径1509个实际Face FP中，1506个（99.8012%）在本段25次样本参与内从未被选为负例；3个各选中过一次，没有选中过两次或更多的末尾FP。** 这是逐候选实际重放所得，不是概率估计。

本次仅读取已有数组、重放CPU负采样和记录新审计文件；未加载模型权重、未运行网络或GPU前向、未执行optimizer update。没有启动新的A/B训练，也没有修改既有训练代码。运行约95.3秒，退出码0。

## 范围与核验

- 同一固定100 UID，VAE新step0→500（累计34220→34720），训练epoch1711—1735；每条恰好参与25次。
- 起末μ及同5组噪声，共12个条件、1200条逐mesh记录。全部Face分片齐全；校验network-output文件SHA、checkpoint绑定、局部编号、原坐标和GT。
- 使用原`triangle_chunks()`在CPU从已保存预测Edge图重新枚举候选，与保存的Face IDs逐块核对；重新核对标签、TP/FP/FN及缺候选FN。
- 500条更新日志连续；每轮100 UID顺序与原seed/epoch调度匹配。调用原`negative_faces()`重放2500次，**2500/2500哈希与训练日志逐条一致**。
- 当前定义为Hard4，负例为全体合法非GT三元组中每mesh每epoch选`ceil(1.5F)`，并受可用数量上限约束。不是历史Soft4固定pool。
- 使用int64三元组键，避免大mesh编码溢出。合成测试和独立CSV重算均通过；CUDA未初始化。

## μ路径覆盖

| 候选集合 | 数量 | 本段0次 | 1次 | ≥2次 |
|---|---:|---:|---:|---:|
| 起点FP | 1508 | 1505 | 3 | 0 |
| 末尾FP | 1509 | 1506 | 3 | 0 |
| 起末都FP | 1428 | 1426 | 2 | 0 |
| 起点FP、末尾不再FP | 80 | 79 | 1 | 0 |
| 末尾新增FP | 81 | 80 | 1 | 0 |
| 末尾Edge严格正确样本的Face FP | 602 | 599 | 3 | 0 |
| 末尾三条组成边均为GT的Face FP | 1430 | 1427 | 3 | 0 |

末尾66条Edge严格正确，其中38条Face未通过，602个FP中599个（99.5017%）未直接选为负例。这部分不能归因于错误Edge。

全量末尾1430/1509（94.7647%）个FP的三条组成边都是GT边；其余79个含非GT边。GT边组成的三环不一定是GT面，不能将这些三环改成正标签。

80个消失FP中，**74个因末尾Edge图变化未进入候选，6个仍在候选中且Face logit≤0**。不能将80个全部计作Face分类器学会了排除。1428个起末共同FP只表示两端状态，不表示训练期间始终判错。

## 直接选中过的候选

下表为μ起末FP并集里所有命中过的候选，均只命中一次。顶点编号为既定局部编号。

| UID | 三元组 | 选中epoch | 起点logit | 末尾logit | 末尾仍FP |
|---|---|---:|---:|---:|---|
| nexus_2k_001150 | (0,3,14) | 1717 | -0.273333 | 0.998487 | True |
| nexus_2k_001843 | (32,35,74) | 1733 | 4.719064 | 2.159253 | True |
| nexus_2k_000796 | (98,103,106) | 1733 | 1.838900 | -0.646978 | False |
| nexus_2k_000675 | (99,104,110) | 1726 | 3.366260 | 0.184308 | True |

这些记录不能证明命中当时也是FP，也不能把一轮命中理解成充分训练；本审计没有重跑对应中间网络或梯度。

## 五组已保存噪声

| 条件 | 末尾FP | 0次 | 1次 | ≥2次 |
|---|---:|---:|---:|---:|
| mu | 1509 | 1506 | 3 | 0 |
| noise-861001 | 1517 | 1514 | 3 | 0 |
| noise-861002 | 1519 | 1516 | 3 | 0 |
| noise-861003 | 1510 | 1507 | 3 | 0 |
| noise-861004 | 1520 | 1517 | 3 | 0 |
| noise-861005 | 1533 | 1530 | 3 | 0 |

各条件分别统计；重复出现的同一三元组不作为独立试验累加。六种条件的命中epoch一致性已另行核对。

## 当前判断与边界

当前应优先检验负例来源，而不是先把大量FP解释为“反复监督仍学不会”。证据支持后续同父状态、相同负例数量、仅替换部分负例来源的配对对照；它仍不能证明覆盖是唯一原因，也不能保证补入后必然严格成功。

本次审计只覆盖epoch1711—1735，**不推论完整AE/VAE历史从未监督**。没有直接采中不等于没有共享表示带来的间接训练信号；命中次数也不等于梯度大小。仍需保留少数曾命中候选作为后续观察对象。

不新增训练、不调整β/LR/架构、不做GPU梯度诊断。原方案的后续配对实验仍未执行。

## 文件与重跑

- `face_fp_coverage.csv`：9618行，按条件、UID、三元组记录起末logit/候选存在性、FP标记、GT边关系、命中次数/epoch。
- `per_mesh_coverage.csv`：600行，六条件各100条逐mesh汇总。
- `negative_replay_verification.csv`：2500行，原日志及重放SHA。
- `face_fp_coverage_summary.json`：完整汇总、输入清单、限制。
- `source_hashes.json`：实际读取的源码、日志、输入及预测分片哈希。
- `missing_files.json`：空列表，无缺失。
- `execution_verification.json`：执行范围、退出状态、合成测试和独立重算。Torch导入出现一条UCX inotify警告，但审计正常完成，未修改系统配置。
- `audit_cpu.py`：本次实际运行的独立审计脚本；通过原只读代码导入负采样/候选枚举函数。

服务器根目录：`/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929`。原预测NPZ共约5.34GB、Face分片约17.53MB，继续保留服务器，紧凑交付包不重复装入这些文件或checkpoint；逐候选最小证据在CSV中。

起点checkpoint SHA：`ed9b29decf0c7e788d36e9273f572e034ef15a00371ce317ae6c9efdbeedde04`。
末尾checkpoint SHA：`7fb1e2a5d128762e147a4b87524d4cc212909625b706841c80e3519576e05562`。
这两项通过已保存评价绑定核对；本次没有重新读取大checkpoint。

在原服务器复核命令（CPU）：

```bash
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python -B -u /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/cpu_face_fp_coverage_20260930/audit_cpu.py --root /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929 --output /ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_small_noise_20260929/cpu_face_fp_coverage_20260930
```
