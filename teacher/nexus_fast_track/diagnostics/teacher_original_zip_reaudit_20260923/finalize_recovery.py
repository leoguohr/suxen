"""Assemble the bounded original-ZIP reverse-recovery handoff."""
import hashlib
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
CODE=OUT/'reconstructed_from_original'

def main():
    manifest=json.loads((OUT/'FILE_MANIFEST.json').read_text())
    evidence=json.loads((OUT/'DERIVED_RECOVERY_EVIDENCE.json').read_text())
    verify=json.loads((CODE/'VERIFICATION_RESULT.json').read_text())
    cascade=json.loads((CODE/'cascade_cpu_34567_12345/CASCADE_RESULT.json').read_text())
    spec=json.loads((CODE/'verification_input.json').read_text())
    mapping={kind:{'original_to_recovered':{k:k for k in shapes},
                   'recovered_to_original':{k:k for k in shapes},
                   'tensor_shapes':shapes,'coverage_fraction':1.0}
             for kind,shapes in spec['state_shapes'].items()}
    (CODE/'STATE_KEY_MAPPING.json').write_text(json.dumps(mapping,indent=2)+'\n')
    status={
        'author_model':'gpt-6-astra','reasoning_effort':'xhigh','date':'2026-09-23',
        'original_archive_sha256':manifest['archive_sha256'],
        'same_original_archive_as_yesterday':True,'new_teacher_files_discovered':False,
        'optimizer_updates':0,'gpu_execution':False,
        'original_pt_files_freshly_inspected':22,
        'implementation':'Newly written recovery; original evidence plus explicitly inherited tested forward hypotheses, not blind recovery.',
        'code_directory':str(CODE),'strict_loads':verify['strict_loads'],
        'ae':{'weights_and_structure':'recovered and strict-loaded',
              'forward':'all50 CPU checked, embeddings exactly equal yesterday under same runtime',
              'metrics':verify['ae_50_summary'],
              'training':'loss components/config/logged schedule partially recovered; final AE optimizer absent; exact trainer absent'},
        'topology_flow':{'weights_and_structure':'recovered and strict-loaded',
              'forward_sampler':'fixed-input paired tests and full50 generated-point cascade executed',
              'optimizer':'112 ordered moment shapes match Flow, Adam step14000; cumulative Flow counter128000',
              'rng_keys':['python','numpy','torch','cuda'],
              'training':'checkpoint state partly resumable in principle; exact original trainer/step ordering unavailable'},
        'point_flow':{'weights_and_structure':'recovered and strict-loaded;18layer width144 prior512 slots274',
              'forward_sampler':'all50 count correct and100step point generation in full cascade',
              'prior_optimizer':'6 prior tensors+1 residual_scale, Adam step50000, prior_steps160000, noRNG',
              'source_step_difference':{'candidate':267500,'inference_frozen_source':258000,'resolved':False},
              'training':'exact denoiser/prior trainer and cached frozen outputs unavailable; raw-text encoder/tokenizer unavailable'},
        'full50_cascade':{k:cascade[k] for k in ['torch','device','point_seed','topology_seed','rng_context','total','point_total','generation_seconds','yesterday_environment','yesterday_total','comparison_limit']},
        'do_not_claim':['original source exactly recovered','original complete training protocol recovered',
                        'from-scratch training success','cross-runtime bitwise reproducibility','unseen text generalization'],
        'next_minimal_check':'To close training recovery, request the original trainer/negative sampler, pre-hard-example checkpoint and original point-source/cache files. Do not infer those from F1 or launch another training run.'}
    (OUT/'NETWORK_RECOVERY_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    score=cascade['total']['face'];point=cascade['point_total'];ae=verify['ae_50_summary']
    report=f'''# 原始老师 ZIP：本次实际逆向实现与核验

作者：gpt-6-astra，reasoning effort=xhigh；日期：2026-09-23。实际 optimizer 更新 **0**，未运行 GPU、未安装依赖、未改原包/原工程/历史 checkpoint。

本次交付包含重新编写的三网络实现、评分/采样器、可确定的损失组件、加载/评价入口，以及实际 CPU 验证。此前已经看过昨天候选代码；权重不能唯一确定的前向语义显式承接其已验证约定。**这是有证据的再次恢复与实质对照，不是盲逆向，也不是原训练源码找回。**

## 原件身份与直接重新取证

- 唯一输入：指定的 `nexus_overfit_data_results_no_code.zip`。完整授权路径见 `FILE_MANIFEST.json`。
- 大小：241,939,514 bytes；SHA256：`{manifest['archive_sha256']}`，与此前完全同包。
- 全部1,330成员逐项CRC及SHA256重读；总展开270,079,268 bytes；无路径穿越、符号链接、重复名。
- 原包内22个`.pt`全部重新解析，使用严格白名单pickle GLOBAL映射到惰性张量描述；只读取数据，不执行包中源码。未为审计复制巨型权重。
- `ORIGINAL_CHECKPOINT_METADATA.json`记录每个原件顶层字段、全部张量形状/存储hash、优化器槽/RNG。`ORIGINAL_TRAINING_RECORDS.json`保存44个原配置/报告/训练日志及其原件hash。
- 昨天证据中列出的10个checkpoint，本次原件hash全部一致，可比state shape全部一致。原包没有新增源文件或隐藏的新训练器。

## 本次实际恢复代码

`reconstructed_from_original/`包含6个Python文件及运行说明、完整来源等级表、验证输入和结果。`recovered_networks.py`、`topology_scores.py`、`recovered_objectives.py`是本次实际重新编写的实现；`verify_recovery.py`、`verify_objectives.py`、`replay_cascade.py`是实际运行入口。

| 网络 | 原件直接证明的结构 | 本次严格加载 | 当前恢复边界 |
|---|---|---|---|
| 拓扑AE | 8层、宽128、latent64、39维位置入口；244个状态项 | 全244项同名覆盖，missing/unexpected均0；3,476,032参数 | 推理前向与全50重建已实测；4heads/pre-norm/Fourier等是已验证候选语义；完整训练器未恢复 |
| 顶点条件拓扑Flow | 10层DiT、宽144、latent64、位置入口39；112项 | 全112项同名覆盖；3,809,008参数 | 固定输入速度场、50步采样和全50串联已实测；RoPE/time/AdaLN精确语义不是权重唯一证据 |
| 文本条件点网络 | 18层DiT、宽144、2048文本、274有序slots、275类count；先验2048→512→822；206项 | 全206项同名覆盖；8,668,301参数 | 文本缓存→点、先验+小残差、100步采样已实测；raw text编码器未随包提供 |

点残差scale直接从原件读取为 `1.2198240256111603e-5`。旧JSON里的6层是历史配置，最终应读checkpoint的18层。最终精度主要来自文本坐标先验对50训练提示的记忆，不能视为纯点diffusion独自收敛或新文本泛化证明。

AE本次改用标准 `torch.nn.TransformerEncoderLayer` 重建原件兼容层；键名天然相同，没有忽略或转换key。完整双向identity映射见 `STATE_KEY_MAPPING.json`。原形状/字段为A级证据，原文字配置为B级，承接并再验证的候选为C级，仍缺为U级；详见 `CONFIG_PROVENANCE.json`。

## 本次代码的实际数值检查

运行环境：CPU、1线程、torch `{verify['torch']}`，MHA fastpath关闭。所用六个远程原件先逐文件与本次ZIP成员hash核对。

- **全部50条AE**：Edge TP/FP/FN={ae['edge']['tp']}/{ae['edge']['fp']}/{ae['edge']['fn']}，F1={ae['edge']['f1']:.10f}；Face TP/FP/FN={ae['face']['tp']}/{ae['face']['fp']}/{ae['face']['fn']}，F1={ae['face']['f1']:.10f}。49/50完全重建；49/50面集合等于老师保存输出。
- **与昨天实现同运行时对照**：50/50 AE edge和face embedding逐元素完全相同；50/50面集合相同。说明本次原生Transformer层复建与昨日手写前向等价，不代表历史训练已恢复。
- **两个固定输入/固定噪声检查**：CAD02和CAD00；点34567、拓扑12345，各样本分别初始化generator；拓扑使用恢复原顺序的GT顶点。新旧点采样最大差1.8626451e-9，拓扑latent最大差9.5367432e-7，两条预测面集合相同。CAD02 FaceF1=1；CAD00 FaceF1=0.6697247706。这个CAD00值属于该单样本GT顶点上下文，不能冒称历史全50串联结果。
- **损失组件**：四组BCE、KL每元素后取mean、拓扑velocity MSE、点velocity MSE/count CE及组合、cached-prior MSE，在固定合成输入上与昨日组件差值全0；BCE/KL梯度有限。仅证明可微公式一致，未运行optimizer。

### 本次全50完整串联

点seed34567、拓扑seed12345，各generator仅初始化一次，顺序00→49，点100步+拓扑50步；不复用旧点目录。全部50条先从文本缓存和噪声生成点，再从生成点与噪声生成拓扑，保存新预测后才载入GT和昨天预测评价。未输入GT点/面到生成器、未修补mesh。

- Face TP/FP/FN：**{score['tp']}/{score['fp']}/{score['fn']}**，micro-F1 **{score['f1']:.10f}**；joint strict={cascade['total']['joint_strict']}/50。
- 点数准确率100%；全局匹配坐标RMSE={point['coordinate_rmse']:.10g}，最大误差/最小间距={point['max_spacing_ratio']:.10g}。
- 生成耗时{cascade['generation_seconds']:.2f}秒；50个新预测及逐mesh记录见 `cascade_cpu_34567_12345/`。
- 昨天固定第一组CPU串联F1=0.9894037356（5509/51/67），torch2.10.0+cpu；本次与该历史记录面集合相同{cascade['total']['exact_face_sets_vs_yesterday']}/50。两个runtime不同，同名seed不保证算子和噪声跨版本逐位相同，不能把历史差异全部归因本次代码。逐mesh差异与旧记录均保留；不挑选第二个更好seed。

## 原训练状态，文件名不能代替身份

| 原件 | 本次直接核实 | 含义 |
|---|---|---|
| `minkowski_target_099/best_ae.pt`、`vae.pt`、`latest.pt:vae` | 244个状态项的shape、dtype、存储hash逐项一致 | 最终AE权重确实在包内；这三处都不是最终AE优化器 |
| `minkowski_target_099/latest.pt` | stage=flow，112个Adam槽与Flow完整有序参数shape对应；moment两个槽都匹配；每槽step14000 | 保存当前Flow优化器；不是AE优化器。总Flow counter128000与当前Adam counter不同 |
| 同上RNG | python/numpy/torch/cuda四项均存在 | 是最终Flow阶段RNG快照，不可当作AE结束时随机状态 |
| `minkowski_continue_01/training_state.pt` | stage=flow，4层Flow，52个优化器槽；累计step14000，当前槽step10000，含四类RNG | 这是早期Flow续训态，也没有最终AE optimizer |
| `point_diffusion/latest.pt` | 推理state_dict/config/说明，无optimizer/RNG | 不能直接等价恢复原点训练 |
| `point_prior_candidate/candidate.pt` | 与最终点206个状态项全部hash相同；6个prior张量+1个scale优化器槽；无RNG | 保存先验末阶段优化器，不是整个18层去噪器优化器 |

优化器只存数字ID，没有显式parameter name。本次利用完整shape序列+stage+模型证据确认归属；重复shape本身不能唯一证明注册次序。prior两个group最终LR分别5.4e-8和5.4e-7、weight_decay0；7槽step50000对应当前优化器生命周期，`prior_steps=160000`是累计先验更新数。原config明确depth/stage变化用fresh AdamW，因此这些计数不同本身不是bug。

原件`candidate.source_step=267500`与最终推理`frozen_denoiser_source_step=258000`仍存在未解释元数据差异。原time-coverage日志确有258000评价，但未找到足以唯一连接两个字段的原文件；不强行判错或拼造历史。昨天README/REPORT已指出这些问题，本次是原件独立复核。

## 本次重新整理出的训练线索与仍缺的地方

原history的AE评价深度依次2层(10000–60000)、4层(62000–90000)、6层(92000–100000)、8层(102000–130000)。Flow依次4层(14000–94000)、6层(96000–104000)、8层(106000–114000)、10层(116000–128000)。这些是记录到的评价范围，不是未经证明的准确初始化时点。

原日志还明确：AE加深2→4→6→8；8层阶段LR先降到5e-5，120000评价后再降到2.5e-5；118000处恢复并启用困难样本采样，记载保留完整状态以及`pre_hard_example_sampling.pt`旧备份，但该备份不在本ZIP内。配置写50%原均匀组、50%从完整面错误最多的两个样本中均匀选取。原Sampler函数、batch构成细节和更新次序未提供，不能据这一句话冒造唯一训练循环。

原点日志保留6→8→10→12→14→16→18层、precision阶段LR2e-6、训练t改为100个Euler查询网格及LR5e-7、后续2.5e-7等线索。先验history记录cached-clean目标下降和160000累计步，但原缓存生成、原冻结代码及之前完整denoiser optimizer/RNG不在包内。

已恢复的loss是四组BCE、标准正态KL基本式、flow线性路径velocity MSE、点X1转velocity加count CE、候选cached-prior MSE。仍缺：负三元组原采样器/顺序，训练dropout，KL归约/后验采样clamp，count系数，精确扩深初始化/阶段切换，原点cached-frozen-output构造。早期config给kl_weight1e-6等不等价于最终训练器完整恢复。

## 与昨天交付相比

没有拿到新老师包。昨天的结构/优化器缺口判断得到重新确认；本次新增的是完整22个checkpoint原件级证据、全状态逐张量hash关系、原日志的阶段/LR出处，以及重新编写并实际跑过的三网络代码、全50 AE和固定全50串联结果。逐文件/函数一致与差异见 `COMPARISON_WITH_YESTERDAY.md`。原论文完整octree系统、raw-text encoder以及从零训练成功均不在本次已完成范围。

与当前512/固定100mesh路线的因果边界和结构对照由另一位Astra/xhigh完成，见 [FIXED100_FAILURE_ANALYSIS.md](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/fixed100_teacher_comparison_20260923/FIXED100_FAILURE_ANALYSIS.md)；不把当前20k训练与老师逐阶段130k混作同预算复现。根代理总汇见 [REPORT.md](/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_reverse_comparison_delivery_20260923/REPORT.md)。

最小后续核验是补齐老师原训练脚本/负例函数、AE阶段结束或pre-hard-example状态、点denoiser来源checkpoint和原缓存；现有证据无法从最终权重唯一逆推出它们。无需以新训练来伪装补齐这些缺口。
'''
    (OUT/'ORIGINAL_ZIP_REVERSE_AUDIT.md').write_text(report)
    files={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='RECOVERY_OUTPUT_MANIFEST.json' and '__pycache__' not in p.parts}
    (OUT/'RECOVERY_OUTPUT_MANIFEST.json').write_text(json.dumps({'author_model':'gpt-6-astra','reasoning_effort':'xhigh','files':files},indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'files':len(files),'face_f1':score['f1'],'point_rmse':point['coordinate_rmse'],'optimizer_updates':0},indent=2))

if __name__=='__main__':main()
