"""Finalize CPU-audited reports and a weight-free evidence archive; no model execution."""
from pathlib import Path
import hashlib,json,shutil,zipfile,subprocess
R=Path(__file__).resolve().parent;O=R/'repro_outputs'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,text):(O/name).write_text(text)
audit=json.loads((O/'ACCEPTANCE.json').read_text());assert audit['passed']
launch=json.loads((O/'launch.json').read_text())
runtime_states=list((O/'_runtime/train').glob('*/state.json'));assert len(runtime_states)==1
state=json.loads(runtime_states[0].read_text());assert state['status']=='success',state
status=json.loads((O/'status.json').read_text())
status['runtime']=dict(state_path=str(runtime_states[0]),state='success',scope='H only; S reused')
(O/'status.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
write('COMMANDS.md',f'''# 实际执行命令

全部命令均由本轮用户授权协议适配；不是作者官方训练命令。工作目录：`{R}`。

```sh
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python preflight_cpu.py
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python H_paper_hard4/test_hard4.py
/opt/conda/bin/python launch_h.py
CUDA_VISIBLE_DEVICES= /opt/conda/bin/python audit_results.py
```

launch_h.py实际通过既有RigorPilot run-train执行：
```sh
{launch['command']}
```

训练子进程固定CUDA_VISIBLE_DEVICES={launch['gpu_uuid']}；物理GPU1。原始argv、启动前GPU进程、PID见launch.json；完整运行时spec/state/stdout/stderr/events/resources见`_runtime/train`。资源日志是设备整体采样，不等于独占归因。

H新增100次更新；S没有新启动。CPU审计新增optimizer更新0，不调用CUDA。完整model/Adam/RNG的读取/恢复命令和固定参数保留在train.py；不使用推理权重加fresh Adam。
''')
write('LOG.md','''# 证据日志

1. 只读查到旧Soft4/stopgrad两支已完成，未修改它们。旧S实际100步对照通过CPU资格核验，复用它而不重复训练。
2. 用户确认新端口可见GPU0和1均可使用；本轮只使用GPU1，启动前没有计算进程。
3. 对新目录loss改动执行CPU小测试，全部通过。原Soft4路径值与梯度逐位一致；Hard4跨chunk FP32归约误差在已记录容差内。
4. H从指定B2500恢复全部model/Adam/RNG；0步50条完整预测与父记录逐项一致。teacher_cad50_20上原Soft4完整网络梯度与旧S记录逐位一致，随后恢复模型模式及RNG；启动检查更新0。
5. H只运行100次全50条累积更新，每25步真实网络完整评价。完整原始训练日志保留在_runtime下；逐步记录在H_paper_hard4/updates.jsonl。
6. 训练结束后，CPU审计重数500份保存预测、核验10份checkpoint/状态与源哈希。此项是本次事后复算，未伪称为原训练时额外日志。

有效代码入口和diff已保存。用户提及的参考附件未出现在可访问附件中；本轮以用户完整给定公式为依据，没有冒称读取或采用作者官方源码。
''')
write('SCIENTIFIC_CHANGELOG.md','''# 科学定义改动

唯一训练干预是Edge和Face重建loss：fully-differentiable Soft4 → TP/TN/FP/FN硬四组BCE。硬组由logit.detach()>0与GT确定，BCE及各组分子保持可微。整条mesh跨chunk汇总N/C，各组N/max(C,1)，四组之和固定除4。空组0是用户明确指定的本轮约定。

没有改动数据、样本权重、Face pool、网络、评分scale/符号/阈值、中心化、Adam历史、LR、clip、更新频率、μ路径、sampling/KL、logvar冻结、math00/FP32 MATH与重计算上下文。H继承相同B2500完整状态；S复用已完成的合格100步对照。

本实验不是Soft4-stopgrad对照，不把Hard4的改变归因于单独删除Soft4第二项。100步不是充分收敛预算；继承旧Adam下的方向结果不代表随机初始化训练上限。
''')
write('COMPARABILITY_REPORT.md','''# 可比性与证据范围

- 对照复用：CONTROL_REUSE_AUDIT.json核对父完整model、Adam一阶/二阶矩/step/参数组、RNG、参与进度、数据/pool哈希、代码及18个实际依赖文件。S新增100次已在旧实验完成，本轮S新增0。
- H与S各自从同一B2500到2600；每步UID顺序相同，50条累积一次Adam。锁定字段逐项比较见ACCEPTANCE.json。
- 旧S和新H在不同物理A100上执行，型号及Python/torch/CUDA/cuDNN一致；H的0步全部50条预测与父记录逐项一致，teacher_cad50_20上同权重原Soft4完整网络梯度与旧S记录逐位一致。该梯度核验只覆盖这一个UID，没有声称全50条梯度逐位对照。
- 模型eval模式、μ路径、确定性Graph及FP32 MATH前向/重计算保持不变。历史通用loss函数仍可能存在于依赖文件，实际objective调用新目录中的Hard4函数；代码路径和启动调用记录已核对。
- 只按同预算实际重建、困难16条及严格成功覆盖比较。每步训练loss仅作各自轨迹；两种loss数值不可直接排名。
- 完整Face只在0/25/50/75/100检查；逐步Edge不能替代逐步实际Face。
- 原始GPU执行日志与本次CPU事后复算分开保留。保存预测重数不等于又执行一次完整网络前向。
''')
git_record=json.loads((O/'CODE_PROVENANCE.json').read_text())
write('PATCHES.md',f'''# 代码改动记录

独立分支：`repro/2026-09-22-cad-hard4`；训练实现commit：`{git_record['training_commit']}`。

风险：loss切换属于改变科学定义的实验干预，已由本轮用户明确授权；不是暗中修复原模型。所有改动只在本轮独立目录。原工程README、旧Soft4/stopgrad、原100条与源checkpoint保持不动。

有效差异见EFFECTIVE_CODE_DIFF.patch：新增Hard4组统计及归约，接入Edge/Face objective；起点检查仅移除“不同loss应数值相同”的旧断言，保留全部预测/状态检查；以复用对照的核验替代双新进程屏障。预算、反传、clip、Adam及checkpoint/eval顺序不变。

验证：hard4_test.json、CONTROL_REUSE_AUDIT.json、H_paper_hard4/startup_gate.json与ACCEPTANCE.json。原README意图没有被冒称为论文结果复现；本README仅描述用户批准的有限实验。
''')

# Copy immutable fixed inputs once; do not rebuild or mutate the source pool.
fresh=R.parent/'teacher_cad50_fresh512_20260921'
inputs=R/'fixed_inputs';inputs.mkdir(exist_ok=True)
for name in ['data','pools']:shutil.copytree(fresh/name,inputs/name,dirs_exist_ok=True)
shutil.copy2(fresh/'pool_manifest.json',inputs/'pool_manifest.json')
# Include the exact pre-existing skill runtime used to supervise this run.
runtime=R.parent/'decoder_last3_trainability_pair_A500_20260921/_rigorpilot'
dest=R/'skill_runtime_source'
shutil.copytree(runtime,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
write('PACKAGE_SCOPE.md','''# 包内容

包含本轮S/H有效代码及动态依赖源码快照、差异、实际配置、命令、原始日志、skill运行时证据、数据/pool、500份完整检查点预测、每步与逐mesh指标、困难16条单列结果和验收报告。S为复制的只读旧对照材料，REUSED_CONTROL.json记录原始身份。

完整大权重不放入ZIP；父、最高Face F1、最高严格成功及末尾完整model/Adam/RNG已保留服务器。CHECKPOINT_MANIFEST.json和BEST_CHECKPOINTS.json给出绝对路径、大小、SHA256及角色。解包路径中的历史依赖目录只用于解释运行时来源，不声称已在另一台机器上完成可移植重放。

不包含其他旧实验/CAD以外数据、凭据、完整conda/CUDA安装、pyc、临时文件或ZIP本身。实际依赖环境版本已记录。zip内FILE_MANIFEST.json可逐文件校验。
''')
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
(O/'RESOURCE_AT_COMPLETION.json').write_text(json.dumps(dict(gpu_state=gpu,compute_processes=procs,our_training_runtime_state=state['status']),indent=2)+'\n')
context=dict(user_language='zh',status='success',selected_goal='training',lane='trusted',
    documented_command='/opt/conda/bin/python launch_h.py',documented_command_section='命令与资源',
    documented_command_source_file='README.md',completed_steps=100,local_dataset_present=True,
    result_match=dict(status='not_evaluated'),next_action='预算已完成；停止，无后继训练',
    readme_commands=[dict(command='/opt/conda/bin/python launch_h.py',section='命令与资源',category='training')])
(O/'annotation_context.json').write_text(json.dumps(context,ensure_ascii=False,indent=2)+'\n')
print('REPORTS_READY',flush=True)
